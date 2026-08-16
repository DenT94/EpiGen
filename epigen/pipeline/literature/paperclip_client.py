"""Thin wrapper around the `paperclip` CLI's `/proteins/` SQL surface (UniProt + PDB).

Shells out rather than hitting an API directly -- Paperclip auth/session state
lives with the CLI (see the `paperclip` skill), and the protein tables are
only exposed via `paperclip sql -s proteins "SELECT ..."`. Output is a plain
pipe-table on stdout; this module is the only place that parses it, so the
rest of `literature/` deals in plain Python values.
"""

from __future__ import annotations

import logging
import subprocess
import time

logger = logging.getLogger(__name__)

PAPERCLIP_TIMEOUT_S = 30
PAPERCLIP_MAX_ATTEMPTS = 3
PAPERCLIP_RETRY_DELAY_S = 2.0


class PaperclipError(RuntimeError):
    """The `paperclip` CLI failed or returned something this module can't parse."""


def run_paperclip(args: list[str], *, timeout: float = PAPERCLIP_TIMEOUT_S) -> str:
    """Run a `paperclip` CLI subcommand (e.g. `["sql", "-s", "proteins", "SELECT ..."]`,
    `["search", "-s", "pmc", ..., query]`) and return its stdout, retrying transient
    failures up to `PAPERCLIP_MAX_ATTEMPTS` times.

    Observed live: the *exact same* query sometimes returns in ~200ms and sometimes
    hangs to `timeout` or comes back with the CLI's own "[error] Request timed out",
    back to back, with no change on our end -- a flaky backend, not a bad query. A
    couple of short-delay retries papers over that instead of every caller
    (`run_protein_sql`, `literature.papers._search`) silently degrading a transient
    blip into "no annotations"/"no papers found". `FileNotFoundError` (the `paperclip`
    binary itself missing) is not retried -- that's not transient.
    """
    last_error: str = ""
    for attempt in range(1, PAPERCLIP_MAX_ATTEMPTS + 1):
        try:
            result = subprocess.run(["paperclip", *args], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            last_error = f"timed out after {timeout}s: {exc}"
        except FileNotFoundError as exc:
            raise PaperclipError(f"could not run `paperclip {' '.join(args)}`: {exc}") from exc
        else:
            if result.returncode == 0:
                return result.stdout
            last_error = f"exited {result.returncode}: {result.stderr.strip()}"

        if attempt < PAPERCLIP_MAX_ATTEMPTS:
            logger.warning(
                f"`paperclip {' '.join(args)}` failed (attempt {attempt}/{PAPERCLIP_MAX_ATTEMPTS}): "
                f"{last_error} -- retrying in {PAPERCLIP_RETRY_DELAY_S}s"
            )
            time.sleep(PAPERCLIP_RETRY_DELAY_S)

    raise PaperclipError(f"`paperclip {' '.join(args)}` failed after {PAPERCLIP_MAX_ATTEMPTS} attempts: {last_error}")


def _parse_pipe_table(stdout: str) -> list[dict[str, str]]:
    """Parse `paperclip sql`'s `col | col | ...` table into row dicts.

    Format (see `paperclip sql --help` / live testing): a header row, a
    `---+---` separator, one row per line, then a trailing `(N rows, ...)`
    summary line. An empty result is just that summary line with no header.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines or lines[0].startswith("("):
        return []
    header = [col.strip() for col in lines[0].split("|")]
    rows = []
    for line in lines[2:]:  # skip header + '---+---' separator
        if line.startswith("("):
            break
        values = [cell.strip() for cell in line.split("|")]
        rows.append(dict(zip(header, values)))
    return rows


def run_protein_sql(query: str) -> list[dict[str, str]]:
    """Run a read-only SELECT against Paperclip's protein tables (`uniprot_v.*`, `pdb_v.*`).

    Returns one dict per row, string-valued (the CLI's table output has no
    type info -- callers cast as needed). Raises `PaperclipError` on CLI
    failure (after retries -- see `run_paperclip`); a query that
    legitimately matches nothing returns `[]`.
    """
    stdout = run_paperclip(["sql", "-s", "proteins", query], timeout=PAPERCLIP_TIMEOUT_S)
    return _parse_pipe_table(stdout)
