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

logger = logging.getLogger(__name__)

PAPERCLIP_TIMEOUT_S = 30


class PaperclipError(RuntimeError):
    """The `paperclip` CLI failed or returned something this module can't parse."""


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
    failure; a query that legitimately matches nothing returns `[]`.
    """
    try:
        result = subprocess.run(
            ["paperclip", "sql", "-s", "proteins", query],
            capture_output=True,
            text=True,
            timeout=PAPERCLIP_TIMEOUT_S,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise PaperclipError(f"could not run `paperclip sql`: {exc}") from exc
    if result.returncode != 0:
        raise PaperclipError(f"`paperclip sql` exited {result.returncode}: {result.stderr.strip()}")
    return _parse_pipe_table(result.stdout)
