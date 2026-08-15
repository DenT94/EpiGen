"""Local run-history log backing the "past experiments" picker.

`st.cache_data` has no API to list or introspect its own cached keys, so
this keeps a small parallel JSON record (in the repo's gitignored `data/`
dir) of every run's exact call kwargs plus a few display fields. Loading a
past experiment just replays those kwargs through
`pipeline_cache.cached_run_end_to_end` -- a near-instant cache hit if the
disk cache still has it, a normal recompute otherwise.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HISTORY_PATH = Path(__file__).resolve().parents[2] / "data" / "run_history.json"
_MAX_ENTRIES = 50


def _entry_id(kwargs: dict[str, Any]) -> str:
    blob = json.dumps(kwargs, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def record_run(kwargs: dict[str, Any], summary: dict[str, Any]) -> None:
    """Add one run to the history log, or bump it to the top if already logged."""
    entries = [e for e in load_history() if e["id"] != _entry_id(kwargs)]
    entries.insert(
        0,
        {
            "id": _entry_id(kwargs),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kwargs": kwargs,
            "summary": summary,
        },
    )
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_PATH.write_text(json.dumps(entries[:_MAX_ENTRIES], indent=2))


def load_history() -> list[dict[str, Any]]:
    """Most-recent-first list of past runs; `[]` if none logged yet or the file is unreadable."""
    if not _HISTORY_PATH.exists():
        return []
    try:
        return json.loads(_HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def derive_inputs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Build the `st.session_state.epigen_inputs` shape from a run's call kwargs.

    Single source of truth for this derivation -- used both right after a
    fresh Design submit and after loading a past run from history, so both
    paths populate session state identically.
    """
    edit_start = kwargs["edit_start"]
    edit_sequence = kwargs["edit_sequence"]
    return {
        "wt_sequence": kwargs["wt_sequence"],
        "edit_start": edit_start,
        "edit_sequence": edit_sequence,
        "edit_positions": list(range(edit_start, edit_start + len(edit_sequence))),
        "window_positions": kwargs["window_positions"],
        "pdb_id": kwargs.get("pdb_id"),
        "chain_id": kwargs["chain_id"],
        "seed": kwargs.get("seed"),
    }


def format_label(entry: dict[str, Any]) -> str:
    kwargs = entry["kwargs"]
    summary = entry["summary"]
    timestamp = entry["timestamp"].replace("T", " ")[:16]
    seq_preview = kwargs["wt_sequence"][:10]
    edit = f"{kwargs['edit_start']}:{kwargs['edit_sequence']}"
    n_candidates = summary.get("n_candidates", "?")
    wt_len = summary.get("wt_len", "?")
    return f"{timestamp} UTC · {seq_preview}… ({wt_len}aa) · edit {edit} · {n_candidates} candidates"
