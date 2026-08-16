"""Checkpoint/resume support for `oracle.mcmc.run_mcmc_search`.

`run_mcmc_search` holds all chain state in memory and, without this, only
returns once -- at the very end of `steps` rounds. Any crash mid-search
(Modal timeout, network blip, OOM, ...) loses the *entire* run's compute,
not just the tail (this is exactly what happened to the ~2500-chain,
50-step run that died at ~2h against the old 30min Modal timeout -- see
`oracle/modal_app.py`'s `MCMC_SEARCH_TIMEOUT_S` comment).

This module periodically snapshots enough state to resume a search from
its last completed round instead of starting over: every chain's sequence,
score, nt sequence, RNG state (so resuming reproduces the same random draws
a continuous run would have made), and frozen/active flag, plus the shared
best-candidates tables. `run_mcmc_search` calls `save`/`load` directly; the
caller (e.g. `oracle/modal_app.py`) only needs to pick a `checkpoint_dir`
that survives a crash (a Modal Volume mount) and, if using one, pass
`on_checkpoint=volume.commit` so each write is actually durable outside the
crashing container -- Modal Volumes buffer writes locally until committed.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STATE_FILENAME = "state.pkl"
_PROGRESS_FILENAME = "progress.json"


@dataclass
class ChainCheckpoint:
    sequence: str
    score: float
    nt_sequence: str | None
    rng_state: tuple[Any, ...]  # `random.Random.getstate()`, restored via `.setstate()`
    active: bool  # False once this chain froze (see mcmc.run_mcmc_search's docstring)


@dataclass
class CheckpointState:
    """Everything needed to resume `run_mcmc_search` from `round` onward.

    `window_positions`/`num_chains`/`use_evo2` aren't needed to resume the
    search math itself (that's all in `chains`) -- they're a compatibility
    check, so resuming with a call whose config doesn't match the
    checkpoint's fails loudly instead of silently continuing a different
    search under the same `checkpoint_dir`.
    """

    round: int
    window_positions: list[int]
    num_chains: int
    use_evo2: bool
    chains: list[ChainCheckpoint]
    best_score_by_sequence: dict[str, float]
    best_nt_by_sequence: dict[str, str | None]
    starting_sequences_per_chain: list[str]
    wt_score: float


def save(checkpoint_dir: str, state: CheckpointState) -> None:
    """Write `state`, atomically (temp file + rename, so a crash mid-write can't leave a
    corrupt checkpoint `load` would choke on) plus a small human-readable `progress.json`
    sidecar for cheap polling (round/best-so-far) without unpickling the full chain state."""
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)

    state_path = directory / _STATE_FILENAME
    tmp_path = state_path.with_suffix(".pkl.tmp")
    tmp_path.write_bytes(pickle.dumps(state))
    tmp_path.replace(state_path)  # atomic on POSIX

    top_candidates = sorted(state.best_score_by_sequence.items(), key=lambda kv: kv[1], reverse=True)[:5]
    progress = {
        "round": state.round,
        "num_chains": state.num_chains,
        "wt_score": state.wt_score,
        "converged_chain_count": sum(1 for c in state.chains if not c.active),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "top_candidates": [{"sequence": seq, "score": score} for seq, score in top_candidates],
    }
    (directory / _PROGRESS_FILENAME).write_text(json.dumps(progress, indent=2))


def load(checkpoint_dir: str) -> CheckpointState | None:
    """`None` if there's no checkpoint yet, or if the file is unreadable/corrupt (e.g. a
    crash during a previous write somehow evaded the atomic rename) -- either way the
    caller should just start fresh rather than fail the whole run over a checkpoint that
    isn't safely resumable."""
    state_path = Path(checkpoint_dir) / _STATE_FILENAME
    if not state_path.exists():
        return None
    try:
        return pickle.loads(state_path.read_bytes())
    except (pickle.UnpicklingError, EOFError, OSError, AttributeError):
        return None
