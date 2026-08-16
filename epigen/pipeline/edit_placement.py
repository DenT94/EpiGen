"""mypipelinethoughts.md step 4: "score edits everywhere" / "sequences that are above
WT are sent to a candidates list right away."

This is *not* a deep-mutational-scan over arbitrary single-AA substitutions (that's
`oracle.correlation.fraction_below_wt`'s job, over the compensatory window). It's the
other reading: the same fixed EDIT_SEQUENCE the user chose, tried as a substitution at
every position it could occupy in the WT sequence -- "where else could this exact edit
go, for free (no compensatory mutation needed), because it already scores above WT
there?" Independent of any one `edit_start`.

Cheap because it reuses `oracle.scoring`'s per-position ESM2/ProteinMPNN tables computed
*once* on the unedited WT structure -- valid at every trial position for a substitution,
since (unlike an insertion) placing `edit_sequence` elsewhere doesn't change the
backbone, so no per-position ESMFold2 call is needed (see mypipelinethoughts.md step 4's
own insertion caveat -- that's specifically what doesn't hold for insertions, deferred
per todo.md).

Deterministic given (WT protein, edit type, edit sequence) alone -- doesn't depend on
`edit_start`, the compensatory window, or any MCMC/seed setting -- so it's cached to
disk under `edit_placement/cache/`, one CSV per (protein, edit), entirely separate from
this repo's other caches (`literature.cache`, `st.cache_data`).
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from proto_tools.entities.structures import Structure

from epigen.pipeline.oracle.mcmc import window_score
from epigen.pipeline.oracle.scoring import position_scores_esm2, position_scores_proteinmpnn

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "edit_placement_cache"


@dataclass(frozen=True)
class PlacementScore:
    """One trial placement of the edit sequence, as a substitution starting at `position`."""

    position: int  # 1-indexed start of the substitution
    sequence: str  # full WT-length sequence with edit_sequence substituted in at `position`
    score: float  # combined ESM2+ProteinMPNN score of the edit's own AAs at this span
    wt_score: float  # same span's score for WT's own (un-substituted) AAs
    above_wt: bool  # score > wt_score -- no compensatory mutation needed at this placement


def _protein_key(wt_sequence: str) -> str:
    # Short hash, not the raw sequence -- keeps the cache directory name filesystem-
    # friendly regardless of protein length, while still being deterministic per protein.
    return hashlib.sha1(wt_sequence.encode()).hexdigest()[:12]


def _cache_path(wt_sequence: str, edit_sequence: str, *, edit_type: str = "subs") -> Path:
    return CACHE_DIR / _protein_key(wt_sequence) / f"{edit_type}_{edit_sequence}.csv"


def load(wt_sequence: str, edit_sequence: str, *, edit_type: str = "subs") -> list[PlacementScore] | None:
    """Cached placement scan for (`wt_sequence`, `edit_type`, `edit_sequence`), or `None` on a miss."""
    path = _cache_path(wt_sequence, edit_sequence, edit_type=edit_type)
    if not path.exists():
        return None
    try:
        with path.open(newline="") as f:
            return [
                PlacementScore(
                    position=int(row["position"]),
                    sequence=row["sequence"],
                    score=float(row["score"]),
                    wt_score=float(row["wt_score"]),
                    above_wt=row["above_wt"] == "True",
                )
                for row in csv.DictReader(f)
            ]
    except (OSError, KeyError, ValueError) as exc:
        logger.warning(f"Ignoring unreadable edit-placement cache entry {path}: {exc}")
        return None


def save(wt_sequence: str, edit_sequence: str, placements: list[PlacementScore], *, edit_type: str = "subs") -> None:
    path = _cache_path(wt_sequence, edit_sequence, edit_type=edit_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["position", "sequence", "score", "wt_score", "above_wt"])
        writer.writeheader()
        for p in placements:
            writer.writerow(
                {"position": p.position, "sequence": p.sequence, "score": p.score, "wt_score": p.wt_score, "above_wt": p.above_wt}
            )


def _scan_substitution_placements(
    wt_sequence: str,
    wt_structure: Structure,
    edit_sequence: str,
    *,
    weight_esm2: float = 0.5,
    weight_pmpnn: float = 0.5,
) -> list[PlacementScore]:
    """Uncached scan: every valid substitution start position for `edit_sequence` in
    `wt_sequence`, scored against WT's own per-position ESM2+ProteinMPNN tables (two
    Modal calls total, regardless of protein length or edit length -- not one per
    position). Use `scan_or_load_substitution_placements` instead, which wraps this
    with the disk cache.
    """
    edit_len = len(edit_sequence)
    if edit_len == 0 or edit_len > len(wt_sequence):
        raise ValueError(f"edit_sequence length {edit_len} must be in [1, {len(wt_sequence)}] for {wt_sequence!r}")

    esm2_scores = position_scores_esm2(wt_sequence)
    pmpnn_scores = position_scores_proteinmpnn(wt_structure, wt_sequence)

    placements = []
    for position in range(1, len(wt_sequence) - edit_len + 2):
        span = list(range(position, position + edit_len))
        candidate_sequence = wt_sequence[: position - 1] + edit_sequence + wt_sequence[position - 1 + edit_len :]
        score = window_score(candidate_sequence, span, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
        wt_score = window_score(wt_sequence, span, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
        placements.append(
            PlacementScore(position=position, sequence=candidate_sequence, score=score, wt_score=wt_score, above_wt=score > wt_score)
        )
    return placements


def scan_or_load_substitution_placements(
    wt_sequence: str,
    wt_structure: Structure,
    edit_sequence: str,
    *,
    weight_esm2: float = 0.5,
    weight_pmpnn: float = 0.5,
    force_recompute: bool = False,
) -> list[PlacementScore]:
    """`load()` the cached scan if there is one; otherwise `_scan_substitution_placements`
    (2 Modal calls) and cache the result. `force_recompute=True` bypasses a cache hit
    (e.g. after a WT structure fix) without needing to manually delete the cache file.
    """
    if not force_recompute:
        cached = load(wt_sequence, edit_sequence)
        if cached is not None:
            return cached

    placements = _scan_substitution_placements(
        wt_sequence, wt_structure, edit_sequence, weight_esm2=weight_esm2, weight_pmpnn=weight_pmpnn
    )
    save(wt_sequence, edit_sequence, placements)
    return placements
