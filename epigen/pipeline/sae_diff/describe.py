"""Hybrid-checkpoint description pass: human-readable labels for one chosen candidate.

The broad multi-candidate pass (`sae_diff.run.diff_many_candidates`) uses
the cheap `esmc_300m` checkpoint, which has no published feature
descriptions. Real labels only exist for one specific SAE
(`esmc_6b`/layer60/k64/codebook16384 -- `sae_diff.run.DESCRIBABLE_CONFIG`),
which is much heavier, so this is deliberately a separate, on-demand,
single-candidate call -- not run for every candidate by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from epigen.pipeline.alignment import PositionMap, identity_map
from epigen.pipeline.sae_diff.run import (
    DESCRIBABLE_CONFIG,
    FeatureDelta,
    ThreeStateSAEDiff,
    describe_top_features,
    diff_three_states,
    top_k_deltas,
)


@dataclass(frozen=True)
class DescribedCandidate:
    """One candidate's top-k ΔΔSAE features, at the describable SAE config, with labels."""

    candidate_sequence: str
    top_deltas: list[FeatureDelta]  # compensated_vs_original, at DESCRIBABLE_CONFIG
    descriptions: dict[int, dict[str, Any]]  # feature_index -> {label, description, category, ...}
    diff: ThreeStateSAEDiff  # full diff at DESCRIBABLE_CONFIG -- e.g. `.compensated` feeds structural_viz.feature_color_map


def describe_candidate(
    wt_sequence: str,
    edit_only_sequence: str,
    candidate_sequence: str,
    *,
    k: int = 3,
    position_map: PositionMap = identity_map(),
) -> DescribedCandidate:
    """Re-diff one candidate at the describable SAE config and label its top-k ΔΔSAE features.

    This re-runs the diff from scratch at `esmc_6b`/layer60 (3 Modal calls:
    original/edit-only/compensated) -- nothing from a prior `esmc_300m` pass
    is reusable, since the two checkpoints have entirely different feature
    spaces. That's expected: this function is meant to be called for one (or
    a small handful of) chosen candidates, not every MCMC candidate.
    """
    diff = diff_three_states(
        wt_sequence,
        edit_only_sequence,
        candidate_sequence,
        model_checkpoint=DESCRIBABLE_CONFIG["model_checkpoint"],
        layer=DESCRIBABLE_CONFIG["layers"][0],
        position_map=position_map,
    )
    top_deltas = top_k_deltas(diff.compensated_vs_original, k=k)
    descriptions = describe_top_features(
        top_deltas,
        model_checkpoint=DESCRIBABLE_CONFIG["model_checkpoint"],
        layer=DESCRIBABLE_CONFIG["layers"][0],
        k=DESCRIBABLE_CONFIG["k"],
        codebook_size=DESCRIBABLE_CONFIG["codebook_size"],
    )
    return DescribedCandidate(candidate_sequence=candidate_sequence, top_deltas=top_deltas, descriptions=descriptions, diff=diff)
