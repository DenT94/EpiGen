"""Stage 3: ESM C SAE feature diff across original / edit-only / compensated states.

Runs `esmc-sae-features` (Modal, `device="modal"`) once per sequence state and
diffs the sparse per-residue feature activations pairwise. Scoped to a single
layer and top-k deltas only, per CLAUDE.md's hackathon time budget -- no full
feature atlas.

Note on human-readable feature labels: Biohub's `describe_sae_features` only
has published descriptions for one specific SAE --
`model_checkpoint="esmc_6b"`, `layers=[60]`, `k=64`, `codebook_size=16384`
(see `proto_tools.tools.masked_models.esmc.helpers`). The default config here
uses the much lighter `esmc_300m` checkpoint for fast iteration; switch to
the 6B config (heavier Modal cold start / weight download) if labeled
features are needed for the demo narrative.
"""

from __future__ import annotations

from dataclasses import dataclass

from proto_tools import ESMCSAEFeaturesConfig, ESMCSAEFeaturesInput, run_esmc_sae_features
from proto_tools.tools.masked_models.esmc.helpers import describe_sae_features

from epigen.pipeline.alignment import PositionMap, identity_map

DEVICE = "modal"

# The only SAE configuration with published human-readable feature descriptions.
DESCRIBABLE_CONFIG = {"model_checkpoint": "esmc_6b", "layers": [60], "k": 64, "codebook_size": 16384}

# Position (1-indexed) -> feature index -> activation magnitude.
FeatureVector = dict[int, dict[int, float]]


def get_feature_vector(
    sequence: str,
    *,
    model_checkpoint: str = "esmc_300m",
    layer: int | None = None,
    k: int = 64,
    codebook_size: int = 16384,
) -> FeatureVector:
    """Fetch one sequence's active SAE features at a single layer (one Modal call).

    `layer=None` uses Biohub's published ~75%-depth sweep layer for the
    chosen checkpoint (see `ESMCSAEFeaturesConfig.resolved_layers`).
    """
    return get_feature_vectors_batch([sequence], model_checkpoint=model_checkpoint, layer=layer, k=k, codebook_size=codebook_size)[0]


def get_feature_vectors_batch(
    sequences: list[str],
    *,
    model_checkpoint: str = "esmc_300m",
    layer: int | None = None,
    k: int = 64,
    codebook_size: int = 16384,
) -> list[FeatureVector]:
    """Fetch active SAE features for every sequence, in one Modal call.

    `run_esmc_sae_features`'s `sequences` field is already list-typed
    (`iterable_input_fields=["sequences"]`), so N sequences cost one call
    here instead of N -- this is what makes `diff_many_candidates` cheap
    regardless of how many MCMC candidates there are.
    """
    config = ESMCSAEFeaturesConfig(
        device=DEVICE,
        model_checkpoint=model_checkpoint,
        layers=None if layer is None else [layer],
        k=k,
        codebook_size=codebook_size,
    )
    output = run_esmc_sae_features(ESMCSAEFeaturesInput(sequences=sequences), config)
    vectors = []
    for result in output.results:
        layer_features = result.layers[0]
        vectors.append(
            {
                position: dict(zip(indices, magnitudes, strict=True))
                for position, (indices, magnitudes) in enumerate(
                    zip(layer_features.feature_indices, layer_features.feature_magnitudes, strict=True), start=1
                )
            }
        )
    return vectors


@dataclass(frozen=True)
class FeatureDelta:
    """One (position, feature) activation change between two states."""

    position: int  # 1-indexed
    feature_index: int
    magnitude_a: float  # 0.0 if the feature wasn't active in state A
    magnitude_b: float  # 0.0 if the feature wasn't active in state B
    delta: float  # magnitude_b - magnitude_a


def reindex_to_wt(vector: FeatureVector, position_map: PositionMap) -> FeatureVector:
    """Reindex a candidate-sequence feature vector into WT-native position numbering.

    Positions inside an inserted span (no WT counterpart) are dropped -- this
    is the literal implementation of "if insertion, SAE diff is computed by
    ignoring the edit sequence" (mypipelinethoughts.md step 5): the inserted
    motif's own residues never enter the diff, only the flanking scaffold
    positions do, correctly realigned.
    """
    reindexed: FeatureVector = {}
    for cand_pos, features in vector.items():
        wt_pos = position_map.to_wt(cand_pos)
        if wt_pos is not None:
            reindexed[wt_pos] = features
    return reindexed


def diff_feature_vectors(vector_a: FeatureVector, vector_b: FeatureVector) -> list[FeatureDelta]:
    """Per-(position, feature) deltas between two feature vectors, magnitude_b - magnitude_a.

    Both vectors must already share the same position space (e.g. both
    WT-native -- see `reindex_to_wt` for candidate-sequence vectors that
    aren't).
    """
    positions = sorted(set(vector_a) | set(vector_b))
    deltas: list[FeatureDelta] = []
    for pos in positions:
        features_a = vector_a.get(pos, {})
        features_b = vector_b.get(pos, {})
        for feature_index in sorted(set(features_a) | set(features_b)):
            mag_a = features_a.get(feature_index, 0.0)
            mag_b = features_b.get(feature_index, 0.0)
            if mag_a == mag_b:
                continue
            deltas.append(
                FeatureDelta(position=pos, feature_index=feature_index, magnitude_a=mag_a, magnitude_b=mag_b, delta=mag_b - mag_a)
            )
    return deltas


def top_k_deltas(deltas: list[FeatureDelta], k: int = 20) -> list[FeatureDelta]:
    """The `k` largest deltas by absolute magnitude change."""
    return sorted(deltas, key=lambda d: abs(d.delta), reverse=True)[:k]


@dataclass(frozen=True)
class ThreeStateSAEDiff:
    """Pairwise SAE feature diffs across the original / edit-only / compensated states."""

    original: FeatureVector
    edit_only: FeatureVector
    compensated: FeatureVector
    edit_vs_original: list[FeatureDelta]
    compensated_vs_edit: list[FeatureDelta]
    compensated_vs_original: list[FeatureDelta]


def diff_three_states(
    original_sequence: str,
    edit_only_sequence: str,
    compensated_sequence: str,
    *,
    model_checkpoint: str = "esmc_300m",
    layer: int | None = None,
    position_map: PositionMap = identity_map(),
) -> ThreeStateSAEDiff:
    """Fetch SAE features for all three states and diff them pairwise.

    "edit-only" is the scaffold with the lactocepin motif inserted but no
    compensatory mutation; "compensated" adds the candidate compensatory
    mutation on top. `compensated_vs_edit` is usually the most direct read on
    whether the mutation rescues what the insertion disrupted.

    `position_map` maps edit-only/compensated positions back to
    `original_sequence`'s (WT) numbering -- pass `identity_map()` (default)
    for substitution-only edits, or `insertion_map(...)` when
    edit_only/compensated are longer than `original_sequence`. Positions
    inside the inserted span are excluded from every diff.
    """
    original = get_feature_vector(original_sequence, model_checkpoint=model_checkpoint, layer=layer)
    edit_only_raw = get_feature_vector(edit_only_sequence, model_checkpoint=model_checkpoint, layer=layer)
    compensated_raw = get_feature_vector(compensated_sequence, model_checkpoint=model_checkpoint, layer=layer)
    edit_only = reindex_to_wt(edit_only_raw, position_map)
    compensated = reindex_to_wt(compensated_raw, position_map)
    return ThreeStateSAEDiff(
        original=original,
        edit_only=edit_only,
        compensated=compensated,
        edit_vs_original=diff_feature_vectors(original, edit_only),
        compensated_vs_edit=diff_feature_vectors(edit_only, compensated),
        compensated_vs_original=diff_feature_vectors(original, compensated),
    )


def diff_many_candidates(
    original_sequence: str,
    edit_only_sequence: str,
    candidate_sequences: list[str],
    *,
    model_checkpoint: str = "esmc_300m",
    layer: int | None = None,
    position_map: PositionMap = identity_map(),
) -> list[ThreeStateSAEDiff]:
    """SAE-diff every candidate against the same original/edit-only pair.

    `original`/`edit_only` are fetched once (2 Modal calls, shared across
    every candidate) and every candidate's compensated state is fetched in
    a single batched call (`get_feature_vectors_batch`) -- so scoring
    `len(candidate_sequences)` candidates costs 3 Modal calls total, not
    `3 * len(candidate_sequences)`. This is what makes it practical to
    SAE-diff every MCMC candidate (per mypipelinethoughts.md step 5) rather
    than just the winner.
    """
    original = get_feature_vector(original_sequence, model_checkpoint=model_checkpoint, layer=layer)
    edit_only_raw = get_feature_vector(edit_only_sequence, model_checkpoint=model_checkpoint, layer=layer)
    edit_only = reindex_to_wt(edit_only_raw, position_map)

    compensated_raw_batch = get_feature_vectors_batch(candidate_sequences, model_checkpoint=model_checkpoint, layer=layer)

    diffs = []
    for compensated_raw in compensated_raw_batch:
        compensated = reindex_to_wt(compensated_raw, position_map)
        diffs.append(
            ThreeStateSAEDiff(
                original=original,
                edit_only=edit_only,
                compensated=compensated,
                edit_vs_original=diff_feature_vectors(original, edit_only),
                compensated_vs_edit=diff_feature_vectors(edit_only, compensated),
                compensated_vs_original=diff_feature_vectors(original, compensated),
            )
        )
    return diffs


def describe_top_features(deltas: list[FeatureDelta], *, model_checkpoint: str, layer: int, k: int, codebook_size: int):
    """Look up human-readable labels for the given deltas' feature indices.

    Only works for the one SAE configuration Biohub has published
    descriptions for (`DESCRIBABLE_CONFIG`); raises otherwise so a caller
    can't silently get wrong labels for a different SAE's feature indices.
    """
    actual_config = {"model_checkpoint": model_checkpoint, "layers": [layer], "k": k, "codebook_size": codebook_size}
    if actual_config != DESCRIBABLE_CONFIG:
        raise ValueError(
            f"Feature descriptions are only published for {DESCRIBABLE_CONFIG}, got {actual_config}. "
            "Re-run diff_three_states() with that config to get describable feature indices."
        )
    return describe_sae_features([d.feature_index for d in deltas])
