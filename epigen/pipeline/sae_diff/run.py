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
    """Fetch one sequence's active SAE features at a single layer.

    `layer=None` uses Biohub's published ~75%-depth sweep layer for the
    chosen checkpoint (see `ESMCSAEFeaturesConfig.resolved_layers`).
    """
    config = ESMCSAEFeaturesConfig(
        device=DEVICE,
        model_checkpoint=model_checkpoint,
        layers=None if layer is None else [layer],
        k=k,
        codebook_size=codebook_size,
    )
    output = run_esmc_sae_features(ESMCSAEFeaturesInput(sequences=[sequence]), config)
    layer_features = output.results[0].layers[0]
    return {
        position: dict(zip(indices, magnitudes, strict=True))
        for position, (indices, magnitudes) in enumerate(
            zip(layer_features.feature_indices, layer_features.feature_magnitudes, strict=True), start=1
        )
    }


@dataclass(frozen=True)
class FeatureDelta:
    """One (position, feature) activation change between two states."""

    position: int  # 1-indexed
    feature_index: int
    magnitude_a: float  # 0.0 if the feature wasn't active in state A
    magnitude_b: float  # 0.0 if the feature wasn't active in state B
    delta: float  # magnitude_b - magnitude_a


def diff_feature_vectors(vector_a: FeatureVector, vector_b: FeatureVector) -> list[FeatureDelta]:
    """Per-(position, feature) deltas between two feature vectors, magnitude_b - magnitude_a."""
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
) -> ThreeStateSAEDiff:
    """Fetch SAE features for all three states and diff them pairwise.

    "edit-only" is the scaffold with the lactocepin motif inserted but no
    compensatory mutation; "compensated" adds the candidate compensatory
    mutation on top. `compensated_vs_edit` is usually the most direct read on
    whether the mutation rescues what the insertion disrupted.
    """
    original = get_feature_vector(original_sequence, model_checkpoint=model_checkpoint, layer=layer)
    edit_only = get_feature_vector(edit_only_sequence, model_checkpoint=model_checkpoint, layer=layer)
    compensated = get_feature_vector(compensated_sequence, model_checkpoint=model_checkpoint, layer=layer)
    return ThreeStateSAEDiff(
        original=original,
        edit_only=edit_only,
        compensated=compensated,
        edit_vs_original=diff_feature_vectors(original, edit_only),
        compensated_vs_edit=diff_feature_vectors(edit_only, compensated),
        compensated_vs_original=diff_feature_vectors(original, compensated),
    )


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
