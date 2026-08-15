"""Cross-candidate SAE feature vectorization + PCA.

mypipelinethoughts.md step 5: "take all the top 3 saes for all candidates,
see all the K unique SAE features activated and have each candidate mapped
into a K-dim ΔΔSAE vector" -- then PCA-scatter candidates against each
other. Runs entirely locally (no Modal calls); operates on
`ThreeStateSAEDiff`s already produced by `sae_diff.run.diff_many_candidates`.
"""

from __future__ import annotations

import numpy as np

from epigen.pipeline.sae_diff.run import FeatureDelta, ThreeStateSAEDiff, top_k_deltas

# A feature is identified by (position, feature_index) -- the same SAE
# feature index means different things at different residues, so both
# coordinates are needed to name a "unique feature" the way the doc means it.
FeatureKey = tuple[int, int]


def select_top_features(diffs: list[ThreeStateSAEDiff], k: int = 3) -> list[list[FeatureDelta]]:
    """Top-`k` `compensated_vs_original` deltas per candidate.

    `compensated_vs_original` is the literal "WT vs MU_STAR" comparison
    mypipelinethoughts.md calls ΔΔSAE (`edit_vs_original`, WT vs MU, is ΔSAE).
    """
    return [top_k_deltas(diff.compensated_vs_original, k=k) for diff in diffs]


def build_feature_matrix(per_candidate_top_features: list[list[FeatureDelta]]) -> tuple[list[FeatureKey], np.ndarray]:
    """Union the top features across candidates into a shared K-dim space.

    Returns the ordered list of K unique `(position, feature_index)` keys
    and an N (candidates) x K matrix of each candidate's ΔΔSAE `delta` for
    that key -- 0.0 for a candidate whose own top-k list never included that
    key. Most entries being 0 is expected: a candidate only "owns" a cell
    for the handful of features it actually contributed to the union.
    """
    keys: dict[FeatureKey, None] = {}  # insertion-ordered set
    for top_features in per_candidate_top_features:
        for delta in top_features:
            keys[(delta.position, delta.feature_index)] = None
    key_list = list(keys)
    key_index = {key: i for i, key in enumerate(key_list)}

    matrix = np.zeros((len(per_candidate_top_features), len(key_list)))
    for row, top_features in enumerate(per_candidate_top_features):
        for delta in top_features:
            matrix[row, key_index[(delta.position, delta.feature_index)]] = delta.delta
    return key_list, matrix


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """2D PCA projection of an N x K matrix, for a candidate scatterplot.

    Returns an N x 2 array. `n_components` is capped at
    `min(2, n_samples, n_features)` -- PCA can't produce more components
    than that, which matters for tiny candidate counts (e.g. `candidate_num=1`
    during a quick smoke test); missing columns are zero-filled so callers
    always get a 2-column array back.
    """
    from sklearn.decomposition import PCA

    n_components = min(2, matrix.shape[0], matrix.shape[1])
    if n_components < 1:
        return np.zeros((matrix.shape[0], 2))
    coords = PCA(n_components=n_components).fit_transform(matrix)
    if n_components < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - n_components)))
    return coords
