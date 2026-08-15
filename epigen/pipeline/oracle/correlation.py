"""Sanity checks on the two-expert oracle, per mypipelinethoughts.md step 4.

"Scores are saved separately: important to check that the correlation
between the experts is low" (they should be adding independent information,
not both re-deriving the same signal), and "print out percentage of edited
sequences that are below WT" as a sanity check before treating any candidate
as promising.
"""

from __future__ import annotations

import numpy as np

from epigen.pipeline.oracle.scoring import CANONICAL_AA, PositionScores


def _delta_vs_wt(scores: PositionScores, wt_sequence: str, positions: list[int]) -> np.ndarray:
    """Flattened (position x AA) log-likelihood deltas vs the WT AA's own score, at `positions`."""
    deltas = []
    for pos in positions:
        wt_aa = wt_sequence[pos - 1]
        row = scores[pos - 1]
        wt_ll = row.get(wt_aa, 0.0)
        deltas.extend(row.get(aa, 0.0) - wt_ll for aa in CANONICAL_AA if aa in row)
    return np.array(deltas)


def expert_agreement(
    esm2_scores: PositionScores,
    pmpnn_scores: PositionScores,
    wt_sequence: str,
    positions: list[int],
) -> float:
    """Pearson correlation between the two experts' delta-log-likelihood-vs-WT, over `positions`.

    Returns 0.0 (rather than NaN) if either expert's deltas are constant
    (e.g. a single-position window with no variation) -- there's no
    correlation to speak of, not an error.
    """
    a = _delta_vs_wt(esm2_scores, wt_sequence, positions)
    b = _delta_vs_wt(pmpnn_scores, wt_sequence, positions)
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def fraction_below_wt(
    esm2_scores: PositionScores,
    pmpnn_scores: PositionScores,
    wt_sequence: str,
    positions: list[int],
    *,
    weight_esm2: float = 0.5,
    weight_pmpnn: float = 0.5,
) -> float:
    """Fraction of (position, AA) substitutions whose combined score is below the WT AA's own
    combined score at that position."""
    total = 0
    below = 0
    for pos in positions:
        wt_aa = wt_sequence[pos - 1]
        esm2_row, pmpnn_row = esm2_scores[pos - 1], pmpnn_scores[pos - 1]
        wt_combined = weight_esm2 * esm2_row.get(wt_aa, 0.0) + weight_pmpnn * pmpnn_row.get(wt_aa, 0.0)
        for aa in CANONICAL_AA:
            if aa == wt_aa or aa not in esm2_row or aa not in pmpnn_row:
                continue
            combined = weight_esm2 * esm2_row[aa] + weight_pmpnn * pmpnn_row[aa]
            total += 1
            if combined < wt_combined:
                below += 1
    return below / total if total else 0.0
