"""Score-distribution figure: MCMC chain starting/ending points vs. the WT baseline.

Answers "did the search actually move chains to better-scoring sequences than
where they started, and how does either compare to WT?" -- the three
quantities `orchestrate.run_end_to_end` computes for free (no extra Modal
call) via `oracle.mcmc.window_score` reusing the ESM2/ProteinMPNN per-position
tables already spent on the oracle sanity checks (see that module's
`EndToEndResult` fields: `wt_score`, `chain_starting_scores`,
`chain_ending_scores`).
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

STARTING_COLOR = "#1f4e9c"  # blue, matches literature/plot.py's STRUCTURAL_COLOR
ENDING_COLOR = "#2e9e5b"  # green
WT_COLOR = "#b03a2e"  # red, matches literature/plot.py's FUNCTIONAL_COLOR


def plot_score_comparison(wt_score: float, starting_scores: list[float], ending_scores: list[float]) -> Figure:
    """Overlaid histograms of chain starting vs. ending window_score, WT as a dashed vertical line.

    `starting_scores`/`ending_scores` are one per MCMC chain, same order and
    length (`EndToEndResult.chain_starting_scores`/`.chain_ending_scores`) --
    not deduped, so a warm start reused across `chains_per_start` chains
    counts once per chain, matching how many independent trajectories that
    starting point actually seeded.
    """
    # Bumped from font.size=9 for readability (matching literature/plot.py's pass) --
    # done via rcParams rather than doubling individual fontsize= calls, so axes.labelsize/
    # xtick.labelsize/ytick.labelsize/legend.fontsize (which default to *relative* keywords
    # like "medium"/"large" off font.size) scale along with everything else instead of
    # silently staying tiny -- that mismatch is exactly what literature/plot.py's readability
    # pass missed for its own xlabel/ylabel/tick labels.
    mpl.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 200,
        }
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    all_scores = [*starting_scores, *ending_scores, wt_score]
    lo, hi = min(all_scores), max(all_scores)
    pad = (hi - lo) * 0.05 or 1.0
    bins = 20

    ax.hist(starting_scores, bins=bins, range=(lo - pad, hi + pad), color=STARTING_COLOR, alpha=0.55, zorder=2)
    ax.hist(ending_scores, bins=bins, range=(lo - pad, hi + pad), color=ENDING_COLOR, alpha=0.55, zorder=3)
    ax.axvline(wt_score, color=WT_COLOR, lw=1.8, ls="--", zorder=4)
    ax.text(
        wt_score, ax.get_ylim()[1] * 0.98, " WT", ha="left", va="top", fontsize=16, color=WT_COLOR, fontweight="bold"
    )

    ax.set_xlabel("MOE score (higher = better)")
    ax.set_ylabel("chains")
    ax.set_title(
        f"MCMC chain scores: starting vs. ending points ({len(starting_scores)} chains)",
        loc="left",
        fontweight="bold",
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.legend(
        handles=[
            Patch(fc=STARTING_COLOR, alpha=0.55, label="starting points"),
            Patch(fc=ENDING_COLOR, alpha=0.55, label="ending points"),
        ],
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    return fig
