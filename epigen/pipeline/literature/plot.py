"""Static annotation-track figure: one row per functional/structural range, plus the
edit position and compensatory window overlaid.

Same visual idiom as the reference map (a backbone bar, one
labeled row per feature below it, colored by category) -- just for a single
construct instead of a multi-protein, and with the edit-window
overlay a design tool needs that a literature figure doesn't.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from epigen.pipeline.literature.annotations import AnnotationRange

FUNCTIONAL_COLOR = "#b03a2e"
STRUCTURAL_COLOR = "#1f4e9c"
EDIT_COLOR = "#e8891a"
WINDOW_COLOR = "#e8891a"
CONFLICT_EDGE_COLOR = "#111111"
BACKBONE_COLOR = "#c9c9c9"

# UniProt feature_types whose start/end are two specific paired residues (e.g. a
# disulfide's two cysteines), not a contiguous functional span -- a filled bar across
# the whole range reads as "every residue in between is part of this," which is wrong
# for these two. Drawn as a bracket (two end-cap ticks + a thin connecting line)
# instead -- see the `is_bond` branch in plot_annotation_map below.
BOND_FEATURE_TYPES = {"Disulfide bond", "Cross-link"}


def plot_annotation_map(
    sequence_length: int,
    ranges: list[AnnotationRange],
    *,
    edit_position: int | tuple[int, int] | None = None,
    window_positions: list[int] | None = None,
    conflicts: list[AnnotationRange] | None = None,
    construct_label: str = "construct",
) -> Figure:
    """One row per `ranges` entry (sorted by start), colored functional/structural,
    with a full-length backbone bar on top. `edit_position` is a single 1-indexed
    position (a plain vertical marker) or a `(start, end)` tuple for a multi-residue
    edit (shaded like `window_positions`, since a dashed line can't represent a span);
    `window_positions` shades the compensatory search window; `conflicts` (a subset of
    `ranges`, e.g. `orchestrate.EndToEndResult.annotation_conflicts`) get a bold outline.
    """
    # Bumped for readability. Done via rcParams (not just doubling individual fontsize=
    # calls below, which is how this got left inconsistent the first pass) so
    # axes.labelsize/xtick.labelsize/ytick.labelsize -- which the xlabel and tick labels
    # actually use, and which default to *relative* keywords like "medium" off font.size --
    # scale along with everything else instead of silently staying at the old small size.
    # See oracle/plot.py's plot_score_comparison for the same fix, same rationale.
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
    conflict_ids = {id(r) for r in (conflicts or [])}
    ordered = sorted(ranges, key=lambda r: (r.start, r.end))

    n_rows = len(ordered)
    # + 1.2in fixed: room for the legend now living below the axes (bbox_to_anchor,
    # see the fig.legend call below) plus the xlabel -- fig.tight_layout() doesn't
    # reliably account for a legend anchored outside the axes bbox, so that space is
    # reserved explicitly via fig.subplots_adjust(bottom=...) below instead.
    fig_height = 1.4 + 0.32 * max(n_rows, 1) + 1.2
    fig, ax = plt.subplots(figsize=(11, fig_height))

    backbone_y = n_rows + 0.9
    ax.add_patch(
        Rectangle((1, backbone_y - 0.3), sequence_length, 0.6, fc=BACKBONE_COLOR, ec="white", lw=1.0, zorder=3)
    )
    ax.text(
        1 - sequence_length * 0.01,
        backbone_y,
        construct_label,
        ha="right",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    if window_positions:
        w_lo, w_hi = min(window_positions), max(window_positions)
        ax.axvspan(w_lo - 0.5, w_hi + 0.5, color=WINDOW_COLOR, alpha=0.12, zorder=1, lw=0)
        ax.text(
            (w_lo + w_hi) / 2, backbone_y + 1.45, "compensatory window",
            ha="center", va="bottom", fontsize=15, color=WINDOW_COLOR, style="italic",
        )
    if isinstance(edit_position, tuple):
        e_lo, e_hi = edit_position
        if e_lo == e_hi:
            ax.axvline(e_lo, color=EDIT_COLOR, lw=1.4, ls="--", zorder=4)
        else:
            ax.axvspan(e_lo - 0.5, e_hi + 0.5, color=EDIT_COLOR, alpha=0.22, zorder=2, lw=0)
        ax.text((e_lo + e_hi) / 2, backbone_y + 0.5, "edit", ha="center", va="bottom",
                fontsize=15, color=EDIT_COLOR, fontweight="bold")
    elif edit_position is not None:
        ax.axvline(edit_position, color=EDIT_COLOR, lw=1.4, ls="--", zorder=4)
        ax.text(edit_position, backbone_y + 0.5, "edit", ha="center", va="bottom",
                fontsize=15, color=EDIT_COLOR, fontweight="bold")

    for i, r in enumerate(ordered):
        y = n_rows - i - 1
        color = FUNCTIONAL_COLOR if r.kind == "functional" else STRUCTURAL_COLOR
        is_conflict = id(r) in conflict_ids
        width = max(r.end - r.start + 1, sequence_length * 0.006)
        if r.feature_type in BOND_FEATURE_TYPES and r.start != r.end:
            # |----| bracket: two end-cap ticks at the paired residues, joined by a thin
            # line -- not a filled Rectangle, which would visually claim every residue
            # between r.start and r.end is part of the bond (see BOND_FEATURE_TYPES).
            line_color = CONFLICT_EDGE_COLOR if is_conflict else color
            lw = 2.0 if is_conflict else 1.6
            ax.plot([r.start, r.end], [y + 0.4, y + 0.4], color=line_color, lw=lw, zorder=3)
            for x in (r.start, r.end):
                ax.plot([x, x], [y, y + 0.6], color=line_color, lw=lw, zorder=3)
        else:
            ax.add_patch(
                Rectangle(
                    (r.start, y), width, 0.6,
                    fc=color, ec=CONFLICT_EDGE_COLOR if is_conflict else "none",
                    lw=1.6 if is_conflict else 0, zorder=3,
                )
            )
        ax.plot([r.start, r.start], [0.0, backbone_y - 0.3], color="#e6e6e6", lw=0.6, ls=(0, (2, 2)), zorder=1)
        prefix = "⚠ " if is_conflict else ""
        ax.text(r.start - sequence_length * 0.008, y + 0.4, f"{prefix}{r.label}", ha="right", va="center", fontsize=15)
        pos_label = f"{r.start}" if r.start == r.end else f"{r.start}–{r.end}"
        ax.text(r.start + width + sequence_length * 0.008, y + 0.4, pos_label, ha="left", va="center",
                fontsize=14, color="#666666")

    ax.set_xlim(1 - sequence_length * 0.28, sequence_length * 1.06)
    ax.set_ylim(-0.5, backbone_y + 1.7)
    ax.set_yticks([])
    ax.set_xlabel("residue position (construct numbering)")
    ax.set_title(
        f"Literature annotations for {construct_label} ({sequence_length} aa) — "
        f"{sum(1 for r in ordered if r.kind == 'functional')} functional, "
        f"{sum(1 for r in ordered if r.kind == 'structural')} structural",
        # Explicit fontsize smaller than rcParams' axes.titlesize (20): this title's text
        # is unusually long for a plot title, and at the full rcParams size it runs off
        # the figure's fixed 11in width -- shrunk locally rather than lowering
        # axes.titlesize globally, which would undersize every other (shorter) title
        # using this same rcParams setup (e.g. oracle/plot.py's).
        loc="left", fontweight="bold", y=1.05, fontsize=15,
    )
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)

    legend_handles = [
        Rectangle((0, 0), 1, 1, fc=FUNCTIONAL_COLOR, label="functional"),
        Rectangle((0, 0), 1, 1, fc=STRUCTURAL_COLOR, label="structural"),
    ]
    if conflict_ids:
        legend_handles.append(Rectangle((0, 0), 1, 1, fc="white", ec=CONFLICT_EDGE_COLOR, lw=1.6,
                                         label="overlaps edit/window"))
    # Below the axes, horizontally centered: at the pre-readability-pass font size, the
    # left margin (data x < 0) had room for both this and the row labels living there
    # ("center left" placement, vertically centered in the full ylim); at the current
    # larger font the legend box is tall/wide enough to collide with whichever row
    # happens to fall near vertical-center (row labels, "construct", or both -- not
    # dependent on n_rows, so raising it here rather than shrinking the legend back down).
    #
    # fig.legend (figure-fraction bbox_to_anchor), not ax.legend (axes-fraction): an
    # axes-fraction offset is relative to the *axes'* height, which grows with n_rows --
    # a many-row protein (e.g. lysozyme's ~27 annotation rows) has a much taller axes than
    # a 3-row test case, so the same axes-fraction offset put the legend a huge, growing
    # absolute distance below the plot instead of a small fixed one. Figure-fraction keeps
    # it a constant distance from the bottom of the figure regardless of n_rows.
    fig.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles), frameon=False,
    )

    fig.tight_layout()
    # Explicit, after tight_layout (which would otherwise fight this): reserves the
    # +1.3in added to fig_height above for the legend below the axes -- tight_layout's
    # own automatic bottom margin doesn't know about a legend anchored outside the axes
    # bbox, so left alone the legend just draws past the figure's bottom edge into (or
    # past) the xlabel whenever the save/render path doesn't crop with bbox_inches="tight"
    # (st.pyplot doesn't).
    fig.subplots_adjust(bottom=1.2 / fig_height)
    return fig
