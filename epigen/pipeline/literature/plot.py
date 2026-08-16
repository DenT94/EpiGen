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
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 200,
        }
    )
    conflict_ids = {id(r) for r in (conflicts or [])}
    ordered = sorted(ranges, key=lambda r: (r.start, r.end))

    n_rows = len(ordered)
    fig_height = 1.4 + 0.32 * max(n_rows, 1)
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
        fontsize=9,
        fontweight="bold",
    )

    if window_positions:
        w_lo, w_hi = min(window_positions), max(window_positions)
        ax.axvspan(w_lo - 0.5, w_hi + 0.5, color=WINDOW_COLOR, alpha=0.12, zorder=1, lw=0)
        ax.text(
            (w_lo + w_hi) / 2, backbone_y + 0.85, "compensatory window",
            ha="center", va="bottom", fontsize=7.5, color=WINDOW_COLOR, style="italic",
        )
    if isinstance(edit_position, tuple):
        e_lo, e_hi = edit_position
        if e_lo == e_hi:
            ax.axvline(e_lo, color=EDIT_COLOR, lw=1.4, ls="--", zorder=4)
        else:
            ax.axvspan(e_lo - 0.5, e_hi + 0.5, color=EDIT_COLOR, alpha=0.22, zorder=2, lw=0)
        ax.text((e_lo + e_hi) / 2, backbone_y + 0.5, "edit", ha="center", va="bottom",
                fontsize=7.5, color=EDIT_COLOR, fontweight="bold")
    elif edit_position is not None:
        ax.axvline(edit_position, color=EDIT_COLOR, lw=1.4, ls="--", zorder=4)
        ax.text(edit_position, backbone_y + 0.5, "edit", ha="center", va="bottom",
                fontsize=7.5, color=EDIT_COLOR, fontweight="bold")

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
            ax.plot([r.start, r.end], [y + 0.3, y + 0.3], color=line_color, lw=lw, zorder=3)
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
        ax.text(r.start - sequence_length * 0.008, y + 0.3, f"{prefix}{r.label}", ha="right", va="center", fontsize=7.6)
        pos_label = f"{r.start}" if r.start == r.end else f"{r.start}–{r.end}"
        ax.text(r.start + width + sequence_length * 0.008, y + 0.3, pos_label, ha="left", va="center",
                fontsize=6.8, color="#666666")

    ax.set_xlim(1 - sequence_length * 0.28, sequence_length * 1.06)
    ax.set_ylim(-0.2, backbone_y + 0.9)
    ax.set_yticks([])
    ax.set_xlabel("residue position (construct numbering)")
    ax.set_title(
        f"Literature annotations for {construct_label} ({sequence_length} aa) — "
        f"{sum(1 for r in ordered if r.kind == 'functional')} functional, "
        f"{sum(1 for r in ordered if r.kind == 'structural')} structural",
        loc="left", fontweight="bold",
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
    # Placed in the left margin (data x < 0), which is otherwise blank at every row --
    # unlike a lower/upper corner, this can't collide with a feature row or the
    # edit/window labels regardless of how many rows or where the window falls.
    ax.legend(handles=legend_handles, loc="center left", frameon=False, fontsize=7.5)

    fig.tight_layout()
    return fig
