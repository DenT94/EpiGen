"""Structure viewer: 3D WT/edit/candidate structures with the edit and
compensatory positions highlighted.

Reads the last completed run from `st.session_state.epigen_result` (set by
`app_pages/design.py` -- run a design first). WT, edit-only, and the top
MCMC candidate already have real folded structures from that run; every
other MCMC candidate is sequence-only until folded here on demand (an extra
Modal call), same on-demand pattern as Design's "Describe" SAE step.
"""

from __future__ import annotations

import streamlit as st

from epigen.pipeline.fold_invert_refold.run import fold_sequence
from epigen.pipeline.sae_diff.structural_viz import render_structure_html

# Solarized accents, consistent with .streamlit/config.toml.
EDIT_COLOR = "#dc322f"  # red -- the fixed disruptive edit
COMPENSATORY_COLOR = "#b58900"  # yellow -- compensatory mutation vs edit-only
BASE_COLOR = "#93a1a1"  # gray -- everything else


@st.cache_data(show_spinner=False)
def _cached_fold(sequence: str, seed: int):
    return fold_sequence(sequence, seed=seed)


def _diff_positions(reference: str, other: str, positions: list[int]) -> list[int]:
    """1-indexed positions in `positions` where `other` differs from `reference`."""
    return [p for p in positions if reference[p - 1] != other[p - 1]]


def _render(structure, color_map: dict[int, str], chain_id: str, *, height: int = 480) -> None:
    html = render_structure_html(structure, color_map, chain_id=chain_id)
    st.html(html, unsafe_allow_javascript=True)


def _legend(*, edit: bool = True, compensatory: bool = False) -> None:
    with st.container(horizontal=True):
        if edit:
            st.badge("Edit", icon=":material/dangerous:", color="red")
        if compensatory:
            st.badge("Compensatory mutation", icon=":material/build:", color="orange")
        st.badge("Unchanged", icon=":material/circle:", color="gray")


result = st.session_state.get("epigen_result")
inputs = st.session_state.get("epigen_inputs")

st.subheader(":material/view_in_ar: Structure viewer", divider="gray")

if result is None or inputs is None:
    st.info(
        "Run a design on the Design page first -- the structure viewer needs a completed run.",
        icon=":material/info:",
    )
    st.page_link("app_pages/design.py", label="Go to Design", icon=":material/edit_note:")
    st.stop()

edit_positions = inputs["edit_positions"]
window_positions = inputs["window_positions"]
chain_id = inputs["chain_id"]
seed = inputs["seed"]

view_options = [":material/biotech: WT", ":material/edit: Edit-only"]
if result.top_candidate is not None:
    view_options.append(":material/check_circle: Top candidate (compensated)")
other_candidates = [
    c
    for c in result.mcmc_candidates
    if result.top_candidate is None or c.sequence != result.top_candidate.candidate.sequence
]
if other_candidates:
    view_options.append(":material/hub: Other MCMC candidate")

view = st.segmented_control("View", view_options, default=view_options[0], label_visibility="collapsed")

if view is None:
    st.caption("Pick a view above.")

elif view.endswith("WT"):
    st.caption(f"WT structure ({result.original.source}) with the edit's target positions highlighted.")
    _legend(edit=True, compensatory=False)
    color_map = {p: EDIT_COLOR for p in edit_positions}
    _render(result.original.structure, color_map, chain_id)

elif view.endswith("Edit-only"):
    st.caption(f"Edit applied, no compensation yet (pLDDT={result.edit_only.plddt:.3f}).")
    _legend(edit=True, compensatory=False)
    color_map = {p: EDIT_COLOR for p in edit_positions}
    _render(result.edit_only.structure, color_map, chain_id)

elif view.endswith("Top candidate (compensated)"):
    tc = result.top_candidate
    compensatory_positions = _diff_positions(result.edit_only.sequence, tc.candidate.sequence, window_positions)
    st.caption(
        f"Winning candidate, refolded (pLDDT={tc.folded.plddt:.3f}, "
        f"TM-score vs edit-only={tc.tm_score:.3f}, "
        f"{len(compensatory_positions)} compensatory mutation(s))."
    )
    _legend(edit=True, compensatory=True)
    color_map = {p: EDIT_COLOR for p in edit_positions} | {p: COMPENSATORY_COLOR for p in compensatory_positions}
    _render(tc.folded.structure, color_map, chain_id)

else:  # "Other MCMC candidate"
    candidate_choice = st.selectbox(
        "Candidate",
        other_candidates,
        format_func=lambda c: f"{c.sequence[:24]}...  (combined_score={c.combined_score:.3f})",
    )
    compensatory_positions = _diff_positions(result.edit_only.sequence, candidate_choice.sequence, window_positions)
    st.caption(
        f"{len(compensatory_positions)} compensatory mutation(s) vs edit-only. "
        "Not part of the automatic pipeline (only the top candidate gets refolded/TM-gated) -- "
        "folding this one is an extra on-demand Modal call."
    )
    if st.button("Fold & view this candidate", icon=":material/view_in_ar:"):
        with st.spinner("Folding candidate (Modal)..."):
            try:
                folded = _cached_fold(candidate_choice.sequence, seed)
            except Exception as exc:
                st.exception(exc)
                st.stop()
        st.caption(f"pLDDT={folded.plddt:.3f} (no self-consistency TM-score -- this candidate wasn't refold-gated).")
        _legend(edit=True, compensatory=True)
        color_map = {p: EDIT_COLOR for p in edit_positions} | {p: COMPENSATORY_COLOR for p in compensatory_positions}
        _render(folded.structure, color_map, chain_id)
