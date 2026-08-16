"""Structure viewer: 3D WT/edit/candidate structures with the edit and
compensatory positions highlighted.

Reads the last completed run from `st.session_state.epigen_result` (set by
`app_pages/landing.py` -- run a design first). WT, edit-only, and the top
MCMC candidate already have real folded structures from that run; every
other MCMC candidate is sequence-only until folded here on demand (an extra
Modal call), same on-demand pattern as Results' "Describe" SAE step.
"""

from __future__ import annotations

import streamlit as st

from epigen.pipeline.fold_invert_refold.run import fold_sequence
from epigen.pipeline.naming import mutation_name
from epigen.pipeline.sae_diff.run import top_k_deltas
from epigen.pipeline.sae_diff.structural_viz import (
    align_to_reference,
    compute_reference_camera,
    feature_color_map,
    render_structure_html,
)

# Solarized accents, consistent with .streamlit/config.toml.
EDIT_COLOR = "#dc322f"  # red -- the fixed disruptive edit
COMPENSATORY_COLOR = "#b58900"  # yellow -- compensatory mutation vs edit-only
BASE_COLOR = "#93a1a1"  # gray -- everything else


@st.cache_data(show_spinner=False)
def _cached_fold(sequence: str, seed: int):
    return fold_sequence(sequence, seed=seed)


def _folded_cache_keys() -> set[tuple[str, int]]:
    """(sequence, seed) pairs already folded via `_cached_fold` this session -- tracked
    ourselves since `st.cache_data` has no public "is this cached" query, only "call it
    (cheaply, on a hit)". Lets the 'Other MCMC candidate' picker default to something
    that renders instantly instead of always defaulting to an uncached candidate."""
    return st.session_state.setdefault("structure_folded_cache_keys", set())


@st.cache_data(show_spinner=False)
def _cached_reference_camera(structure_pdb: str, chain_id: str):
    """WT's fixed py2Dmol camera, cached by its own PDB text -- cheap (local PCA, no
    Modal call) but no reason to recompute it on every Streamlit rerun."""
    from proto_tools.entities.structures import Structure

    return compute_reference_camera(Structure(structure=structure_pdb, structure_format="pdb"), chain_id)


@st.cache_data(show_spinner=False)
def _cached_describe_candidate(wt_sequence: str, edit_only_sequence: str, candidate_sequence: str, k: int):
    """Cached wrapper around `sae_diff.describe.describe_candidate` -- a real Modal call
    (re-diffs at esmc_6b/layer60), so identical args should hit cache instead of re-spending it,
    same as `_cached_fold`."""
    from epigen.pipeline.sae_diff.describe import describe_candidate

    return describe_candidate(wt_sequence, edit_only_sequence, candidate_sequence, k=k)


def _diff_positions(reference: str, other: str, positions: list[int]) -> list[int]:
    """1-indexed positions in `positions` where `other` differs from `reference`."""
    return [p for p in positions if reference[p - 1] != other[p - 1]]


def _top_ddsae_deltas(sae_diff, k: int = 3):
    """Top-k ΔΔSAE deltas for `sae_diff` -- `compensated_vs_original`, the literal WT-vs-MU_STAR
    double diff mypipelinethoughts.md calls ΔΔSAE (see `sae_diff.pca`'s module docstring).
    Already computed for every candidate by the cheap esmc_300m pass in orchestrate.py -- no
    extra Modal call to pick from these, unlike results.py's heavier "Describe" (esmc_6b/layer60)."""
    return top_k_deltas(sae_diff.compensated_vs_original, k=k)


def _render(
    structure,
    color_map: dict[int, str],
    chain_id: str,
    *,
    reference_structure,
    reference_camera,
    height: int = 480,
) -> None:
    """Kabsch-align `structure` onto `reference_structure` and render it with the fixed
    `reference_camera`, so every view in the Structure viewer shares one viewpoint instead
    of each independently computing its own best-fit camera angle (see structural_viz.py's
    module docstring)."""
    aligned = align_to_reference(structure, reference_structure, chain_id)
    html = render_structure_html(aligned, color_map, chain_id=chain_id, reference_camera=reference_camera)
    # st.html(..., unsafe_allow_javascript=True) silently drops py2Dmol's ~100KB inline
    # rendering script when it re-executes scripts client-side (only small scripts survive,
    # so the control panel chrome renders but the actual WebGL viewer never populates -- an
    # empty canvas). An iframe srcdoc has no such limit; confirmed via a live DOM inspection
    # (the big script tag was simply absent from the page). st.iframe (the non-deprecated
    # replacement for st.components.v1.html, removed after 2026-06-01) uses the same
    # srcdoc mechanism for an HTML-string src, so this keeps that fix.
    st.iframe(html, height=height)


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
        "Run a design on the Landing page first -- the structure viewer needs a completed run.",
        icon=":material/info:",
    )
    st.page_link("app_pages/landing.py", label="Go to Landing", icon=":material/edit_note:")
    st.stop()

wt_sequence = inputs["wt_sequence"]
edit_positions = inputs["edit_positions"]
window_positions = inputs["window_positions"]
chain_id = inputs["chain_id"]
seed = inputs["seed"]

# WT is the shared reference every other view aligns onto and borrows the camera from.
reference_camera = _cached_reference_camera(result.original.structure.structure_pdb, chain_id)

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
    _render(
        result.original.structure,
        color_map,
        chain_id,
        reference_structure=result.original.structure,
        reference_camera=reference_camera,
    )

elif view.endswith("Edit-only"):
    st.caption(f"Edit applied, no compensation yet (pLDDT={result.edit_only.plddt:.3f}).")
    _legend(edit=True, compensatory=False)
    color_map = {p: EDIT_COLOR for p in edit_positions}
    _render(
        result.edit_only.structure,
        color_map,
        chain_id,
        reference_structure=result.original.structure,
        reference_camera=reference_camera,
    )

elif view.endswith("Top candidate (compensated)"):
    tc = result.top_candidate
    compensatory_positions = _diff_positions(result.edit_only.sequence, tc.candidate.sequence, window_positions)
    st.caption(
        f"Winning candidate, refolded (pLDDT={tc.folded.plddt:.3f}, "
        f"TM-score vs edit-only={tc.tm_score:.3f}, "
        f"{len(compensatory_positions)} compensatory mutation(s))."
    )
    _legend(edit=True, compensatory=True)
    # Edit/compensatory highlighting always wins over any SAE overlay below -- both overlay
    # blocks merge as `overlay | base_color_map`, never chained onto each other (chaining would
    # make the second overlay a no-op: `feature_color_map` returns a color for *every* position,
    # so it would always lose the merge to an already-full first overlay sitting on the right).
    base_color_map = {p: EDIT_COLOR for p in edit_positions} | {p: COMPENSATORY_COLOR for p in compensatory_positions}
    color_map = base_color_map

    sae_diff = result.sae_diffs.get(tc.candidate.sequence)
    if sae_diff is not None:
        # One control, not two: the numeric (esmc_300m, free) and labeled (esmc_6b/layer60,
        # on-demand Modal call) ΔΔSAE lists live in different feature spaces, so showing both
        # dropdowns at once ("SAE1" meaning two different features depending which one you
        # look at) was confusing. Describing upgrades in place -- once labels are fetched, the
        # numeric dropdown is replaced by the labeled one instead of sitting alongside it.
        describe_key = f"structure_describe_{tc.candidate.sequence}"
        if not st.session_state.get(describe_key) and st.button(
            "Describe (human-readable labels)", icon=":material/description:"
        ):
            st.session_state[describe_key] = True
            st.rerun()

        if st.session_state.get(describe_key):
            with st.spinner("Re-diffing at the describable SAE config (esmc_6b)..."):
                try:
                    described = _cached_describe_candidate(
                        inputs["wt_sequence"], result.edit_only.sequence, tc.candidate.sequence, 3
                    )
                except Exception as exc:
                    st.exception(exc)
                    described = None
            if described is not None:
                labeled_options = ["None"] + [
                    f"SAE{i + 1}: {described.descriptions.get(d.feature_index, {}).get('label', '(no label)')}"
                    for i, d in enumerate(described.top_deltas)
                ]
                labeled_choice = st.selectbox(
                    "Color by ΔΔSAE feature (esmc_6b/layer60, labeled)", labeled_options
                )
                if labeled_choice != "None":
                    d = described.top_deltas[labeled_options.index(labeled_choice) - 1]
                    desc = described.descriptions.get(d.feature_index, {})
                    st.caption(
                        f"feature {d.feature_index} @ position {d.position} (ΔΔ={d.delta:+.3f}): "
                        f"{desc.get('description', '(no description)')}"
                    )
                    color_map = feature_color_map(described.diff.compensated, d.feature_index) | base_color_map
        else:
            top_deltas = _top_ddsae_deltas(sae_diff, k=3)
            sae_choice = st.selectbox(
                "Color by ΔΔSAE feature (esmc_300m, numeric)",
                ["None"] + [f"SAE{i + 1}" for i in range(len(top_deltas))],
                help="Overlays one of the candidate's top-3 ΔΔSAE (compensated vs WT) feature "
                "activations -- white -> red by magnitude, wherever that feature is active in "
                "the compensated structure. Edit/compensatory positions stay highlighted on "
                "top. No extra Modal call -- reuses the cheap esmc_300m pass already run for "
                "every candidate. Numeric only (feature index, no label); click 'Describe' "
                "above for human-readable labels instead (a different, heavier SAE config).",
            )
            if sae_choice != "None":
                delta = top_deltas[int(sae_choice.removeprefix("SAE")) - 1]
                st.caption(
                    f"{sae_choice}: feature {delta.feature_index} @ position {delta.position} "
                    f"(ΔΔ={delta.delta:+.3f})."
                )
                color_map = feature_color_map(sae_diff.compensated, delta.feature_index) | base_color_map

    _render(
        tc.folded.structure,
        color_map,
        chain_id,
        reference_structure=result.original.structure,
        reference_camera=reference_camera,
    )

else:  # "Other MCMC candidate"
    cache_keys = _folded_cache_keys()
    already_cached = [c for c in other_candidates if (c.sequence, seed) in cache_keys]
    # Default to the top-scoring candidate that's already folded (renders instantly, no Modal
    # call) instead of always defaulting to other_candidates[0] regardless of cache state --
    # other_candidates is already ranked by combined_score, so the first cached one found is
    # also the best-scoring one among those free to show.
    default_choice = already_cached[0] if already_cached else other_candidates[0]

    candidate_choice = st.selectbox(
        "Candidate",
        other_candidates,
        index=other_candidates.index(default_choice),
        format_func=lambda c: (
            f"{mutation_name(wt_sequence, c.sequence)}  (combined_score={c.combined_score:.3f})"
            + ("  [cached]" if (c.sequence, seed) in cache_keys else "")
        ),
    )
    compensatory_positions = _diff_positions(result.edit_only.sequence, candidate_choice.sequence, window_positions)
    is_cached = (candidate_choice.sequence, seed) in cache_keys
    st.caption(
        f"{len(compensatory_positions)} compensatory mutation(s) vs edit-only. "
        "Not part of the automatic pipeline (only the top candidate gets refolded/TM-gated)"
        + ("." if is_cached else " -- folding this one is an extra on-demand Modal call.")
    )

    # Cached candidates render immediately (no button needed, it's a free cache hit);
    # uncached ones still require an explicit click before spending a Modal call.
    show = is_cached or st.button("Fold & view this candidate", icon=":material/view_in_ar:")
    if show:
        with st.spinner("Folding candidate (Modal)..."):
            try:
                folded = _cached_fold(candidate_choice.sequence, seed)
            except Exception as exc:
                st.exception(exc)
                st.stop()
        cache_keys.add((candidate_choice.sequence, seed))
        st.caption(f"pLDDT={folded.plddt:.3f} (no self-consistency TM-score -- this candidate wasn't refold-gated).")
        _legend(edit=True, compensatory=True)
        color_map = {p: EDIT_COLOR for p in edit_positions} | {p: COMPENSATORY_COLOR for p in compensatory_positions}
        _render(
            folded.structure,
            color_map,
            chain_id,
            reference_structure=result.original.structure,
            reference_camera=reference_camera,
        )
