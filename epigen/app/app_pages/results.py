"""EpiGen results page: detailed tables/plots for the last completed run.

Reads `st.session_state.epigen_result`/`epigen_inputs` (set by the Landing
page's form submit, or loaded from the sidebar's "past experiments" picker,
see `run_history.py`) -- doesn't run the pipeline itself. Split out of what
used to be the Design page so Landing stays just the input form; see
`app_pages/structure.py` for the 3D viewer on the same session state.
"""

from __future__ import annotations

import streamlit as st

from epigen.pipeline.literature import attach_papers, get_accession_metadata, plot_annotation_map
from epigen.pipeline.naming import mutation_name
from epigen.pipeline.oracle.plot import plot_score_comparison

st.subheader(":material/table_chart: Results", divider="gray")

result = st.session_state.get("epigen_result")
inputs = st.session_state.get("epigen_inputs")

if result is None or inputs is None:
    st.info(
        "Run a design on the Landing page first -- Results needs a completed run.",
        icon=":material/info:",
    )
    st.page_link("app_pages/landing.py", label="Go to Landing", icon=":material/edit_note:")
    st.stop()

wt_sequence = inputs["wt_sequence"]
edit_start = inputs["edit_start"]
edit_sequence_clean = inputs["edit_sequence"]
edit_positions = inputs["edit_positions"]
window_positions = inputs["window_positions"]
pdb_id = inputs.get("pdb_id")
chain_id = inputs["chain_id"]

st.subheader(":material/view_in_ar: Structure source", divider="gray")
with st.container(horizontal=True):
    with st.container(border=True):
        st.caption("WT")
        if result.original.pdb_id:
            st.badge(f"PDB {result.original.pdb_id}", icon=":material/database:", color="blue")
        else:
            st.metric("pLDDT (folded)", f"{result.original.plddt:.3f}", border=False)
        st.caption(f"source: {result.original.source}")
    with st.container(border=True):
        st.caption(f"Edit-only (pos {edit_start} = {edit_sequence_clean})")
        st.metric("pLDDT", f"{result.edit_only.plddt:.3f}", border=False)
st.page_link(
    "app_pages/structure.py",
    label="View these structures in 3D, with the edit and compensatory positions highlighted",
    icon=":material/view_in_ar:",
)

st.subheader(":material/menu_book: Literature annotation map", divider="gray")
if result.annotation_ranges:
    if result.annotation_conflicts:
        st.warning(
            f"Edit position or compensatory window overlaps {len(result.annotation_conflicts)} "
            "known functional/structural annotation(s) -- see ⚠ rows below.",
            icon=":material/warning:",
        )
    st.pyplot(
        plot_annotation_map(
            len(wt_sequence),
            result.annotation_ranges,
            edit_position=(edit_positions[0], edit_positions[-1]),
            window_positions=window_positions,
            conflicts=result.annotation_conflicts,
        )
    )
    functional_ranges = [r for r in result.annotation_ranges if r.kind == "functional"]
    if functional_ranges:
        st.caption(
            "Best-effort supporting literature for the functional annotations above "
            "(Paperclip full-text search, not verified citations)."
        )
        if st.button("Find supporting papers", icon=":material/search:"):
            with st.spinner("Searching Paperclip for supporting literature..."):
                metadata = get_accession_metadata(wt_sequence, pdb_id=pdb_id)
                annotated = attach_papers(functional_ranges, metadata) if metadata else None
            if metadata is None:
                st.warning(
                    "Could not resolve a UniProt accession for this sequence; skipping paper search.",
                    icon=":material/warning:",
                )
            else:
                for r in annotated:
                    with st.container(border=True):
                        st.write(f"**{r.label}** (residues {r.start}-{r.end})")
                        if not r.papers:
                            st.caption("No matching papers found.")
                        for p in r.papers:
                            st.markdown(f"- [{p.title}]({p.url}) — {p.authors} ({p.source}, {p.date})")
else:
    st.caption("No Paperclip/UniProt annotations found for this sequence.")

st.subheader(":material/verified: Oracle sanity checks", divider="gray")
c1, c2 = st.columns(2)
c1.metric(
    "Expert correlation (ESM2 vs ProteinMPNN)",
    f"{result.expert_correlation:.3f}",
    icon=":material/scatter_plot:",
    border=True,
)
c2.metric(
    "Window substitutions below WT",
    f"{result.fraction_below_wt:.1%}",
    icon=":material/trending_down:",
    border=True,
)

st.subheader(":material/bar_chart: Chain scores: starting vs. ending points vs. WT", divider="gray")
st.caption(
    "Free (no extra Modal call): reuses the ESM2+ProteinMPNN per-position tables above to "
    "score every chain's starting and ending sequence, and WT, on the same scale MCMC itself "
    "optimized (Evo2's whole-sequence term isn't included -- see oracle/plot.py)."
)
st.pyplot(plot_score_comparison(result.wt_score, result.chain_starting_scores, result.chain_ending_scores))

st.subheader(f":material/hub: MCMC candidates (top {len(result.mcmc_candidates)})", divider="gray")
st.dataframe(
    [
        # Named against true WT, not edit-only -- so the edit itself shows up in the name
        # (e.g. "V20WHSPRAL") instead of a candidate with no extra compensatory mutation
        # collapsing to a vacuous "identical".
        {
            "mutation": mutation_name(wt_sequence, c.sequence),
            "sequence": c.sequence,
            "combined_score": c.combined_score,
        }
        for c in result.mcmc_candidates
    ],
    width="stretch",
    column_config={
        "combined_score": st.column_config.NumberColumn("Combined score", format="%.3f"),
    },
)

if result.top_candidate is None:
    st.warning("No candidates produced -- nothing to refold or diff.", icon=":material/warning:")
else:
    st.subheader(":material/check_circle: Top candidate: refold + self-consistency", divider="gray")
    tc = result.top_candidate
    st.code(tc.candidate.sequence, language=None)
    with st.container(horizontal=True):
        st.metric("pLDDT", f"{tc.folded.plddt:.3f}", icon=":material/analytics:", border=True)
        st.metric("TM-score vs edit-only", f"{tc.tm_score:.3f}", icon=":material/compare_arrows:", border=True)
        if tc.passed_self_consistency_gate:
            st.badge("Passed self-consistency gate", icon=":material/check:", color="green")
        else:
            st.badge("Failed self-consistency gate", icon=":material/close:", color="red")

    if result.top_candidate_evidence is not None:
        st.subheader(":material/psychology: Agent explanation", divider="gray")
        st.caption(
            "Grounds a plain-language verdict in the contact/SAE/TM-score evidence above; "
            "a deterministic check cross-verifies the agent's claims against the raw numbers "
            "before anything is shown (calls Claude, not free/instant -- on demand)."
        )
        if st.button("Explain top candidate", icon=":material/psychology:"):
            with st.spinner("Asking the explanation agent..."):
                try:
                    from epigen.pipeline.explain import explain_candidate

                    explanation, grounding_check = explain_candidate(result.top_candidate_evidence)
                except Exception as exc:
                    st.exception(exc)
                else:
                    verdict_color = {
                        "rescues": "green",
                        "partial_rescue": "orange",
                        "does_not_rescue": "red",
                        "inconclusive": "gray",
                    }.get(explanation.verdict, "gray")
                    st.badge(explanation.verdict.replace("_", " "), color=verdict_color)
                    st.write(f"**{explanation.headline}**")
                    st.write(explanation.narrative)
                    st.caption(f"Caveats: {explanation.caveats}")
                    if grounding_check.contradicts_evidence:
                        st.error(
                            "Internal grounding check flagged a contradiction between the "
                            "agent's claims and the raw evidence:\n\n"
                            + "\n".join(f"- {m}" for m in grounding_check.mismatches),
                            icon=":material/report:",
                        )
                    else:
                        st.badge(
                            "Grounding check passed -- claims match the raw evidence",
                            icon=":material/check:",
                            color="green",
                        )

    # Raw contact/SAE deltas aren't very informative read directly -- they're inputs the
    # agent explanation above already grounds its claims in, not a primary display in their
    # own right. Kept, but tucked into a collapsed expander rather than shown by default.
    with st.expander(f":material/scatter_plot: Raw evidence tables ({len(result.contact_deltas)} contact-delta rows)"):
        st.caption("Contact microenvironment deltas (edit-only vs top candidate, every changed position).")
        st.dataframe([vars(d) for d in result.contact_deltas], width="stretch")

        top_sae_diff = result.sae_diffs.get(tc.candidate.sequence)
        if top_sae_diff is not None:
            st.caption("SAE feature diff (compensated vs edit-only, top 20 by |delta|).")
            from epigen.pipeline.sae_diff.run import top_k_deltas

            top_deltas = top_k_deltas(top_sae_diff.compensated_vs_edit, k=20)
            st.dataframe([vars(d) for d in top_deltas], width="stretch")

    if result.sae_diffs:
        st.subheader(
            f":material/bubble_chart: SAE feature space across candidates ({len(result.sae_diffs)} scored)",
            divider="gray",
        )
        from epigen.pipeline.sae_diff.describe import describe_candidate
        from epigen.pipeline.sae_diff.pca import build_feature_matrix, pca_2d, select_top_features

        candidate_sequences = list(result.sae_diffs.keys())
        diffs = list(result.sae_diffs.values())
        top_features_per_candidate = select_top_features(diffs, k=3)
        _, matrix = build_feature_matrix(top_features_per_candidate)
        coords = pca_2d(matrix)
        pca_df = {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "mutation": [mutation_name(wt_sequence, seq) for seq in candidate_sequences],
        }
        st.caption("PCA of each candidate's top-3 ΔΔSAE (compensated vs WT) features, unioned across candidates.")
        st.scatter_chart(pca_df, x="PC1", y="PC2")

        st.caption(
            "Describe a candidate's top SAE features with human-readable labels "
            "(esmc_6b/layer60 -- heavier, on-demand only)."
        )
        describe_choice = st.selectbox(
            "Candidate to describe",
            candidate_sequences,
            format_func=lambda s: mutation_name(wt_sequence, s),
        )
        if st.button("Describe", icon=":material/description:"):
            with st.spinner("Re-diffing at the describable SAE config (esmc_6b)..."):
                try:
                    described = describe_candidate(wt_sequence, result.edit_only.sequence, describe_choice)
                except Exception as exc:
                    st.exception(exc)
                else:
                    st.dataframe(
                        [
                            {
                                "position": d.position,
                                "feature_index": d.feature_index,
                                "delta": d.delta,
                                **described.descriptions.get(d.feature_index, {}),
                            }
                            for d in described.top_deltas
                        ],
                        width="stretch",
                    )

                    if describe_choice == tc.candidate.sequence:
                        from epigen.pipeline.sae_diff.structural_viz import feature_color_map, render_structure_html

                        st.caption("Color the top candidate's structure by one of the features above.")
                        feature_choice = st.selectbox(
                            "Feature to color by",
                            [d.feature_index for d in described.top_deltas],
                            format_func=lambda fi: f"{fi}: {described.descriptions.get(fi, {}).get('label', '(no label)')}",
                        )
                        color_map = feature_color_map(described.diff.compensated, feature_choice)
                        html = render_structure_html(tc.folded.structure, color_map, chain_id=chain_id)
                        # st.html(..., unsafe_allow_javascript=True) silently drops py2Dmol's
                        # ~100KB inline rendering script when it re-executes scripts client-side
                        # (only the small ones survive) -- an iframe srcdoc via components.v1.html
                        # has no such re-execution/size limit. See structure.py's `_render`.
                        st.components.v1.html(html, height=480, scrolling=False)
                    else:
                        st.caption(
                            "Structural coloring is only available for the top candidate "
                            "(the only one that's actually been refolded into a real structure)."
                        )
