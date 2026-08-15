"""EpiGen design page: input form -> get_structure (PDB-first) -> oracle/MCMC
search -> refold+TM-gate -> contact/SAE diffs.

First full end-to-end pass, per mypipelinethoughts.md. Substitution-only MVP
(see todo.md for insertion notes). On a successful run (or a past run loaded
from the sidebar, see `run_history.py`), the result is stashed in
`st.session_state.epigen_result`/`epigen_inputs` so this page can redisplay
it and the Structure viewer page can render it, without re-running the
pipeline.
"""

from __future__ import annotations

import streamlit as st

from epigen.app import run_history
from epigen.app.pipeline_cache import cached_run_end_to_end
from epigen.pipeline.literature import attach_papers, get_accession_metadata, plot_annotation_map


def _render_results(result, inputs: dict) -> None:
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

    st.subheader(f":material/hub: MCMC candidates (top {len(result.mcmc_candidates)})", divider="gray")
    st.dataframe(
        [{"sequence": c.sequence, "combined_score": c.combined_score} for c in result.mcmc_candidates],
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

        st.subheader(
            f":material/scatter_plot: Contact microenvironment deltas ({len(result.contact_deltas)} rows)",
            divider="gray",
        )
        st.dataframe([vars(d) for d in result.contact_deltas], width="stretch")

        top_sae_diff = result.sae_diffs.get(tc.candidate.sequence)
        if top_sae_diff is not None:
            st.subheader(
                ":material/insights: SAE feature diff (compensated vs edit-only, top 20 by |delta|)",
                divider="gray",
            )
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
                "sequence": [seq[:12] + "..." for seq in candidate_sequences],
            }
            st.caption("PCA of each candidate's top-3 ΔΔSAE (compensated vs WT) features, unioned across candidates.")
            st.scatter_chart(pca_df, x="PC1", y="PC2")

            st.caption(
                "Describe a candidate's top SAE features with human-readable labels "
                "(esmc_6b/layer60 -- heavier, on-demand only)."
            )
            describe_choice = st.selectbox("Candidate to describe", candidate_sequences, format_func=lambda s: s[:20] + "...")
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


with st.form("run_form"):
    st.subheader(":material/edit_note: Design input", divider="gray")
    wt_sequence = st.text_area(
        "WT sequence",
        value="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
        help="Wild-type protein sequence (default: hen egg-white lysozyme, the CLAUDE.md demo case).",
    )
    with st.container(horizontal=True):
        pdb_id = st.text_input(
            "PDB ID (optional)",
            help="Skip the RCSB search and use this entry directly. Leave blank to always search first.",
        )
        chain_id = st.text_input("Chain ID", value="A")

    col1, col2 = st.columns(2)
    with col1:
        edit_start = st.number_input("Edit start position (1-indexed)", min_value=1, value=1)
        edit_sequence = st.text_input(
            "Edit sequence",
            value="A",
            help="Amino acid sequence to substitute in, starting at the edit start position. "
            "A single letter is a normal one-residue substitution; a longer string "
            "(e.g. 'WHSPRAL') replaces that many consecutive residues.",
        )
    with col2:
        use_full_window = st.checkbox(
            "Compensatory window = entire protein (excluding the edit)",
            value=True,
            help="MCMC's per-round cost doesn't scale with window size (still one proposal "
            "per chain per round), so the whole protein is a reasonable default.",
        )
        window_start = st.number_input("Compensatory window start", min_value=1, value=2, disabled=use_full_window)
        window_end = st.number_input("Compensatory window end", min_value=1, value=10, disabled=use_full_window)

    with st.expander("MCMC / oracle settings", icon=":material/tune:"):
        num_starting_points = st.number_input("num_starting_points", min_value=1, value=2)
        chains_per_start = st.number_input("chains_per_start", min_value=1, value=2)
        steps = st.number_input(
            "steps per chain",
            min_value=10,
            value=50,
            help="Each round dispatches ~2-3 Modal calls (ESM2/ProteinMPNN/Evo2), so this is "
            "the main lever on total Modal round trips -- lower it if runs feel slow or "
            "'connecting to container' logs are frequent. Was 200; the MCMC search gets "
            "less time to converge at lower values.",
        )
        temperature = st.number_input("temperature", min_value=0.01, value=1.0)
        candidate_num = st.number_input("candidate_num", min_value=1, value=5)
        seed = st.number_input("seed", value=0)
        use_modal_mcmc = st.checkbox(
            "Run MCMC search whole-loop on Modal",
            value=False,
            help="Runs the entire MCMC search inside a single Modal function (oracle/modal_app.py) "
            "instead of one laptop<->Modal round trip per round -- fewer 'connecting to container' "
            "reconnects, near-zero per-round latency. Requires "
            "`modal deploy -e proto-env epigen/pipeline/oracle/modal_app.py` to have been run once.",
        )

    submitted = st.form_submit_button("Run", icon=":material/play_arrow:", type="primary")

if submitted:
    edit_sequence_clean = edit_sequence.strip().upper()
    edit_positions = list(range(int(edit_start), int(edit_start) + len(edit_sequence_clean)))
    if use_full_window:
        window_positions = [p for p in range(1, len(wt_sequence.strip()) + 1) if p not in edit_positions]
    else:
        window_positions = list(range(int(window_start), int(window_end) + 1))
    if set(edit_positions) & set(window_positions):
        st.error("Edit positions must not overlap the compensatory window.", icon=":material/error:")
        st.stop()

    run_kwargs = dict(
        wt_sequence=wt_sequence.strip(),
        edit_start=int(edit_start),
        edit_sequence=edit_sequence_clean,
        window_positions=window_positions,
        pdb_id=pdb_id.strip() or None,
        chain_id=chain_id.strip() or "A",
        num_starting_points=int(num_starting_points),
        chains_per_start=int(chains_per_start),
        steps=int(steps),
        temperature=float(temperature),
        candidate_num=int(candidate_num),
        seed=int(seed),
        use_modal_mcmc=use_modal_mcmc,
    )

    with st.spinner("Running fold -> oracle/MCMC -> diffs (this calls Modal several times)..."):
        try:
            result = cached_run_end_to_end(**run_kwargs)
        except Exception as exc:
            st.exception(exc)
            st.stop()

    inputs = run_history.derive_inputs(run_kwargs)
    st.session_state.epigen_result = result
    st.session_state.epigen_inputs = inputs
    run_history.record_run(
        run_kwargs,
        summary={"wt_len": len(run_kwargs["wt_sequence"]), "n_candidates": len(result.mcmc_candidates)},
    )

    _render_results(result, inputs)

elif st.session_state.epigen_result is not None:
    st.caption("Showing the last completed/loaded run below. Fill in the form above and click Run to start a new one.")
    _render_results(st.session_state.epigen_result, st.session_state.epigen_inputs)

else:
    st.info("Fill in the form and click Run.", icon=":material/info:")
