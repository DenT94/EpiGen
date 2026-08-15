"""EpiGen Streamlit UI.

First full end-to-end pass, per mypipelinethoughts.md: input -> get_structure
(PDB-first) -> oracle/MCMC search -> refold+TM-gate -> contact/SAE diffs.
Substitution-only MVP (see todo.md for insertion notes). Deliberately raw --
tables/JSON, no styling or structural viz yet; the point is proving the
whole chain runs end to end from a form before investing in presentation.
"""

from __future__ import annotations

import streamlit as st

from epigen.pipeline.literature import plot_annotation_map
from epigen.pipeline.orchestrate import run_end_to_end

st.set_page_config(page_title="EpiGen", layout="wide")

st.title("EpiGen")
st.caption(
    "Protease-gated selective antibiotic design with agentic "
    "compensatory-mutation explanation. Substitution-only MVP."
)

with st.form("run_form"):
    wt_sequence = st.text_area(
        "WT sequence",
        value="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
        help="Wild-type protein sequence (default: hen egg-white lysozyme, the CLAUDE.md demo case).",
    )
    pdb_id = st.text_input(
        "PDB ID (optional)",
        help="Skip the RCSB search and use this entry directly. Leave blank to always search first.",
    )
    chain_id = st.text_input("Chain ID", value="A")

    col1, col2 = st.columns(2)
    with col1:
        edit_position = st.number_input("Edit position (1-indexed)", min_value=1, value=1)
        edit_residue = st.text_input("Edit residue (one letter)", value="A", max_chars=1)
    with col2:
        window_start = st.number_input("Compensatory window start", min_value=1, value=2)
        window_end = st.number_input("Compensatory window end", min_value=1, value=10)

    with st.expander("MCMC / oracle settings"):
        num_starting_points = st.number_input("num_starting_points", min_value=1, value=2)
        chains_per_start = st.number_input("chains_per_start", min_value=1, value=2)
        steps = st.number_input("steps per chain", min_value=10, value=200)
        temperature = st.number_input("temperature", min_value=0.01, value=1.0)
        candidate_num = st.number_input("candidate_num", min_value=1, value=5)
        seed = st.number_input("seed", value=0)

    submitted = st.form_submit_button("Run")

if submitted:
    window_positions = list(range(int(window_start), int(window_end) + 1))
    if int(edit_position) in window_positions:
        st.error("Edit position must not fall inside the compensatory window.")
        st.stop()

    with st.spinner("Running fold -> oracle/MCMC -> diffs (this calls Modal several times)..."):
        try:
            result = run_end_to_end(
                wt_sequence.strip(),
                edit_position=int(edit_position),
                edit_residue=edit_residue.strip().upper(),
                window_positions=window_positions,
                pdb_id=pdb_id.strip() or None,
                chain_id=chain_id.strip() or "A",
                num_starting_points=int(num_starting_points),
                chains_per_start=int(chains_per_start),
                steps=int(steps),
                temperature=float(temperature),
                candidate_num=int(candidate_num),
                seed=int(seed),
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.subheader("Structure source")
    st.write(
        f"**WT**: source={result.original.source}"
        + (f", pdb_id={result.original.pdb_id}" if result.original.pdb_id else f", pLDDT={result.original.plddt:.3f}")
    )
    st.write(f"**Edit-only** (pos {edit_position}={edit_residue}): pLDDT={result.edit_only.plddt:.3f}")

    st.subheader("Literature annotation map")
    if result.annotation_ranges:
        if result.annotation_conflicts:
            st.warning(
                f"Edit position or compensatory window overlaps {len(result.annotation_conflicts)} "
                "known functional/structural annotation(s) -- see ⚠ rows below."
            )
        st.pyplot(
            plot_annotation_map(
                len(wt_sequence.strip()),
                result.annotation_ranges,
                edit_position=int(edit_position),
                window_positions=window_positions,
                conflicts=result.annotation_conflicts,
            )
        )
    else:
        st.caption("No Paperclip/UniProt annotations found for this sequence.")

    st.subheader("Oracle sanity checks")
    c1, c2 = st.columns(2)
    c1.metric("Expert correlation (ESM2 vs ProteinMPNN)", f"{result.expert_correlation:.3f}")
    c2.metric("Window substitutions below WT", f"{result.fraction_below_wt:.1%}")

    st.subheader(f"MCMC candidates (top {len(result.mcmc_candidates)})")
    st.dataframe(
        [{"sequence": c.sequence, "combined_score": c.combined_score} for c in result.mcmc_candidates],
        use_container_width=True,
    )

    if result.top_candidate is None:
        st.warning("No candidates produced -- nothing to refold or diff.")
    else:
        st.subheader("Top candidate: refold + self-consistency")
        tc = result.top_candidate
        st.write(
            f"sequence={tc.candidate.sequence}  \n"
            f"pLDDT={tc.folded.plddt:.3f}  |  TM-score vs edit-only={tc.tm_score:.3f}  |  "
            f"passed self-consistency gate={tc.passed_self_consistency_gate}"
        )

        st.subheader(f"Contact microenvironment deltas ({len(result.contact_deltas)} rows)")
        st.dataframe([vars(d) for d in result.contact_deltas], use_container_width=True)

        top_sae_diff = result.sae_diffs.get(tc.candidate.sequence)
        if top_sae_diff is not None:
            st.subheader("SAE feature diff (compensated vs edit-only, top 20 by |delta|)")
            from epigen.pipeline.sae_diff.run import top_k_deltas

            top_deltas = top_k_deltas(top_sae_diff.compensated_vs_edit, k=20)
            st.dataframe([vars(d) for d in top_deltas], use_container_width=True)

        if result.sae_diffs:
            st.subheader(f"SAE feature space across candidates ({len(result.sae_diffs)} scored)")
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
            if st.button("Describe"):
                with st.spinner("Re-diffing at the describable SAE config (esmc_6b)..."):
                    try:
                        described = describe_candidate(wt_sequence.strip(), result.edit_only.sequence, describe_choice)
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
                            use_container_width=True,
                        )
else:
    st.info("Fill in the form and click Run.")
