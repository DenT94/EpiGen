"""EpiGen landing page: the design input form.

Just the starting box -- input -> get_structure (PDB-first) -> oracle/MCMC
search -> refold+TM-gate -> contact/SAE diffs, per mypipelinethoughts.md.
Substitution-only MVP (see todo.md for insertion notes). On a successful
submit, the result is stashed in `st.session_state.epigen_result`/
`epigen_inputs` and the app switches to the Results page to display it;
`app_pages/results.py` (detailed tables/plots) and `app_pages/structure.py`
(3D viewer) both just read that session state, no re-running the pipeline.
"""

from __future__ import annotations

import streamlit as st

from epigen.app import run_history
from epigen.app.pipeline_cache import cached_run_end_to_end

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
            value="193L",
            help="Skip the RCSB search and use this entry directly. Leave blank to always search first. "
            "Defaults to 193L, chain A -- the entry the WT sequence above already auto-resolves to "
            "(100% identity), just given directly so a run doesn't pay for the RCSB search every time.",
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

    st.session_state.epigen_result = result
    st.session_state.epigen_inputs = run_history.derive_inputs(run_kwargs)
    run_history.record_run(
        run_kwargs,
        summary={"wt_len": len(run_kwargs["wt_sequence"]), "n_candidates": len(result.mcmc_candidates)},
    )
    st.switch_page("app_pages/results.py")

elif st.session_state.epigen_result is not None:
    st.caption("You already have a loaded run.")
    st.page_link("app_pages/results.py", label="Go to Results", icon=":material/table_chart:")

else:
    st.info("Fill in the form and click Run.", icon=":material/info:")
