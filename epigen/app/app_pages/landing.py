"""EpiGen landing page: the design input form.

Just the starting box -- input -> get_structure (PDB-first) -> oracle/MCMC
search -> refold+TM-gate -> contact/SAE diffs, per mypipelinethoughts.md.
Substitution-only MVP (see todo.md for insertion notes). On a successful
submit, the result is stashed in `st.session_state.epigen_result`/
`epigen_inputs` and the app switches to the Results page to display it;
`app_pages/results.py` (detailed tables/plots) and `app_pages/structure.py`
(3D viewer) both just read that session state, no re-running the pipeline.

`num_edit_positions` (how many different edit positions this run tries at
all -- 1 is the original single-position flow; >1 means edit_start becomes
a *list* of that many positions, typed in or randomly picked, see
`edit_positions.py`) also drives `oracle.mcmc.run_mcmc_search`'s own
`num_starting_points` for each position's compensatory search -- there's no
separate UI field for it; the two are always equal by construction, one
number in the form. Each of the `num_edit_positions` positions gets its own
full, independent pipeline run (fold -> MCMC search -> evidence).
"""

from __future__ import annotations

import streamlit as st

from epigen.app import run_history
from epigen.app.pipeline_cache import cached_run_end_to_end
from epigen.pipeline.edit_positions import pick_random_edit_starts

if "epigen_multi_results" not in st.session_state:
    st.session_state.epigen_multi_results = None

# Outside the form: needs to trigger an immediate rerun when changed (st.form widgets
# don't rerun until submit), since it decides whether the form below shows a single edit
# position field or the list/random picker. Also drives run_mcmc_search's own
# num_starting_points for each position -- see this file's docstring.
st.subheader(":material/edit_note: Design input", divider="gray")
num_edit_positions = st.number_input(
    "Number of edit positions",
    min_value=1,
    value=1,
    help="1: the classic single-edit-position flow below. >1: try the edit at this many "
    "different positions, each a full independent pipeline run -- see this page's own "
    "docstring.",
)

with st.form("run_form"):
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
        if num_edit_positions == 1:
            edit_start = st.number_input(
                "Edit start position (1-indexed)",
                min_value=1,
                value=None,
                placeholder="e.g. 20",
                help="Required -- there's no meaningful default edit position for an arbitrary "
                "WT sequence, so this starts blank rather than silently defaulting to 1.",
            )
            edit_position_mode = None
            edit_starts_text = ""
        else:
            edit_position_mode = st.radio(
                f"Edit positions ({num_edit_positions} needed)",
                ["Pick randomly", "Specify a list"],
                horizontal=True,
                help="One full independent pipeline run per position -- see Number of edit "
                "positions above.",
            )
            if edit_position_mode == "Specify a list":
                edit_starts_text = st.text_input(
                    "Edit start positions (comma-separated)",
                    value="",
                    placeholder=f"e.g. 20, 45, 80 ({num_edit_positions} values needed)",
                )
            else:
                edit_starts_text = ""
                st.caption(
                    f"{int(num_edit_positions)} random positions will be picked when you click Run "
                    "(seeded by the seed field below, for reproducibility)."
                )
            edit_start = None
        edit_sequence = st.text_input(
            "Edit sequence",
            value="",
            placeholder="e.g. WHSPRAL",
            help="Required. Amino acid sequence to substitute in, starting at each edit start "
            "position above. A single letter is a normal one-residue substitution; a longer "
            "string (e.g. 'WHSPRAL') replaces that many consecutive residues -- the same motif "
            "is used at every position when trying more than one. Starts blank rather than "
            "defaulting to a real (if trivial) edit you might submit by accident.",
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
        st.caption(
            "num_starting_points (compensatory-search warm starts per edit position) always "
            "equals Number of edit positions above -- no separate field for it."
        )
        chains_per_start = st.number_input(
            "chains_per_start",
            min_value=1,
            value=2,
            help="Independent MCMC chains per compensatory warm start, per edit position.",
        )
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

        st.caption("Expert weights -- set any to 0 to completely neglect that expert (skips its Modal call, not just its contribution to the score).")
        with st.container(horizontal=True):
            weight_esm2 = st.number_input("weight_esm2", min_value=0.0, value=0.5, step=0.05)
            weight_pmpnn = st.number_input("weight_pmpnn", min_value=0.0, value=0.5, step=0.05)
            weight_evo2 = st.number_input(
                "weight_evo2",
                min_value=0.0,
                value=0.34,
                step=0.05,
                help="0 also skips resolving a coding sequence for Evo2 entirely (no GenBank/CodonFM lookup).",
            )
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
    wt_sequence_clean = wt_sequence.strip()
    edit_sequence_clean = edit_sequence.strip().upper()
    if not edit_sequence_clean:
        st.error("Edit sequence is required.", icon=":material/error:")
        st.stop()

    if num_edit_positions == 1:
        if edit_start is None:
            st.error("Edit start position is required.", icon=":material/error:")
            st.stop()
        edit_starts = [int(edit_start)]
    elif edit_position_mode == "Specify a list":
        raw = [p.strip() for p in edit_starts_text.split(",") if p.strip()]
        if len(raw) != num_edit_positions:
            st.error(
                f"Need exactly {num_edit_positions} comma-separated edit start positions, got {len(raw)}.",
                icon=":material/error:",
            )
            st.stop()
        try:
            edit_starts = [int(p) for p in raw]
        except ValueError:
            st.error("Edit start positions must all be integers.", icon=":material/error:")
            st.stop()
    else:
        try:
            edit_starts = pick_random_edit_starts(
                len(wt_sequence_clean), len(edit_sequence_clean), int(num_edit_positions), seed=int(seed)
            )
        except ValueError as exc:
            st.error(str(exc), icon=":material/error:")
            st.stop()

    per_position_results = []
    with st.spinner(
        f"Running fold -> oracle/MCMC -> diffs for {len(edit_starts)} edit position(s) "
        "(this calls Modal several times per position)..."
    ):
        for this_edit_start in edit_starts:
            edit_positions = list(range(this_edit_start, this_edit_start + len(edit_sequence_clean)))
            if use_full_window:
                window_positions = [p for p in range(1, len(wt_sequence_clean) + 1) if p not in edit_positions]
            else:
                window_positions = list(range(int(window_start), int(window_end) + 1))
            if set(edit_positions) & set(window_positions):
                st.error(
                    f"Edit position {this_edit_start} overlaps the compensatory window.",
                    icon=":material/error:",
                )
                st.stop()

            run_kwargs = dict(
                wt_sequence=wt_sequence_clean,
                edit_start=this_edit_start,
                edit_sequence=edit_sequence_clean,
                window_positions=window_positions,
                pdb_id=pdb_id.strip() or None,
                chain_id=chain_id.strip() or "A",
                num_starting_points=int(num_edit_positions),
                chains_per_start=int(chains_per_start),
                steps=int(steps),
                temperature=float(temperature),
                candidate_num=int(candidate_num),
                seed=int(seed),
                weight_esm2=float(weight_esm2),
                weight_pmpnn=float(weight_pmpnn),
                weight_evo2=float(weight_evo2),
                use_modal_mcmc=use_modal_mcmc,
            )
            try:
                result = cached_run_end_to_end(**run_kwargs)
            except Exception as exc:
                st.exception(exc)
                st.stop()
            run_history.record_run(
                run_kwargs,
                summary={"wt_len": len(wt_sequence_clean), "n_candidates": len(result.mcmc_candidates)},
            )
            per_position_results.append((this_edit_start, result, run_kwargs))

    if len(per_position_results) == 1:
        this_edit_start, result, run_kwargs = per_position_results[0]
        st.session_state.epigen_result = result
        st.session_state.epigen_inputs = run_history.derive_inputs(run_kwargs)
        st.session_state.epigen_multi_results = None
        st.switch_page("app_pages/results.py")
    else:
        st.session_state.epigen_multi_results = per_position_results

if st.session_state.epigen_multi_results:
    st.subheader(":material/compare_arrows: Edit position comparison", divider="gray")
    rows = []
    for this_edit_start, result, _ in st.session_state.epigen_multi_results:
        best = max((c.combined_score for c in result.mcmc_candidates), default=None)
        rows.append(
            {
                "edit_start": this_edit_start,
                "wt_score": result.wt_score,
                "best_combined_score": best,
                "gap": (best - result.wt_score) if best is not None else None,
                "n_candidates": len(result.mcmc_candidates),
            }
        )
    rows.sort(key=lambda r: (r["best_combined_score"] is None, -(r["best_combined_score"] or 0)))
    st.dataframe(
        rows,
        width="stretch",
        column_config={
            "wt_score": st.column_config.NumberColumn("WT score", format="%.3f"),
            "best_combined_score": st.column_config.NumberColumn("Best candidate score", format="%.3f"),
            "gap": st.column_config.NumberColumn("Gap to WT", format="%+.3f"),
        },
    )
    st.caption("Pick one position to inspect in full (Results + Structure viewer):")
    for this_edit_start, result, run_kwargs in st.session_state.epigen_multi_results:
        if st.button(f"View position {this_edit_start}", key=f"view_{this_edit_start}"):
            st.session_state.epigen_result = result
            st.session_state.epigen_inputs = run_history.derive_inputs(run_kwargs)
            st.switch_page("app_pages/results.py")
elif st.session_state.epigen_result is not None:
    st.caption("You already have a loaded run.")
    st.page_link("app_pages/results.py", label="Go to Results", icon=":material/table_chart:")
else:
    st.info("Fill in the form and click Run.", icon=":material/info:")
