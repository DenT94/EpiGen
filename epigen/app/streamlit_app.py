"""EpiGen Streamlit UI entry point.

Sets up shared page config/header/session state, then delegates to
`app_pages/*.py` via `st.navigation`. See `app_pages/landing.py` for the
input form, `app_pages/results.py` for the detailed tables/plots on a
completed run, and `app_pages/structure.py` for the structural viewer
(WT/edit/candidate structures with the edit and compensatory positions
highlighted).
"""

from __future__ import annotations

import streamlit as st

from epigen.app import run_history
from epigen.app.pipeline_cache import cached_run_end_to_end

st.set_page_config(page_title="EpiGen", page_icon=":material/biotech:", layout="wide")

# Shared across pages: the last completed pipeline run (None until Landing's
# form is submitted successfully, or a past run is loaded from the sidebar
# below). `epigen_inputs` mirrors the run's actual arguments (not raw widget
# state) via `run_history.derive_inputs`, the single source of truth for
# that shape.
if "epigen_result" not in st.session_state:
    st.session_state.epigen_result = None
    st.session_state.epigen_inputs = None

title_col, clear_col = st.columns([5, 1], vertical_alignment="bottom")
with title_col:
    st.title(":material/biotech: EpiGen")
    st.caption(
        "Protein design with agentic "
        "compensatory-mutation explanation. Substitution-only MVP."
    )
with clear_col:
    if st.button(
        "Clear cache",
        icon=":material/refresh:",
        help="Force every run (and every on-demand fold) to recompute instead of loading a cached result.",
    ):
        st.cache_data.clear()
        st.toast("Cache cleared.", icon=":material/check:")

with st.sidebar:
    st.subheader(":material/history: Past experiments", divider="gray")
    history = run_history.load_history()
    if not history:
        st.caption("No past runs yet -- completed Landing runs will show up here.")
    else:
        past_run = st.selectbox(
            "Load a past run",
            history,
            format_func=run_history.format_label,
            label_visibility="collapsed",
        )
        if st.button("Load", icon=":material/download:", width="stretch"):
            with st.spinner("Loading past experiment (instant if still cached on disk)..."):
                try:
                    result = cached_run_end_to_end(**past_run["kwargs"])
                except Exception as exc:
                    st.exception(exc)
                else:
                    st.session_state.epigen_result = result
                    st.session_state.epigen_inputs = run_history.derive_inputs(past_run["kwargs"])
                    st.switch_page("app_pages/results.py")

page = st.navigation(
    [
        st.Page("app_pages/landing.py", title="Landing", icon=":material/edit_note:", default=True),
        st.Page("app_pages/structure.py", title="Structure viewer", icon=":material/view_in_ar:"),
        st.Page("app_pages/results.py", title="Results", icon=":material/table_chart:"),
    ]
)
page.run()
