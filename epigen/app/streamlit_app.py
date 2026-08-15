"""EpiGen Streamlit UI.

Input: edit position + insertion sequence.
Output: ranked compensatory candidates + agent explanations.
"""

import streamlit as st

st.set_page_config(page_title="EpiGen", layout="wide")

st.title("EpiGen")
st.caption(
    "Protease-gated selective antibiotic design with agentic "
    "compensatory-mutation explanation."
)

st.info("Scaffold only -- pipeline stages not yet wired up.")
