"""Shared, disk-persisted cache wrapper around `run_end_to_end`.

Both the Design form and the sidebar's "past experiments" loader call this
same cached function, keyed on its exact call kwargs. `persist="disk"` means
a past experiment survives an app restart, not just the current session --
loading one is then a real cache hit (near-instant) instead of a recompute,
as long as the disk cache hasn't been cleared since.
"""

from __future__ import annotations

import streamlit as st

from epigen.pipeline.orchestrate import run_end_to_end


@st.cache_data(show_spinner=False, persist="disk")
def cached_run_end_to_end(**kwargs):
    """Cache `run_end_to_end`'s result by its exact keyword arguments.

    Each run is several Modal calls deep and takes minutes -- identical
    settings (same sequence, edit, window, MCMC config, seed) should load
    from cache instead of recomputing and re-spending Modal time. MCMC is
    already fully reproducible given the same seed, so this changes nothing
    about correctness, only redundant recomputation. Keyword-only so the
    exact call always matches `run_history`'s stored kwargs dict verbatim.
    """
    return run_end_to_end(**kwargs)
