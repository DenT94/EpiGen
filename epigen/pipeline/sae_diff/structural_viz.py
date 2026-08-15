"""Color a candidate's structure by one chosen SAE feature's per-residue activation.

mypipelinethoughts.md step 5: "color structural prediction by chosen SAE
feature, with a small explanation on what the feature is."

py2Dmol's `set_color` takes a `{position: hex_color}` dict (`position=True`)
-- it does the per-residue *placement*, but expects literal colors, not raw
magnitudes; the magnitude->color mapping happens here. Its `show()` method
is IPython-coupled (calls `display(HTML(...))` directly), so we call the
same HTML-building step it uses internally (`_display_viewer`) ourselves
and hand the raw HTML string to the caller (e.g. Streamlit's
`st.components.v1.html`) instead -- confirmed this works by reading
py2Dmol's source (`view.show`/`view._display_html`), not documented
publicly.
"""

from __future__ import annotations

from proto_tools.entities.structures import Structure

from epigen.pipeline.sae_diff.run import FeatureVector

# Not-active residues (feature isn't in that position's top-k) render as this
# neutral gray, distinct from the low end of the activation colormap.
INACTIVE_COLOR = "#dddddd"


def _magnitude_to_color(value: float, vmax: float) -> str:
    """White (0 activation) to red (`vmax`) linear colormap.

    SAE activations from a TopK codebook are non-negative, so a simple
    sequential (not diverging) scale is the right shape here.
    """
    if vmax <= 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, value / vmax))
    # White (255,255,255) -> red (200,30,30).
    r = int(255 + t * (200 - 255))
    g = int(255 + t * (30 - 255))
    b = int(255 + t * (30 - 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def feature_color_map(feature_vector: FeatureVector, feature_index: int) -> dict[int, str]:
    """Per-residue hex color for one SAE feature's activation across a sequence.

    `feature_vector` is WT-native-indexed (see `sae_diff.run.reindex_to_wt`);
    positions where `feature_index` isn't among that residue's active
    features get `INACTIVE_COLOR`.
    """
    magnitudes = {pos: features[feature_index] for pos, features in feature_vector.items() if feature_index in features}
    vmax = max(magnitudes.values()) if magnitudes else 0.0
    all_positions = feature_vector.keys()
    return {
        pos: _magnitude_to_color(magnitudes[pos], vmax) if pos in magnitudes else INACTIVE_COLOR
        for pos in all_positions
    }


def render_structure_html(structure: Structure, color_map: dict[int, str], *, chain_id: str = "A") -> str:
    """Render `structure` with `color_map`'s per-residue colors, as a self-contained HTML string.

    Pass the returned string to `st.components.v1.html(html, height=...)`
    (or any other raw-HTML embed) -- this does not display anything itself.
    """
    import py2Dmol

    view = py2Dmol.view()
    with structure.temp_file() as pdb_path:
        view.add_pdb(str(pdb_path), chains=[chain_id], name="candidate")
    view.set_color(color_map, position=True)
    return view._display_viewer(static_data=view.objects)
