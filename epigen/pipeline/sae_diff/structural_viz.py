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

Cartoon (secondary-structure ribbons: twisted helices, arrowed strands) is
real in py2Dmol, but only on GitHub main -- the PyPI release this project
originally depended on (1.6.5) predates it entirely (no "style" param, no
`viewer-cartoon.min.js`, no Style dropdown in the rendered widget). Fixed by
pinning `py2Dmol @ git+https://github.com/sokrypton/py2Dmol.git` in
pyproject.toml (see that file's comment) instead of PyPI. With the git
version, the cartoon plugin is bundled into every rendered viewer
automatically (`viewer.py`: "always included so the Style dropdown can
switch between ribbon and cartoon at runtime") -- no code change needed
here; `view = py2Dmol.view()` still defaults to `style="tube"`, and Tube vs
Cartoon becomes a live toggle in the widget's own `Style:` dropdown, the
same way `Color:` already works.

Same-viewpoint alignment: py2Dmol always computes its own PCA-based "best
view" camera (rotation_matrix/center) fresh per structure (`viewer.py`'s
`_update`, "ALWAYS computed for first frame"), so two independently-rendered
structures land at different, arbitrary camera angles -- and ESMFold's
absolute output coordinates aren't consistent across separate folding runs
even for near-identical sequences, so the raw structures don't already
overlay either. `align_to_reference` Kabsch-superimposes a structure's CA
trace onto a shared reference (WT) before rendering; `compute_reference_camera`
captures the reference's own best-view camera once so every aligned
structure can be rendered with that same fixed rotation_matrix/center
(`render_structure_html`'s `reference_camera` param) instead of recomputing
its own.
"""

from __future__ import annotations

from proto_tools.entities.structures import Structure

from epigen.pipeline.sae_diff.run import FeatureVector

RotationMatrix = list[list[float]]
Center = list[float]

# Not-active residues (feature isn't in that position's top-k) render as this
# neutral gray, distinct from the low end of the activation colormap.
INACTIVE_COLOR = "#dddddd"


def _magnitude_to_color(value: float, vmax: float) -> str:
    """White (0 activation) to orange (`vmax`) linear colormap.

    SAE activations from a TopK codebook are non-negative, so a simple
    sequential (not diverging) scale is the right shape here. Orange (not
    red) deliberately -- the Structure viewer already colors the fixed edit
    red (`app_pages/structure.py`'s EDIT_COLOR), so a red SAE overlay would
    visually merge with the edit highlight at the very positions callers
    most want to compare against it. Matches `literature/plot.py`'s own
    EDIT_COLOR/WINDOW_COLOR orange, for consistency across the app.
    """
    if vmax <= 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, value / vmax))
    # White (255,255,255) -> orange (232,137,26).
    r = int(255 + t * (232 - 255))
    g = int(255 + t * (137 - 255))
    b = int(255 + t * (26 - 255))
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


def align_to_reference(structure: Structure, reference: Structure, chain_id: str = "A") -> Structure:
    """Rigidly superimpose `structure`'s chain onto `reference`'s same chain, so every
    structure shown in the Structure viewer -- WT, edit-only, top candidate, any other MCMC
    candidate -- lands in the same coordinate frame as `reference` (conventionally WT)
    before rendering, rather than each keeping whatever arbitrary absolute frame ESMFold
    happened to output it in.

    Uses `superimpose_homologs` (sequence-alignment-anchored), not a plain positional Kabsch
    fit -- WT is frequently a real PDB structure, and real crystal structures routinely have
    a handful of residues missing from unresolved density, so its CA count doesn't reliably
    match the ESMFold2-predicted edit-only/candidate structures' even for a substitution-only
    edit with no length change in the *design*. A plain index-for-index Kabsch fit over
    mismatched-length CA arrays either raises or (in an earlier version of this function)
    silently skipped alignment entirely -- the latter is what made different Structure viewer
    tabs appear to use different camera angles despite all being given the same fixed
    `reference_camera`: the camera was identical, but the underlying coordinates weren't
    actually in the same frame for it to apply to.
    """
    from io import StringIO

    from biotite.structure import superimpose_homologs
    from biotite.structure.io.pdb import PDBFile

    reference_array = reference._get_atom_array(chain_id)
    mobile_array = structure._get_atom_array(chain_id)

    fitted, *_ = superimpose_homologs(reference_array, mobile_array)

    pdb_file = PDBFile()
    pdb_file.set_structure(fitted)
    buffer = StringIO()
    pdb_file.write(buffer)
    return Structure(structure=buffer.getvalue(), structure_format="pdb")


def compute_reference_camera(structure: Structure, chain_id: str = "A") -> tuple[RotationMatrix, Center]:
    """py2Dmol's PCA-based "best view" rotation_matrix + center for `structure`.

    Call once on the reference structure (WT) and pass the result as every
    other call's `render_structure_html(..., reference_camera=...)` so all
    views share one fixed camera instead of each computing its own.
    """
    import py2Dmol

    view = py2Dmol.view()
    with structure.temp_file() as pdb_path:
        view.add_pdb(str(pdb_path), chains=[chain_id], name="reference")
    return view._rotation_matrix.tolist(), view._center.tolist()


def render_structure_html(
    structure: Structure,
    color_map: dict[int, str],
    *,
    chain_id: str = "A",
    reference_camera: tuple[RotationMatrix, Center] | None = None,
) -> str:
    """Render `structure` with `color_map`'s per-residue colors, as a self-contained HTML string.

    Pass the returned string to `st.components.v1.html(html, height=...)`
    (or any other raw-HTML embed) -- this does not display anything itself.

    `reference_camera`: fixed `(rotation_matrix, center)` from
    `compute_reference_camera`, overriding py2Dmol's own per-structure
    best-view computation so multiple structures render from the same
    viewpoint. Combine with `align_to_reference` (align first, then render
    with that reference's camera) -- the camera alone isn't enough if the
    structures aren't already in the same coordinate frame.

    Color mode is py2Dmol's own "auto" default. "plddt" was tried instead
    (py2Dmol reads pLDDT from the PDB B-factor column) but rendered as
    uniform red regardless of true confidence -- py2Dmol's parser expects
    the AlphaFold 0-100 pLDDT scale (its own fallback default is 50.0),
    while our structures' B-factor column is 0-1, so every real value reads
    as near-zero confidence. Would need rescaling the B-factor column
    itself to fix properly; not done here.

    Render style defaults to "cartoon" (Richardson preset -- helix/strand
    ribbons) rather than py2Dmol's own "tube" default, since it reads the
    secondary structure directly. Still switchable live via the widget's own
    `Style:` dropdown (Tube/Cartoon), same as `Color:`.
    """
    import numpy as np
    import py2Dmol

    view = py2Dmol.view(style="cartoon")
    with structure.temp_file() as pdb_path:
        view.add_pdb(str(pdb_path), chains=[chain_id], name="candidate")
    view.set_color(color_map, position=True)
    if reference_camera is not None:
        rotation_matrix, center = reference_camera
        view._rotation_matrix = np.array(rotation_matrix)
        view._center = np.array(center)
    return view._display_viewer(static_data=view.objects)
