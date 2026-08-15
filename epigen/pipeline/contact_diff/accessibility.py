"""Motif solvent-accessibility check via SASA.

CLAUDE.md's validation section requires confirming the lactocepin substrate
motif stays surface-accessible in the compensated structure (not buried by
repacking) -- this module is that check, and it's also one of the named
inputs to the stage-4 explanation agent ("motif accessibility check").

Uses Biotite's Shrake-Rupley SASA (per-atom, summed to per-residue via
`apply_residue_wise`) against a coarse absolute-Å² threshold, rather than a
relative-ASA lookup (e.g. Tien et al. 2013) normalizing by each residue
type's theoretical max. Mirrors `contact_diff/contact_energy.py`'s choice to
avoid hand-transcribing a reference table -- a transcription slip there would
silently corrupt the one number this whole tool is built to keep trustworthy.
Flagged as a coarse proxy for relative accessibility, not a substitute for
one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from biotite.structure import apply_residue_wise, sasa

from proto_tools.entities.structures import Structure

# Absolute per-residue SASA (Å²) above which a residue is considered
# solvent-exposed. Coarse: a full relative-ASA calculation would normalize by
# each residue type's theoretical max (e.g. Trp's max is ~3x Gly's), but this
# flat cutoff sits comfortably below the smallest side chains' exposed range
# and comfortably above what a fully buried residue of any type reads at, so
# it's serviceable as a binary exposed/buried proxy without a reference table.
EXPOSED_THRESHOLD_A2 = 25.0


@dataclass(frozen=True)
class MotifAccessibility:
    """Per-residue SASA for one motif's positions, plus the overall verdict."""

    positions: list[int]  # 1-indexed, chain-native numbering
    sasa_a2: dict[int, float]  # position -> per-residue SASA (Å²)
    exposed: dict[int, bool]  # position -> sasa_a2 >= EXPOSED_THRESHOLD_A2
    motif_accessible: bool  # True iff every position in `positions` is exposed
    threshold_a2: float = EXPOSED_THRESHOLD_A2


def per_residue_sasa(structure: Structure, chain_id: str) -> dict[int, float]:
    """Per-residue absolute SASA (Å²) for every residue in `chain_id`."""
    atom_array = structure._get_atom_array(chain_id)
    sasa_per_atom = sasa(atom_array)
    # NaN atoms (e.g. missing VdW radii) contribute 0, not a dropped residue --
    # matches the docstring example's np.nansum usage.
    sasa_per_residue = apply_residue_wise(atom_array, sasa_per_atom, np.nansum)
    res_ids = apply_residue_wise(atom_array, atom_array.res_id, lambda ids: ids[0])
    return {int(res_id): float(value) for res_id, value in zip(res_ids, sasa_per_residue, strict=True)}


def motif_accessibility_check(
    structure: Structure,
    chain_id: str,
    positions: list[int],
    *,
    threshold_a2: float = EXPOSED_THRESHOLD_A2,
) -> MotifAccessibility:
    """Whether every position in `positions` (the inserted motif's residues) is
    solvent-exposed in `structure` -- i.e. not buried by repacking around the edit.
    """
    per_residue = per_residue_sasa(structure, chain_id)
    sasa_a2 = {pos: per_residue.get(pos, 0.0) for pos in positions}
    exposed = {pos: value >= threshold_a2 for pos, value in sasa_a2.items()}
    return MotifAccessibility(
        positions=list(positions),
        sasa_a2=sasa_a2,
        exposed=exposed,
        motif_accessible=all(exposed.values()) if positions else True,
        threshold_a2=threshold_a2,
    )
