"""Neighbor-list construction around an edited position.

Mirrors the CellList-based neighbor-search pattern proto-tools' own
`Structure.interface_contact_residues()` / `Structure.hotspot_contacts()` use
internally (proto_tools/entities/structures/structure.py), but for an
*intra-chain* neighborhood around a single edited position rather than
inter-chain interfaces -- so it reuses `Structure._get_atom_array()` (the
same Biotite AtomArray those methods build) instead of re-parsing PDB text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from biotite.structure import CellList

from proto_tools.entities.structures import Structure

DEFAULT_RADIUS_A = 10.0


@dataclass(frozen=True)
class Neighbor:
    """One residue found near the edit position, with its representative-atom distance."""

    position: int  # 1-indexed
    residue_name: str  # three-letter code, e.g. "ALA"
    distance_a: float  # representative-atom distance to the edit position, in angstroms


def _representative_atom_mask(atom_array) -> np.ndarray:
    """CB atoms where present, CA for glycine (no CB) -- CLAUDE.md's 'prefer CB-CB ...
    over CA-CA' guidance."""
    is_cb = atom_array.atom_name == "CB"
    is_gly_ca = (atom_array.res_name == "GLY") & (atom_array.atom_name == "CA")
    return is_cb | is_gly_ca


def find_neighbors(
    structure: Structure,
    chain_id: str,
    position: int,
    *,
    radius_a: float = DEFAULT_RADIUS_A,
) -> list[Neighbor]:
    """Residues within `radius_a` of `position`'s representative atom.

    Args:
        structure: Structure to search.
        chain_id: Chain containing `position`.
        position: 1-indexed residue position of the edit.
        radius_a: Distance cutoff in angstroms (CLAUDE.md default: 10).

    Returns:
        Neighbors sorted by ascending distance, excluding `position` itself.

    Raises:
        ValueError: If `position` has no representative atom in `chain_id`
            (e.g. an out-of-range position).
    """
    atom_array = structure._get_atom_array(chain_id)
    rep_atoms = atom_array[_representative_atom_mask(atom_array)]

    edit_mask = rep_atoms.res_id == position
    if not edit_mask.any():
        raise ValueError(f"No representative atom found for chain {chain_id!r} position {position}.")
    edit_coord = rep_atoms.coord[edit_mask][0]

    cells = CellList(rep_atoms.coord, cell_size=radius_a)
    hit_indices = cells.get_atoms(edit_coord, radius=radius_a)

    neighbors: list[Neighbor] = []
    for i in hit_indices:
        res_id = int(rep_atoms.res_id[i])
        if res_id == position:
            continue
        distance = float(np.linalg.norm(rep_atoms.coord[i] - edit_coord))
        neighbors.append(Neighbor(position=res_id, residue_name=str(rep_atoms.res_name[i]), distance_a=distance))
    neighbors.sort(key=lambda n: n.distance_a)
    return neighbors
