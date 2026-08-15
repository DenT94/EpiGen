"""Per-neighbor numeric deltas between an original and edited/compensated structure.

CLAUDE.md is explicit that the explanation agent (stage 4) must be handed
real numeric deltas, never categorical "neighbor changed" labels -- this
module is where those numbers get computed, for one edit position at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

from epigen.pipeline.contact_diff.contact_energy import contact_energy
from epigen.pipeline.contact_diff.neighbors import DEFAULT_RADIUS_A, find_neighbors
from epigen.pipeline.fold_invert_refold.run import FoldedStructure


@dataclass(frozen=True)
class NeighborDelta:
    """Numeric deltas for one neighbor position, original vs candidate structure."""

    position: int  # 1-indexed
    original_residue: str  # one-letter
    candidate_residue: str  # one-letter
    distance_a: float  # neighbor's distance to the edit position, in the candidate structure
    delta_distance_a: float  # candidate - original
    delta_contact_energy: float  # candidate - original (negative = became more favorable)
    delta_plddt: float | None  # candidate - original, local to this neighbor position
    delta_pae: float | None  # candidate - original, edit-position/neighbor cell (needs include_pae_matrix=True)


def compute_neighbor_deltas(
    original: FoldedStructure,
    candidate: FoldedStructure,
    *,
    chain_id: str,
    edit_position: int,
    radius_a: float = DEFAULT_RADIUS_A,
) -> list[NeighborDelta]:
    """Numeric per-neighbor deltas around `edit_position`, original vs candidate.

    Neighbors are the union of both structures' neighbor lists at
    `edit_position` (a compensatory mutation can pull a residue into or push
    one out of the cutoff radius), matched by 1-indexed residue position.
    """
    original_neighbors = {n.position: n for n in find_neighbors(original.structure, chain_id, edit_position, radius_a=radius_a)}
    candidate_neighbors = {n.position: n for n in find_neighbors(candidate.structure, chain_id, edit_position, radius_a=radius_a)}
    positions = sorted(set(original_neighbors) | set(candidate_neighbors))

    original_seq = original.structure.get_chain_sequence(chain_id)
    candidate_seq = candidate.structure.get_chain_sequence(chain_id)
    original_plddt = original.structure.per_residue_plddt()
    candidate_plddt = candidate.structure.per_residue_plddt()
    original_pae = original.structure.metrics.pae if hasattr(original.structure.metrics, "pae") else None
    candidate_pae = candidate.structure.metrics.pae if hasattr(candidate.structure.metrics, "pae") else None

    deltas: list[NeighborDelta] = []
    for pos in positions:
        orig_aa = original_seq[pos - 1]
        cand_aa = candidate_seq[pos - 1]
        edit_orig_aa = original_seq[edit_position - 1]
        edit_cand_aa = candidate_seq[edit_position - 1]

        # A neighbor missing from one side (fell outside the cutoff there) has no
        # distance on that side; treat it as "at the cutoff radius" for the delta
        # rather than dropping the neighbor -- the entry/exit itself is signal.
        orig_dist = original_neighbors[pos].distance_a if pos in original_neighbors else radius_a
        cand_dist = candidate_neighbors[pos].distance_a if pos in candidate_neighbors else radius_a

        delta_plddt = None
        if original_plddt is not None and candidate_plddt is not None and pos - 1 < len(original_plddt) and pos - 1 < len(candidate_plddt):
            delta_plddt = candidate_plddt[pos - 1] - original_plddt[pos - 1]

        delta_pae = None
        if original_pae is not None and candidate_pae is not None:
            delta_pae = candidate_pae[edit_position - 1][pos - 1] - original_pae[edit_position - 1][pos - 1]

        deltas.append(
            NeighborDelta(
                position=pos,
                original_residue=orig_aa,
                candidate_residue=cand_aa,
                distance_a=cand_dist,
                delta_distance_a=cand_dist - orig_dist,
                delta_contact_energy=(
                    contact_energy(edit_cand_aa, cand_aa) - contact_energy(edit_orig_aa, orig_aa)
                ),
                delta_plddt=delta_plddt,
                delta_pae=delta_pae,
            )
        )
    return deltas
