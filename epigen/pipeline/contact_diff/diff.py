"""Per-neighbor numeric deltas between an original (WT) and edited/compensated structure.

CLAUDE.md is explicit that the explanation agent (stage 4) must be handed
real numeric deltas, never categorical "neighbor changed" labels -- this
module is where those numbers get computed.

mypipelinethoughts.md step 5 wants every AA that actually changed (not one
nominal edit position) walked individually, with its own microenvironment.
All position bookkeeping here happens in WT-native numbering, via a
`PositionMap` (see epigen.pipeline.alignment) -- necessary because an
insertion shifts every downstream position, so comparing "position i" of the
original and candidate sequences directly is only valid for substitution
edits (identity_map()). Neighbors that fall inside an inserted span (no WT
counterpart) are excluded, mirroring sae_diff's "ignore the edit sequence"
handling.
"""

from __future__ import annotations

from dataclasses import dataclass

from epigen.pipeline.alignment import PositionMap, identity_map
from epigen.pipeline.contact_diff.contact_energy import contact_energy
from epigen.pipeline.contact_diff.neighbors import DEFAULT_RADIUS_A, find_neighbors
from epigen.pipeline.fold_invert_refold.run import FoldedStructure


@dataclass(frozen=True)
class NeighborDelta:
    """Numeric deltas for one neighbor position (WT-native numbering), original vs candidate."""

    edit_position: int  # WT-native position of the changed AA this neighbor is near
    position: int  # WT-native position of the neighbor itself
    original_residue: str  # one-letter
    candidate_residue: str  # one-letter
    distance_a: float  # neighbor's distance to the edit position, in the candidate structure
    delta_distance_a: float  # candidate - original
    delta_contact_energy: float  # candidate - original (negative = became more favorable)
    delta_plddt: float | None  # candidate - original, local to this neighbor position
    delta_pae: float | None  # candidate - original, edit-position/neighbor cell (needs include_pae_matrix=True)


def find_changed_positions(
    original: FoldedStructure,
    candidate: FoldedStructure,
    chain_id: str,
    position_map: PositionMap = identity_map(),
) -> list[int]:
    """WT-native positions where `candidate` differs from `original`.

    Positions inside an inserted span (no WT counterpart) are excluded --
    the inserted motif isn't a "change" relative to WT in a comparable
    sense, it's new sequence with nothing to diff against.
    """
    original_seq = original.structure.get_chain_sequence(chain_id)
    candidate_seq = candidate.structure.get_chain_sequence(chain_id)
    changed = []
    for wt_pos in range(1, len(original_seq) + 1):
        cand_pos = position_map.to_edited(wt_pos)
        if cand_pos < 1 or cand_pos > len(candidate_seq):
            continue
        if original_seq[wt_pos - 1] != candidate_seq[cand_pos - 1]:
            changed.append(wt_pos)
    return changed


def _neighbor_deltas_for_position(
    original: FoldedStructure,
    candidate: FoldedStructure,
    chain_id: str,
    edit_position_wt: int,
    position_map: PositionMap,
    radius_a: float,
) -> list[NeighborDelta]:
    edit_position_cand = position_map.to_edited(edit_position_wt)

    original_seq = original.structure.get_chain_sequence(chain_id)
    candidate_seq = candidate.structure.get_chain_sequence(chain_id)
    original_plddt = original.structure.per_residue_plddt()
    candidate_plddt = candidate.structure.per_residue_plddt()
    original_pae = getattr(original.structure.metrics, "pae", None)
    candidate_pae = getattr(candidate.structure.metrics, "pae", None)

    original_neighbors = {n.position: n for n in find_neighbors(original.structure, chain_id, edit_position_wt, radius_a=radius_a)}
    candidate_neighbors_native = {n.position: n for n in find_neighbors(candidate.structure, chain_id, edit_position_cand, radius_a=radius_a)}
    # Reindex candidate-side neighbors into WT space; drop ones with no WT counterpart
    # (inside the inserted span) -- there's nothing on the "original" side to diff against.
    candidate_neighbors: dict[int, object] = {}
    for cand_pos, neighbor in candidate_neighbors_native.items():
        wt_pos = position_map.to_wt(cand_pos)
        if wt_pos is not None:
            candidate_neighbors[wt_pos] = neighbor

    wt_positions = sorted(set(original_neighbors) | set(candidate_neighbors))

    deltas: list[NeighborDelta] = []
    for wt_pos in wt_positions:
        cand_pos = position_map.to_edited(wt_pos)
        if wt_pos - 1 >= len(original_seq) or cand_pos - 1 >= len(candidate_seq):
            continue
        orig_aa = original_seq[wt_pos - 1]
        cand_aa = candidate_seq[cand_pos - 1]
        edit_orig_aa = original_seq[edit_position_wt - 1]
        edit_cand_aa = candidate_seq[edit_position_cand - 1]

        # A neighbor missing from one side (fell outside the cutoff there) has no
        # distance on that side; treat it as "at the cutoff radius" for the delta
        # rather than dropping the neighbor -- the entry/exit itself is signal.
        orig_dist = original_neighbors[wt_pos].distance_a if wt_pos in original_neighbors else radius_a
        cand_dist = candidate_neighbors[wt_pos].distance_a if wt_pos in candidate_neighbors else radius_a

        delta_plddt = None
        if (
            original_plddt is not None
            and candidate_plddt is not None
            and wt_pos - 1 < len(original_plddt)
            and cand_pos - 1 < len(candidate_plddt)
        ):
            delta_plddt = candidate_plddt[cand_pos - 1] - original_plddt[wt_pos - 1]

        delta_pae = None
        if original_pae is not None and candidate_pae is not None:
            delta_pae = candidate_pae[edit_position_cand - 1][cand_pos - 1] - original_pae[edit_position_wt - 1][wt_pos - 1]

        deltas.append(
            NeighborDelta(
                edit_position=edit_position_wt,
                position=wt_pos,
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


def diff_all_changed_positions(
    original: FoldedStructure,
    candidate: FoldedStructure,
    *,
    chain_id: str,
    position_map: PositionMap = identity_map(),
    radius_a: float = DEFAULT_RADIUS_A,
) -> list[NeighborDelta]:
    """Numeric per-neighbor deltas for every AA that changed between `original` and `candidate`.

    Replaces the old single-`edit_position` API: mypipelinethoughts.md step 5
    wants every changed position (a compensatory redesign can touch several
    residues in the edit window) walked individually, each with its own
    microenvironment.
    """
    changed_positions = find_changed_positions(original, candidate, chain_id, position_map)
    deltas: list[NeighborDelta] = []
    for edit_position_wt in changed_positions:
        deltas.extend(
            _neighbor_deltas_for_position(original, candidate, chain_id, edit_position_wt, position_map, radius_a)
        )
    return deltas
