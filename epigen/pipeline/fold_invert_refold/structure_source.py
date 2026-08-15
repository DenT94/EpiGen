"""Structure sourcing: try the real PDB before ever falling back to ESMFold2.

Per project decision, the PDB shortcut is *always* attempted for a raw
sequence, not just when a PDB ID is given directly: a real solved structure
is strictly more trustworthy to design against than a prediction, so
ESMFold2 only runs when no sufficiently-identical PDB entry exists.

Substitution-MVP scope only (see todo.md): this assumes `sequence` is the
exact construct being designed against, at a single chain. It does not
handle multichain complexes or sequences with unresolved/missing PDB
density -- both are out of scope for now, not just the insertion case.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher

import requests

from proto_tools.entities.structures import Structure

from epigen.pipeline.fold_invert_refold.run import FoldedStructure, fold_sequence

logger = logging.getLogger(__name__)

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DEFAULT_MIN_IDENTITY = 0.98
DEFAULT_SEARCH_ROWS = 5


def _search_rcsb_candidates(sequence: str, *, rows: int = DEFAULT_SEARCH_ROWS) -> list[str]:
    """Candidate PDB polymer-entity IDs (e.g. '1AKI_1') for `sequence`, by RCSB's public
    sequence-search API, ranked by RCSB's own relevance score.

    This is a shortlist only -- identity is verified locally against each
    candidate's real fetched sequence before any is trusted (RCSB's score
    isn't a verified identity fraction).
    """
    body = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1.0,
                "identity_cutoff": 0.9,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {"paginate": {"start": 0, "rows": rows}},
    }
    response = requests.post(RCSB_SEARCH_URL, json=body, timeout=15)
    if response.status_code == 204:  # RCSB's "no hits" response has no body.
        return []
    response.raise_for_status()
    return [hit["identifier"] for hit in response.json().get("result_set", [])]


def _sequence_identity(a: str, b: str) -> float:
    """Fraction of `a`/`b` in agreement. 0.0 if lengths differ by more than a few residues
    (treated as a different construct, not worth a fuzzy alignment)."""
    if abs(len(a) - len(b)) > 5:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _fetch_pdb_chain(pdb_id: str, chain_id: str) -> Structure:
    return Structure.from_rcsb(pdb_id).select_chain(chain_id)


def get_structure(
    sequence: str,
    *,
    pdb_id: str | None = None,
    chain_id: str = "A",
    min_identity: float = DEFAULT_MIN_IDENTITY,
    seed: int | None = None,
) -> FoldedStructure:
    """Resolve a structure for `sequence`.

    Order: an explicitly given `pdb_id` is always trusted as-is; otherwise
    the RCSB sequence search is always attempted first and the closest
    match (>= `min_identity`) is used; ESMFold2 only runs if no PDB entry
    clears that bar.

    Real PDB structures get `plddt=1.0`, `avg_pae=0.0`,
    `passed_confidence_gate=True` -- there's no model uncertainty to gate on.
    """
    if pdb_id is not None:
        structure = _fetch_pdb_chain(pdb_id, chain_id)
        logger.info(f"Using given PDB entry {pdb_id!r} directly, chain {chain_id!r}.")
        return FoldedStructure(
            sequence=sequence, structure=structure, plddt=1.0, avg_pae=0.0,
            passed_confidence_gate=True, source="pdb", pdb_id=pdb_id,
        )

    for candidate_id in _search_rcsb_candidates(sequence):
        base_id = candidate_id.split("_")[0]
        try:
            candidate_structure = _fetch_pdb_chain(base_id, chain_id)
        except Exception as exc:  # network/format hiccups on one candidate shouldn't abort the search
            logger.warning(f"Skipping PDB candidate {candidate_id!r}: could not fetch chain {chain_id!r} ({exc}).")
            continue
        candidate_sequence = candidate_structure.get_chain_sequence(chain_id)
        identity = _sequence_identity(sequence, candidate_sequence)
        if identity >= min_identity:
            logger.info(f"Using PDB entry {base_id!r} (identity={identity:.3f} >= {min_identity:.3f}).")
            return FoldedStructure(
                sequence=sequence, structure=candidate_structure, plddt=1.0, avg_pae=0.0,
                passed_confidence_gate=True, source="pdb", pdb_id=base_id,
            )
        logger.info(f"PDB candidate {candidate_id!r} identity {identity:.3f} below {min_identity:.3f}; trying next.")

    logger.info(f"No PDB match >= {min_identity:.0%} identity; falling back to ESMFold2.")
    return fold_sequence(sequence, seed=seed)
