"""Real coding-sequence lookup for a PDB entry, via its GenBank cross-reference.

A PDB structure only has atomic coordinates -- no nucleotide sequence. When the
deposited entry names a GenBank nucleotide record (RCSB's `rcsb_polymer_entity_align`
cross-references), that record often carries the actual coding sequence the protein
was expressed from -- a real codon usage, not this pipeline's own guess. This module
finds that record and returns its CDS, but only after verifying (`translate()` against
the PDB's own AA sequence) that it's actually the right one -- GenBank cross-references
sometimes point at a precursor/tagged/differently-numbered construct, and a caller that
trusted a wrong-but-present accession would silently score against the wrong DNA.
"""

from __future__ import annotations

import logging

import requests

from epigen.pipeline.oracle.codon import translate

logger = logging.getLogger(__name__)

RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_GENBANK_DB_NAMES = {"GenBank", "GB"}
_TIMEOUT_S = 15


def _genbank_accessions_for_pdb(pdb_id: str) -> list[str]:
    """Every GenBank accession any polymer entity of `pdb_id` cross-references, in entity order."""
    entry = requests.get(RCSB_ENTRY_URL.format(pdb_id=pdb_id), timeout=_TIMEOUT_S).json()
    entity_ids = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids") or []
    accessions = []
    for entity_id in entity_ids:
        entity = requests.get(
            RCSB_POLYMER_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id), timeout=_TIMEOUT_S
        ).json()
        for ref in entity.get("rcsb_polymer_entity_align") or []:
            if ref.get("reference_database_name") in _GENBANK_DB_NAMES and ref.get("reference_database_accession"):
                accessions.append(ref["reference_database_accession"])
    return accessions


def _fetch_genbank_cds_nt_sequences(accession: str) -> list[str]:
    """Every CDS feature's nt sequence on `accession`'s GenBank record (usually one, but a
    record can carry several genes) -- NCBI's `fasta_cds_na` return type gives each CDS's
    own spliced, in-frame nt sequence directly, so this module never has to parse GenBank's
    full flat-file feature table itself."""
    response = requests.get(
        NCBI_EFETCH_URL,
        params={"db": "nuccore", "id": accession, "rettype": "fasta_cds_na", "retmode": "text"},
        timeout=_TIMEOUT_S,
    )
    response.raise_for_status()
    sequences: list[str] = []
    current: list[str] = []
    for line in response.text.splitlines():
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    return [seq.upper().replace("U", "T") for seq in sequences]


def fetch_pdb_genbank_nt_sequence(pdb_id: str, aa_sequence: str) -> str | None:
    """The real coding sequence for `pdb_id`, if RCSB cross-references a GenBank record whose
    CDS translates to exactly `aa_sequence`.

    `None` on no cross-reference, no CDS that verifies, or any network/lookup failure --
    all expected, non-fatal conditions here; the caller
    (`oracle.nt_resolution.resolve_wt_nt_sequence`) falls back to model-based reverse
    translation when this returns `None`, rather than this function raising.
    """
    try:
        for accession in _genbank_accessions_for_pdb(pdb_id):
            for cds_nt in _fetch_genbank_cds_nt_sequences(accession):
                if len(cds_nt) % 3 != 0:
                    continue
                if translate(cds_nt) == aa_sequence:
                    logger.info(f"Using real GenBank CDS ({accession}) for PDB {pdb_id!r} -- translation matches.")
                    return cds_nt
        logger.info(f"No GenBank CDS found for PDB {pdb_id!r} that translates to the given AA sequence.")
        return None
    except Exception as exc:  # network hiccup, malformed response, RCSB/NCBI outage -- non-fatal, just fall back
        logger.warning(f"GenBank lookup for PDB {pdb_id!r} failed ({exc}); falling back.")
        return None
