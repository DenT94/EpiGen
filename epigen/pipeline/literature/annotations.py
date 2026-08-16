"""Functional + structural annotation ranges for an input protein, via Paperclip.

Two things this hands back for any construct sequence (CLAUDE.md's "Paperclip
for literature search" stack entry, scoped to what step 3 -- edit-window
choice -- needs):

1. Functional annotations: residues UniProt records as *doing* something
   (active site, binding site, disulfide bond, short linear motif, ...).
2. Structural motifs: secondary structure elements UniProt records
   (helix, beta strand, turn, coiled coil) -- a real beta barrel or similar
   higher-order fold shows up here as its constituent strands, not as one
   named barrel (UniProt doesn't curate fold-level motif names; DSSP-on-
   structure was the alternative and was explicitly not chosen for this).

Both come back in the same numbering as the *input* `sequence` (not raw
UniProt numbering, which includes cleaved signal peptides etc.) so a caller
can directly ask "is position N annotated?" against its own edit window.

Source-of-truth precedence for resolving `sequence` to a UniProt accession:
an explicit `pdb_id` (trusted, matches what `structure_source.get_structure`
already resolved) beats a substring search over UniProt sequences, which is
a best-effort fallback and may pick the wrong paralog/species for a short or
highly conserved query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from epigen.pipeline.literature import cache
from epigen.pipeline.literature.paperclip_client import PaperclipError, run_protein_sql

logger = logging.getLogger(__name__)

Kind = Literal["functional", "structural"]

# UniProt uniprot_v.features.feature_type values this module surfaces, and how.
# Everything else (Chain, Domain, Signal, Natural variant, Sequence conflict,
# ...) is family/provenance metadata, not a per-residue functional or
# structural claim -- left out to keep edit-window scoring signal-only.
FUNCTIONAL_FEATURE_TYPES = {
    "Active site",
    "Binding site",
    "Metal binding",
    "DNA binding",
    "Site",
    "Motif",  # UniProt's sense: short linear functional motif, e.g. a localization signal
    "Disulfide bond",
    "Cross-link",
    "Modified residue",
    "Glycosylation",
    "Lipidation",
}
STRUCTURAL_FEATURE_TYPES = {"Helix", "Beta strand", "Turn", "Coiled coil"}


@dataclass(frozen=True)
class PaperReference:
    """One candidate supporting paper for an annotation, from a Paperclip search --
    best-effort provenance (see `literature.papers`), not a verified citation."""

    title: str
    authors: str
    doc_id: str  # e.g. PMC11783178
    source: str  # e.g. PMC
    date: str
    url: str


@dataclass(frozen=True)
class AccessionMetadata:
    """UniProt identity of the accession an input sequence was resolved to -- enough
    to build a literature search query (see `literature.papers.attach_papers`)."""

    accession: str
    protein_name: str
    gene_name: str
    organism_common: str


@dataclass(frozen=True)
class AnnotationRange:
    """One annotated residue range, in the *input construct's* 1-indexed numbering."""

    label: str
    start: int
    end: int
    kind: Kind
    feature_type: str  # raw UniProt feature_type (e.g. "Active site"), for query-building
    papers: list[PaperReference] = field(default_factory=list)  # filled in by literature.papers.attach_papers


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def resolve_accession(sequence: str, pdb_id: str | None = None) -> str | None:
    """UniProt accession for `sequence` -- `pdb_id`'s cross-reference if given, else a
    substring search over UniProt sequences. `None` if neither resolves. Exposed
    publicly (not just used internally by `get_annotations`) so a caller like
    `literature.papers.attach_papers` can get the same accession without a second,
    possibly-divergent resolution.

    Disk-cached by `(pdb_id, sequence)` (`literature.cache.load_accession`/
    `save_accession`) -- this resolution is a property of the protein, not of any
    particular pipeline run, so it's cached permanently and independently of
    `st.cache_data` (see that module's docstring for why).
    """
    cache_hit, cached_accession = cache.load_accession(pdb_id, sequence)
    if cache_hit:
        return cached_accession
    accession = _resolve_accession_uncached(sequence, pdb_id)
    cache.save_accession(pdb_id, sequence, accession)
    return accession


def _resolve_accession_uncached(sequence: str, pdb_id: str | None) -> str | None:
    if pdb_id is not None:
        rows = run_protein_sql(
            "SELECT DISTINCT uniprot_accession FROM pdb_v.polymer_entities "
            f"WHERE entry_id = '{_escape_sql_literal(pdb_id.upper())}' AND uniprot_accession IS NOT NULL"
        )
        if rows:
            accession = rows[0]["uniprot_accession"]
            logger.info(f"Resolved accession {accession!r} from PDB entry {pdb_id!r}.")
            return accession
        logger.warning(f"PDB entry {pdb_id!r} has no UniProt cross-reference; falling back to sequence search.")

    rows = run_protein_sql(
        "SELECT accession, length FROM uniprot_v.protein_sequences "
        f"WHERE sequence LIKE '%{_escape_sql_literal(sequence)}%' ORDER BY length ASC LIMIT 1"
    )
    if not rows:
        logger.warning("No UniProt entry's sequence contains the input construct; no annotations available.")
        return None
    accession = rows[0]["accession"]
    logger.info(f"Resolved accession {accession!r} by sequence substring match.")
    return accession


def _fetch_accession_data(accession: str) -> tuple[str, list[dict[str, str]], dict[str, str]]:
    cached = cache.load(accession)
    if cached is not None:
        return cached["sequence"], cached["features"], cached.get("metadata", {})

    seq_rows = run_protein_sql(f"SELECT sequence FROM uniprot_v.protein_sequences WHERE accession = '{accession}'")
    if not seq_rows:
        raise PaperclipError(f"accession {accession!r} has no row in uniprot_v.protein_sequences")
    uniprot_sequence = seq_rows[0]["sequence"]

    features = run_protein_sql(
        "SELECT feature_type, start_pos, end_pos, description FROM uniprot_v.features "
        f"WHERE accession = '{accession}' ORDER BY start_pos"
    )

    protein_rows = run_protein_sql(
        f"SELECT protein_name, gene_name, organism_common FROM uniprot_v.proteins WHERE accession = '{accession}'"
    )
    metadata = protein_rows[0] if protein_rows else {}

    cache.save(accession, sequence=uniprot_sequence, features=features, metadata=metadata)
    return uniprot_sequence, features, metadata


def get_accession_metadata(sequence: str, *, pdb_id: str | None = None) -> AccessionMetadata | None:
    """UniProt identity (name/gene/organism) for `sequence`'s resolved accession, or
    `None` if it can't be resolved. Separate from `get_annotations` so a caller only
    pays for this (one extra cached SQL row) when it actually wants to build a
    literature search query, e.g. via `literature.papers.attach_papers`.
    """
    accession = resolve_accession(sequence, pdb_id)
    if accession is None:
        return None
    try:
        _, _, metadata = _fetch_accession_data(accession)
    except PaperclipError as exc:
        logger.warning(f"Paperclip lookup failed; no accession metadata: {exc}")
        return None
    if not metadata:
        return None
    return AccessionMetadata(
        accession=accession,
        protein_name=metadata.get("protein_name", ""),
        gene_name=metadata.get("gene_name", ""),
        organism_common=metadata.get("organism_common", ""),
    )


def _label(feature_type: str, description: str, start: int, end: int) -> str:
    base = f"{feature_type} ({description})" if description else feature_type
    return base if start != end else f"{base} @{start}"


def get_annotations(sequence: str, *, pdb_id: str | None = None) -> list[AnnotationRange]:
    """Functional + structural annotation ranges for `sequence`, in its own numbering.

    Resolution: `pdb_id` (if given) -> UniProt accession via the PDB cross-
    reference; otherwise a substring search over UniProt sequences. Returns
    `[]` (with a logged warning, never raises) if no accession, or no local
    match for `sequence` within the resolved accession's UniProt sequence,
    can be found -- literature annotation is an advisory signal, not a gate,
    so a miss should degrade gracefully rather than break the pipeline.
    """
    try:
        accession = resolve_accession(sequence, pdb_id)
        if accession is None:
            return []
        uniprot_sequence, raw_features, _ = _fetch_accession_data(accession)
    except PaperclipError as exc:
        logger.warning(f"Paperclip lookup failed; returning no annotations: {exc}")
        return []

    offset = uniprot_sequence.find(sequence)  # 0-indexed start of `sequence` within the full UniProt sequence
    if offset == -1:
        logger.warning(
            f"Input sequence is not an exact substring of {accession!r}'s UniProt sequence "
            "(construct may include mutations already applied); no annotations available."
        )
        return []

    ranges: list[AnnotationRange] = []
    for row in raw_features:
        feature_type = row["feature_type"]
        if feature_type in FUNCTIONAL_FEATURE_TYPES:
            kind: Kind = "functional"
        elif feature_type in STRUCTURAL_FEATURE_TYPES:
            kind = "structural"
        else:
            continue
        start = int(row["start_pos"]) - offset
        end = int(row["end_pos"]) - offset
        if start < 1 or end > len(sequence):
            continue  # outside the construct (e.g. a signal-peptide-region feature)
        ranges.append(
            AnnotationRange(
                label=_label(feature_type, row["description"], start, end),
                start=start, end=end, kind=kind, feature_type=feature_type,
            )
        )

    return sorted(ranges, key=lambda r: (r.start, r.end))


def flag_positions(ranges: list[AnnotationRange], positions: list[int]) -> list[AnnotationRange]:
    """Which `ranges` overlap any of `positions` -- e.g. an edit position or a candidate
    MCMC search window, to flag "this touches a known functional/structural residue"
    before trusting the edit-window choice. Positions and ranges must share numbering
    (both in the same construct's coordinates, as `get_annotations` returns)."""
    position_set = set(positions)
    return [r for r in ranges if position_set.intersection(range(r.start, r.end + 1))]
