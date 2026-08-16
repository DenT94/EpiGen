"""Single entry point for `orchestrate.py` to get a coding sequence for the wild-type
protein whenever Evo2 scoring needs one and the caller hasn't supplied one directly.

Resolution order:
  1. An explicit `wt_nt_sequence` (the caller already knows the real construct's
     codons) -- always trusted as-is, no lookup.
  2. A resolved PDB entry (`pdb_id` -- pass `original.pdb_id` from
     `structure_source.get_structure`, so this also covers a PDB match
     auto-discovered by sequence search, not just a caller-given ID): look up the
     real coding sequence via its GenBank cross-reference
     (`oracle.genbank_lookup`), which is only trusted once verified by
     re-translating it and checking it matches `wt_sequence` exactly.
  3. Otherwise (no PDB, or no GenBank CDS that verifies): CodonFM-based model
     reverse translation (`oracle.codonfm_translate`) -- still guaranteed
     AA-correct, just not backed by a real deposited coding sequence.

Never returns `None` -- one of the three sources above always produces a usable nt
sequence, so a caller doesn't need its own None-handling fallback path.
"""

from __future__ import annotations

import logging

from epigen.pipeline.oracle.codonfm_translate import reverse_translate_codonfm
from epigen.pipeline.oracle.genbank_lookup import fetch_pdb_genbank_nt_sequence

logger = logging.getLogger(__name__)


def resolve_wt_nt_sequence(
    wt_sequence: str,
    *,
    wt_nt_sequence: str | None = None,
    pdb_id: str | None = None,
    seed: int | None = None,
) -> tuple[str, str]:
    """Returns `(nt_sequence, source)`, `source` one of `"given"`/`"genbank"`/`"codonfm"` --
    surfaced so a caller (e.g. `EndToEndResult`) can show which kind of coding sequence
    Evo2 was actually scored against, rather than silently treating all three the same."""
    if wt_nt_sequence is not None:
        return wt_nt_sequence, "given"
    if pdb_id is not None:
        genbank_nt = fetch_pdb_genbank_nt_sequence(pdb_id, wt_sequence)
        if genbank_nt is not None:
            return genbank_nt, "genbank"
    logger.info(f"No verified GenBank CDS for {wt_sequence[:10]!r}...; reverse-translating with CodonFM instead.")
    return reverse_translate_codonfm(wt_sequence, seed=seed), "codonfm"
