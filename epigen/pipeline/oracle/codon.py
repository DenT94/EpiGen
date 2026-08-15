"""Codon-level nucleotide sequence tracking, alongside the amino-acid sequence.

Needed to score candidates with Evo2 (a DNA-level causal model) as a third
oracle expert: whenever MCMC proposes an amino-acid substitution, the
underlying nt codon at that position must change too, so the AA and nt
sequences stay in sync throughout the search.
"""

from __future__ import annotations

# Standard genetic code (DNA codon -> one-letter AA; '*' = stop). Universal,
# not organism-specific -- no ambiguity here.
CODON_TO_AA: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}  # fmt: skip

assert len(CODON_TO_AA) == 64, f"expected 64 codons, got {len(CODON_TO_AA)}"

# One deterministic codon per amino acid for reverse translation -- commonly
# cited E. coli high-expression codon-usage-bias defaults, NOT a specific
# measured table for any particular construct. This exists so a WT AA-only
# sequence has *some* consistent nt sequence to carry through the pipeline;
# swap in a real (measured or codon-optimized) coding sequence when one is
# available, via reverse_translate()'s codon_table override.
PREFERRED_CODON: dict[str, str] = {
    "A": "GCG", "R": "CGT", "N": "AAC", "D": "GAT", "C": "TGC",
    "Q": "CAG", "E": "GAA", "G": "GGT", "H": "CAT", "I": "ATT",
    "L": "CTG", "K": "AAA", "M": "ATG", "F": "TTT", "P": "CCG",
    "S": "AGC", "T": "ACC", "W": "TGG", "Y": "TAT", "V": "GTG",
}  # fmt: skip

# Self-check: every preferred codon must translate back to the AA it's
# supposed to encode -- catches a transcription error in either table above
# immediately, at import time, rather than as a silent wrong answer later.
for _aa, _codon in PREFERRED_CODON.items():
    assert CODON_TO_AA[_codon] == _aa, (
        f"PREFERRED_CODON[{_aa!r}] = {_codon!r} translates to {CODON_TO_AA[_codon]!r}, not {_aa!r}"
    )
del _aa, _codon


def translate(nt_sequence: str) -> str:
    """Translate an in-frame nt sequence (length a multiple of 3) to its AA sequence.

    Stops at the first in-frame stop codon, if any (not included in the output).
    """
    if len(nt_sequence) % 3 != 0:
        raise ValueError(f"nt_sequence length {len(nt_sequence)} is not a multiple of 3.")
    residues = []
    for i in range(0, len(nt_sequence), 3):
        aa = CODON_TO_AA[nt_sequence[i : i + 3]]
        if aa == "*":
            break
        residues.append(aa)
    return "".join(residues)


def reverse_translate(aa_sequence: str, codon_table: dict[str, str] = PREFERRED_CODON) -> str:
    """One deterministic codon per residue, via `codon_table` (default: PREFERRED_CODON)."""
    return "".join(codon_table[aa] for aa in aa_sequence)


def apply_aa_substitution_to_nt(
    nt_sequence: str,
    position: int,
    new_aa: str,
    codon_table: dict[str, str] = PREFERRED_CODON,
) -> str:
    """Replace the codon at 1-indexed AA `position` with `codon_table[new_aa]`.

    This is the "whenever an AA mutation is proposed, carry the underlying nt
    codon mutation" operation MCMC needs to keep the nt sequence in sync with
    the AA sequence.
    """
    start = 3 * (position - 1)
    return nt_sequence[:start] + codon_table[new_aa] + nt_sequence[start + 3 :]


def apply_aa_substitutions_to_nt(
    nt_sequence: str,
    start_position: int,
    new_residues: str,
    codon_table: dict[str, str] = PREFERRED_CODON,
) -> str:
    """Replace the codons for a contiguous run of residues, e.g. a multi-residue edit
    like "substitute residues 20:27 with WHSPRAL" (`start_position=20`, `new_residues="WHSPRAL"`).

    Same-length substitution only (no insertion/deletion) -- `new_residues`
    replaces exactly `len(new_residues)` consecutive residues starting at
    `start_position`. Just `apply_aa_substitution_to_nt` applied once per
    residue; this exists so callers doing a multi-residue edit don't have to
    hand-loop it themselves.
    """
    for offset, aa in enumerate(new_residues):
        nt_sequence = apply_aa_substitution_to_nt(nt_sequence, start_position + offset, aa, codon_table)
    return nt_sequence
