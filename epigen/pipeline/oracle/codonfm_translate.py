"""Model-based reverse translation via CodonFM (Encodon), used when no real coding
sequence can be found for a construct (see `oracle.genbank_lookup`/`oracle.nt_resolution`).

CodonFM's own sampling tool resamples codons *unconditioned* on which amino acid should
result -- it can introduce a missense change, it just can't introduce a new stop codon
(see proto_tools' CodonFM README). So getting an AA-preserving nt sequence out of it
means starting from *some* valid encoding (`oracle.codon.reverse_translate`'s
deterministic per-amino-acid table), letting CodonFM resample every codon in one pass,
then keeping its resampled codon only where doing so is still synonymous with the
original amino acid and falling back to the deterministic codon everywhere it isn't.
The result is model-informed (real coding-sequence codon-usage statistics learned from
data, not a fixed per-amino-acid table) wherever CodonFM agreed with the target protein,
and provably still correct (`translate(result) == aa_sequence`) everywhere else.
"""

from __future__ import annotations

from proto_tools import CodonFMSampleConfig, CodonFMSampleInput, run_codonfm_sample

from epigen.pipeline.oracle.codon import CODON_TO_AA, reverse_translate, translate

DEVICE = "modal"
DEFAULT_CHECKPOINT = "encodon_80m"


def reverse_translate_codonfm(
    aa_sequence: str,
    *,
    seed: int | None = None,
    model_checkpoint: str = DEFAULT_CHECKPOINT,
) -> str:
    """A codon-model-informed nt sequence, guaranteed to translate back to `aa_sequence`
    exactly (see module docstring for how the guarantee is enforced)."""
    naive_nt = reverse_translate(aa_sequence)
    masked_nt = "___" * len(aa_sequence)  # one whole-codon mask per residue -- resample all of them
    config = CodonFMSampleConfig(device=DEVICE, model_checkpoint=model_checkpoint, seed=seed)
    output = run_codonfm_sample(CodonFMSampleInput(sequences=[masked_nt]), config)
    resampled_nt = next(iter(output))

    codons = []
    for i, aa in enumerate(aa_sequence):
        start = 3 * i
        resampled_codon = resampled_nt[start : start + 3]
        codons.append(resampled_codon if CODON_TO_AA.get(resampled_codon) == aa else naive_nt[start : start + 3])
    result = "".join(codons)
    assert translate(result) == aa_sequence, "reverse_translate_codonfm produced a non-synonymous nt sequence"
    return result
