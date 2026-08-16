"""Model-based reverse translation via CodonFM (Encodon), used when no real coding
sequence can be found for a construct (see `oracle.genbank_lookup`/`oracle.nt_resolution`).

CodonFM's own sampling tool resamples codons *unconditioned* on which amino acid should
result -- it can introduce a missense change, it just can't introduce a new stop codon
(see proto_tools' CodonFM README). So getting an AA-preserving nt sequence out of it
means starting from *some* valid encoding (`oracle.codon.reverse_translate`'s
deterministic per-amino-acid table), letting CodonFM resample a subset of codons at a
time, then keeping its resampled codon only where doing so is still synonymous with the
original amino acid and leaving the deterministic codon everywhere it isn't.

Masking *everything* in one pass (this module's first cut) doesn't work: with every
codon masked simultaneously, CodonFM has zero flanking context to condition on -- it
can't know which amino acid belongs at a masked position, so it almost never predicts a
synonymous codon and the result degenerates to the naive table with extra steps
(verified empirically: 0/45 nt positions differed from `reverse_translate` on a real
run). Masking only a moderate fraction per round, iterated over several rounds, keeps
most of the sequence unmasked as real context each time -- flanking codon-usage
statistics CodonFM can actually use -- while still giving every position a chance to be
resampled over the course of `rounds` passes.
"""

from __future__ import annotations

import random

from proto_tools import CodonFMSampleConfig, CodonFMSampleInput, run_codonfm_sample

from epigen.pipeline.oracle.codon import CODON_TO_AA, reverse_translate, translate

DEVICE = "modal"
DEFAULT_CHECKPOINT = "encodon_80m"
DEFAULT_ROUNDS = 4
DEFAULT_MASK_FRACTION = 0.25


def reverse_translate_codonfm(
    aa_sequence: str,
    *,
    seed: int | None = None,
    model_checkpoint: str = DEFAULT_CHECKPOINT,
    rounds: int = DEFAULT_ROUNDS,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> str:
    """A codon-model-informed nt sequence, guaranteed to translate back to `aa_sequence`
    exactly (see module docstring for how the guarantee is enforced, and why this masks
    incrementally rather than all at once)."""
    n = len(aa_sequence)
    rng = random.Random(seed)
    codons = [reverse_translate(aa_sequence)[3 * i : 3 * i + 3] for i in range(n)]

    num_to_mask = max(1, round(n * mask_fraction))
    for _ in range(rounds):
        positions = rng.sample(range(n), min(num_to_mask, n))
        masked_codons = list(codons)
        for p in positions:
            masked_codons[p] = "___"
        masked_nt = "".join(masked_codons)

        config = CodonFMSampleConfig(
            device=DEVICE, model_checkpoint=model_checkpoint, seed=rng.randint(0, 2**31 - 1)
        )
        output = run_codonfm_sample(CodonFMSampleInput(sequences=[masked_nt]), config)
        resampled_nt = next(iter(output))

        for p in positions:
            candidate_codon = resampled_nt[3 * p : 3 * p + 3]
            if CODON_TO_AA.get(candidate_codon) == aa_sequence[p]:
                codons[p] = candidate_codon  # model's choice is synonymous -- keep it over the naive default

    result = "".join(codons)
    assert translate(result) == aa_sequence, "reverse_translate_codonfm produced a non-synonymous nt sequence"
    return result
