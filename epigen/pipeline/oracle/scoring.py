"""Per-position amino-acid log-probabilities from two independent experts.

ESM2's own "score" operation computes the *true* masked-marginal PLL:
P(aa_i | sequence, position i masked), one forward pass per position (L
passes for a length-L sequence). We instead approximate
P(aa_i | sequence, position i masked) =~ P(aa_i | sequence) -- a single
unmasked forward pass, reading each position's own-token logits directly
(`run_esm2_embeddings(..., return_logits=True)`). This turns L passes into
1, at the cost of the model "seeing" the residue it's predicting (a known,
accepted approximation for this kind of screening). ProteinMPNN's
structure-conditioned score is already single-pass (autoregressive
teacher-forcing), so it needs no equivalent change.

`_batch` variants score many sequences in one call, which is what lets MCMC
recompute every step without one network round-trip per chain per step (see
oracle/mcmc.py's round-synchronized design).
"""

from __future__ import annotations

from proto_tools import (
    ESM2EmbeddingsConfig,
    ESM2EmbeddingsInput,
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    SequenceStructurePair,
    run_esm2_embeddings,
    run_proteinmpnn_score,
)
from proto_tools.entities.structures import Structure

DEVICE = "modal"

# ESM2's fixed 20-canonical-AA vocab order (proto_tools' AMINO_ACIDS_LIST) -- confirmed
# identical for both the "score" and "embeddings" operations (same underlying tokenizer
# gather), but only "score"'s output echoes it back explicitly, so embeddings callers
# must know it out-of-band. See ESM2ScoringConfig's docstring for the citation.
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

# Position (1-indexed) -> {amino acid: log-probability}.
PositionScores = list[dict[str, float]]


def _reindex(logits: list[list[float]], vocab: list[str]) -> PositionScores:
    """Reindex a (seq_len, vocab_size) logits array into CANONICAL_AA order per position.

    Drops any vocab entries outside the 20 canonical amino acids (e.g.
    ProteinMPNN's 'X').
    """
    vocab_index = {aa: i for i, aa in enumerate(vocab)}
    return [
        {aa: row[vocab_index[aa]] for aa in CANONICAL_AA if aa in vocab_index}
        for row in logits
    ]


def position_scores_esm2(sequence: str, *, model_checkpoint: str = "esm2_t33_650M_UR50D") -> PositionScores:
    """P(aa | sequence) per position, single unmasked forward pass (one Modal call)."""
    return position_scores_esm2_batch([sequence], model_checkpoint=model_checkpoint)[0]


def position_scores_esm2_batch(
    sequences: list[str], *, model_checkpoint: str = "esm2_t33_650M_UR50D"
) -> list[PositionScores]:
    """P(aa | sequence) per position for every sequence, in one Modal call."""
    config = ESM2EmbeddingsConfig(device=DEVICE, model_checkpoint=model_checkpoint, return_logits=True)
    output = run_esm2_embeddings(ESM2EmbeddingsInput(sequences=sequences), config)
    return [_reindex(result.logits, CANONICAL_AA) for result in output.results]


def position_scores_proteinmpnn(structure: Structure, sequence: str) -> PositionScores:
    """Structure-conditioned P(aa | structure, context) per position (one Modal call)."""
    return position_scores_proteinmpnn_batch(structure, [sequence])[0]


def position_scores_proteinmpnn_batch(structure: Structure, sequences: list[str]) -> list[PositionScores]:
    """Structure-conditioned P(aa | structure, context) per position, for every sequence
    against the same `structure`, in one Modal call."""
    config = ProteinMPNNScoringConfig(device=DEVICE, return_logits=True)
    pairs = [SequenceStructurePair(sequence=seq, structure=structure) for seq in sequences]
    output = run_proteinmpnn_score(ProteinMPNNScoringInput(sequence_structure_pairs=pairs), config)
    return [_reindex(score.logits, score.vocab) for score in output.scores]
