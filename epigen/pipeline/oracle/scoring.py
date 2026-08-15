"""Full positional mutational scan from two independent experts.

Both ESM2 and ProteinMPNN return per-position log-probabilities over their
vocab from a *single* scoring call on the starting sequence/structure -- this
is already a full positional scan, so "score edits everywhere" needs exactly
two Modal calls total, not one per candidate point mutation.
"""

from __future__ import annotations

from proto_tools import (
    ESM2ScoringConfig,
    ESM2ScoringInput,
    ProteinMPNNScoringConfig,
    ProteinMPNNScoringInput,
    SequenceStructurePair,
    run_esm2_score,
    run_proteinmpnn_score,
)
from proto_tools.entities.structures import Structure

DEVICE = "modal"

# Canonical 20-AA order every expert's logits get reindexed into before combining.
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
    """Per-position, per-amino-acid pseudo-log-likelihood from ESM2 (one Modal call)."""
    config = ESM2ScoringConfig(device=DEVICE, model_checkpoint=model_checkpoint, return_logits=True)
    output = run_esm2_score(ESM2ScoringInput(sequences=[sequence]), config)
    score = output.scores[0]
    return _reindex(score.logits, score.vocab)


def position_scores_proteinmpnn(structure: Structure, sequence: str) -> PositionScores:
    """Per-position, per-amino-acid structure-conditioned log-likelihood from ProteinMPNN
    (one Modal call)."""
    config = ProteinMPNNScoringConfig(device=DEVICE, return_logits=True)
    pair = SequenceStructurePair(sequence=sequence, structure=structure)
    output = run_proteinmpnn_score(ProteinMPNNScoringInput(sequence_structure_pairs=[pair]), config)
    score = output.scores[0]
    return _reindex(score.logits, score.vocab)
