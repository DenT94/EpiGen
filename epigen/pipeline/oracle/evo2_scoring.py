"""Evo2 (DNA-level, causal) sequence log-likelihood as a third oracle expert.

Unlike ESM2/ProteinMPNN's per-position AA tables, Evo2 gives one scalar
`avg_log_likelihood` per whole nt sequence (causal: sum of P(x_t | x_<t) over
the sequence, divided by length) -- so it enters the combined MCMC score as a
single additive term per candidate, not a per-position lookup. Requires the
candidate's nt sequence (see oracle/codon.py for AA<->nt bookkeeping).

`evo2_1b_base` is the default: smallest/fastest checkpoint, appropriate for a
short synthetic construct with no surrounding genomic context to exploit --
Evo2's distinguishing long-context capability isn't exercised here anyway
(see the design discussion in chat / commit history), so there's no accuracy
reason to prefer a larger checkpoint for this use case.
"""

from __future__ import annotations

from proto_tools import Evo2ScoringConfig, Evo2ScoringInput, run_evo2_score

DEVICE = "modal"
DEFAULT_CHECKPOINT = "evo2_1b_base"


def nt_sequence_score(nt_sequence: str, *, model_checkpoint: str = DEFAULT_CHECKPOINT) -> float:
    """Evo2's avg_log_likelihood for one nt sequence (one Modal call)."""
    return nt_sequence_score_batch([nt_sequence], model_checkpoint=model_checkpoint)[0]


def nt_sequence_score_batch(nt_sequences: list[str], *, model_checkpoint: str = DEFAULT_CHECKPOINT) -> list[float]:
    """Evo2's avg_log_likelihood for every sequence, in one Modal call.

    `avg_log_likelihood` (mean per-token, not the raw sum) is used so scores
    stay comparable across candidates even if nt sequences ever differ in
    length -- doesn't happen for pure substitutions, but keeps this safe.
    """
    config = Evo2ScoringConfig(device=DEVICE, model_checkpoint=model_checkpoint)
    output = run_evo2_score(Evo2ScoringInput(sequences=nt_sequences), config)
    return [score.avg_log_likelihood for score in output.scores]
