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


def window_log_prob_batch(
    nt_sequences: list[str],
    window_aa_positions: list[int],
    *,
    model_checkpoint: str = DEFAULT_CHECKPOINT,
) -> list[float]:
    """Sum of Evo2's own-token log-probability over the codons at `window_aa_positions`,
    for every sequence, in one Modal call.

    `avg_log_likelihood` (used by `nt_sequence_score_batch` above, and what
    `oracle.mcmc.run_mcmc_search` used to weight into `combined_score`) is a
    mean over the *entire* nt sequence -- mostly unchanged, fixed-outside-
    the-window positions that dilute the signal MCMC actually cares about.
    ESM2 and ProteinMPNN don't have this problem: `oracle.mcmc.window_score`
    only ever sums their per-position tables over `window_positions`. This
    function gives Evo2 the same treatment, using `Evo2ScoringConfig`'s
    `return_logits=True` -- a `(seq_len, vocab_size)` per-nucleotide table,
    the same shape ESM2/ProteinMPNN's `return_logits=True` output already is
    (see `oracle.scoring`) -- and reading off the log-probability Evo2
    assigned to whichever nucleotide is actually present at each position of
    each window codon (`aa_position -> nt_sequence[3*(aa_position-1) : +3]`,
    matching `oracle.codon`'s AA-position/codon-triplet convention), summed.

    Unlike ESM2/ProteinMPNN's tables, this is *not* reusable across
    different candidate sequences the way `orchestrate.py` reuses a single
    `esm2_scores`/`pmpnn_scores` table computed once on `edit_only`: Evo2 is
    autoregressive, so its logit at position t is conditioned on every
    nucleotide before t. A table computed for one sequence is only valid for
    that exact sequence -- call this fresh per batch of actual candidates,
    same as `nt_sequence_score_batch` already is.
    """
    config = Evo2ScoringConfig(device=DEVICE, model_checkpoint=model_checkpoint, return_logits=True)
    output = run_evo2_score(Evo2ScoringInput(sequences=nt_sequences), config)
    totals = []
    for nt_sequence, score in zip(nt_sequences, output.scores, strict=True):
        vocab_index = {token: i for i, token in enumerate(score.vocab)}
        total = 0.0
        for aa_position in window_aa_positions:
            codon_start = 3 * (aa_position - 1)
            for nt_position in range(codon_start, codon_start + 3):
                actual_nt = nt_sequence[nt_position]
                total += score.logits[nt_position][vocab_index[actual_nt]]
        totals.append(total)
    return totals
