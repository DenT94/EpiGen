"""Metropolis-Hastings search over the edit window.

Round-synchronized across all chains: every chain proposes one substitution
per round, and all proposals are scored together in a single ESM2 call, a
single ProteinMPNN call, and (when a starting nt sequence is supplied) a
single Evo2 call -- so a `steps`-round search costs `steps * (2 or 3)` Modal
calls total, not `steps * num_chains * (2 or 3)`, while still recomputing
each expert's score fresh every round (no stale precomputed table, unlike a
position-independent design).

ESM2/ProteinMPNN score per-position AA log-probs, summed over the window
(see `_window_score`). Evo2 is different in kind -- a single whole-nt-sequence
causal log-likelihood, not a per-position AA table -- so it enters as a third
additive term rather than another per-position sum; see oracle/evo2_scoring.py
and oracle/codon.py for why and how the nt sequence is carried alongside the
AA sequence.

Meant to be called from inside `oracle/modal_app.py`'s Modal function, so
these per-round calls to the already-deployed esm2/proteinmpnn/evo2 services
happen container-to-container instead of laptop-to-Modal -- see that module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from epigen.pipeline.fold_invert_refold.run import (
    CompensatoryCandidate,
    FoldedStructure,
    propose_compensatory_mutations,
    refold_and_gate,
)
from epigen.pipeline.oracle.codon import apply_aa_substitution_to_nt
from epigen.pipeline.oracle.evo2_scoring import nt_sequence_score_batch
from epigen.pipeline.oracle.scoring import (
    CANONICAL_AA,
    PositionScores,
    position_scores_esm2_batch,
    position_scores_proteinmpnn_batch,
)


def _window_score(
    sequence: str,
    window_positions: list[int],
    esm2_scores: PositionScores,
    pmpnn_scores: PositionScores,
    weight_esm2: float,
    weight_pmpnn: float,
) -> float:
    """Sum of the weighted per-position AA score over `window_positions`, for the AA
    actually present at each position in `sequence`."""
    total = 0.0
    for pos in window_positions:
        aa = sequence[pos - 1]
        total += weight_esm2 * esm2_scores[pos - 1].get(aa, float("-inf"))
        total += weight_pmpnn * pmpnn_scores[pos - 1].get(aa, float("-inf"))
    return total


@dataclass
class _ChainState:
    sequence: str
    rng: random.Random
    score: float = 0.0
    nt_sequence: str | None = None  # only tracked when Evo2 scoring is enabled


@dataclass(frozen=True)
class MCMCCandidate:
    """One final candidate sequence, ranked by combined oracle score."""

    sequence: str
    combined_score: float
    passed_structural_check: bool | None  # None when refold_every wasn't used
    nt_sequence: str | None = None  # only populated when Evo2 scoring was enabled


def run_mcmc_search(
    folded: FoldedStructure,
    window_positions: list[int],
    *,
    chain_id: str = "A",
    num_starting_points: int = 4,
    chains_per_start: int = 4,
    steps: int = 50,
    temperature: float = 1.0,
    weight_esm2: float = 0.5,
    weight_pmpnn: float = 0.5,
    nt_sequence: str | None = None,
    weight_evo2: float = 0.5,
    refold_every: int | None = None,
    candidate_num: int = 10,
    seed: int | None = None,
) -> list[MCMCCandidate]:
    """Run `num_starting_points * chains_per_start` round-synchronized MCMC chains and
    return the top `candidate_num` unique sequences by combined score.

    Starting points: `folded`'s own sequence plus `num_starting_points - 1`
    diverse warm starts sampled via stage 1's `propose_compensatory_mutations`
    (reused, not reimplemented).

    `nt_sequence`: `folded`'s coding sequence, in-frame, same length in
    codons as `folded.sequence`. When given, Evo2's whole-sequence
    avg_log_likelihood is added to the combined score (weighted by
    `weight_evo2`) and every accepted AA substitution updates the
    corresponding codon (see oracle/codon.py). When `None` (default), Evo2
    scoring is skipped entirely -- no nt bookkeeping, no extra Modal calls.

    `refold_every` doesn't gate individual rounds -- instead it's applied
    once, as a post-hoc structural safety net: the top `candidate_num * 3`
    pre-filter candidates are refolded and TM-gated via stage 1's
    `refold_and_gate`, and only ones that pass are kept.
    `passed_structural_check` is `None` when this step is skipped.
    """
    use_evo2 = nt_sequence is not None
    rng_master = random.Random(seed)

    starting_sequences = [folded.sequence]
    if num_starting_points > 1:
        warm_starts = propose_compensatory_mutations(
            folded,
            window_positions=window_positions,
            chain_id=chain_id,
            num_sequences=num_starting_points - 1,
            temperature=1.0,
            seed=seed,
        )
        starting_sequences.extend(c.sequence for c in warm_starts)

    chains = [
        _ChainState(
            sequence=start_seq,
            rng=random.Random(rng_master.randint(0, 2**31 - 1)),
            # Warm starts from ProteinMPNN change several residues at once relative to
            # `folded.sequence`; re-deriving each one's own nt sequence from its own AA
            # sequence (rather than patching folded's nt codon-by-codon) keeps every
            # chain's nt sequence consistent with its actual starting AA sequence.
            nt_sequence=(nt_sequence if start_seq == folded.sequence else _reverse_translate_matching(start_seq)) if use_evo2 else None,
        )
        for start_seq in starting_sequences
        for _ in range(chains_per_start)
    ]

    # Round 0: score every chain's starting sequence once, to seed `score`.
    esm2_batch = position_scores_esm2_batch([c.sequence for c in chains])
    pmpnn_batch = position_scores_proteinmpnn_batch(folded.structure, [c.sequence for c in chains])
    evo2_batch = nt_sequence_score_batch([c.nt_sequence for c in chains]) if use_evo2 else [0.0] * len(chains)
    for chain, esm2_scores, pmpnn_scores, evo2_score in zip(chains, esm2_batch, pmpnn_batch, evo2_batch, strict=True):
        chain.score = (
            _window_score(chain.sequence, window_positions, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
            + weight_evo2 * evo2_score
        )

    best_score_by_sequence: dict[str, float] = {c.sequence: c.score for c in chains}
    best_nt_by_sequence: dict[str, str | None] = {c.sequence: c.nt_sequence for c in chains}

    for _ in range(steps):
        proposals = []
        nt_proposals = []
        for chain in chains:
            pos = chain.rng.choice(window_positions)
            new_aa = chain.rng.choice(CANONICAL_AA)
            proposals.append(chain.sequence[: pos - 1] + new_aa + chain.sequence[pos:])
            if use_evo2:
                nt_proposals.append(apply_aa_substitution_to_nt(chain.nt_sequence, pos, new_aa))

        esm2_batch = position_scores_esm2_batch(proposals)
        pmpnn_batch = position_scores_proteinmpnn_batch(folded.structure, proposals)
        evo2_batch = nt_sequence_score_batch(nt_proposals) if use_evo2 else [0.0] * len(chains)

        for chain, proposed_seq, esm2_scores, pmpnn_scores, evo2_score, proposed_nt in zip(
            chains, proposals, esm2_batch, pmpnn_batch, evo2_batch, nt_proposals or [None] * len(chains), strict=True
        ):
            proposed_score = (
                _window_score(proposed_seq, window_positions, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
                + weight_evo2 * evo2_score
            )
            delta = proposed_score - chain.score
            accept = delta >= 0 or chain.rng.random() < pow(2.718281828, delta / temperature)
            if accept:
                chain.sequence, chain.score = proposed_seq, proposed_score
                if use_evo2:
                    chain.nt_sequence = proposed_nt
            if chain.sequence not in best_score_by_sequence or chain.score > best_score_by_sequence[chain.sequence]:
                best_score_by_sequence[chain.sequence] = chain.score
                best_nt_by_sequence[chain.sequence] = chain.nt_sequence

    ranked = sorted(best_score_by_sequence.items(), key=lambda kv: kv[1], reverse=True)

    if refold_every is None:
        top = ranked[:candidate_num]
        return [
            MCMCCandidate(sequence=seq, combined_score=score, passed_structural_check=None, nt_sequence=best_nt_by_sequence[seq])
            for seq, score in top
        ]

    pre_filter = ranked[: candidate_num * 3]
    pseudo_candidates = [CompensatoryCandidate(sequence=seq, perplexity=0.0, sequence_recovery=0.0) for seq, _ in pre_filter]
    refolded = refold_and_gate(pseudo_candidates, folded, seed=seed)
    validated = [r for r in refolded if r.passed_self_consistency_gate]
    validated.sort(key=lambda r: best_score_by_sequence[r.candidate.sequence], reverse=True)
    return [
        MCMCCandidate(
            sequence=r.candidate.sequence,
            combined_score=best_score_by_sequence[r.candidate.sequence],
            passed_structural_check=True,
            nt_sequence=best_nt_by_sequence[r.candidate.sequence],
        )
        for r in validated[:candidate_num]
    ]


def _reverse_translate_matching(aa_sequence: str) -> str:
    """Reverse-translate a warm-start AA sequence with the same preferred-codon table used
    for the reference nt sequence, so per-position codons stay comparable across chains."""
    from epigen.pipeline.oracle.codon import reverse_translate

    return reverse_translate(aa_sequence)
