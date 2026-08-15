"""Metropolis-Hastings search over the edit window.

Round-synchronized across all chains: every chain proposes one substitution
per round, and all proposals are scored together in a single ESM2 call plus
a single ProteinMPNN call -- so a `steps`-round search costs `steps * 2`
Modal calls total, not `steps * num_chains * 2`, while still recomputing
P(aa | current sequence) fresh every round via the single-pass approximation
in oracle/scoring.py (no stale precomputed table, unlike a position-
independent design).

Meant to be called from inside `oracle/modal_app.py`'s Modal function, so
these per-round calls to the already-deployed esm2/proteinmpnn services
happen container-to-container instead of laptop-to-Modal -- see that module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from epigen.pipeline.fold_invert_refold.run import (
    CompensatoryCandidate,
    FoldedStructure,
    propose_compensatory_mutations,
    refold_and_gate,
)
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
    """Sum of the weighted combined per-position score over `window_positions`, for the AA
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


@dataclass(frozen=True)
class MCMCCandidate:
    """One final candidate sequence, ranked by combined oracle score."""

    sequence: str
    combined_score: float
    passed_structural_check: bool | None  # None when refold_every wasn't used


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
    refold_every: int | None = None,
    candidate_num: int = 10,
    seed: int | None = None,
) -> list[MCMCCandidate]:
    """Run `num_starting_points * chains_per_start` round-synchronized MCMC chains and
    return the top `candidate_num` unique sequences by combined score.

    Starting points: `folded`'s own sequence plus `num_starting_points - 1`
    diverse warm starts sampled via stage 1's `propose_compensatory_mutations`
    (reused, not reimplemented).

    `refold_every` doesn't gate individual rounds -- instead it's applied
    once, as a post-hoc structural safety net: the top `candidate_num * 3`
    pre-filter candidates are refolded and TM-gated via stage 1's
    `refold_and_gate`, and only ones that pass are kept.
    `passed_structural_check` is `None` when this step is skipped.
    """
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
        _ChainState(sequence=start_seq, rng=random.Random(rng_master.randint(0, 2**31 - 1)))
        for start_seq in starting_sequences
        for _ in range(chains_per_start)
    ]

    # Round 0: score every chain's starting sequence once, to seed `score`.
    esm2_batch = position_scores_esm2_batch([c.sequence for c in chains])
    pmpnn_batch = position_scores_proteinmpnn_batch(folded.structure, [c.sequence for c in chains])
    for chain, esm2_scores, pmpnn_scores in zip(chains, esm2_batch, pmpnn_batch, strict=True):
        chain.score = _window_score(chain.sequence, window_positions, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)

    best_score_by_sequence: dict[str, float] = {c.sequence: c.score for c in chains}

    for _ in range(steps):
        proposals = []
        for chain in chains:
            pos = chain.rng.choice(window_positions)
            new_aa = chain.rng.choice(CANONICAL_AA)
            proposals.append(chain.sequence[: pos - 1] + new_aa + chain.sequence[pos:])

        esm2_batch = position_scores_esm2_batch(proposals)
        pmpnn_batch = position_scores_proteinmpnn_batch(folded.structure, proposals)

        for chain, proposed_seq, esm2_scores, pmpnn_scores in zip(chains, proposals, esm2_batch, pmpnn_batch, strict=True):
            proposed_score = _window_score(proposed_seq, window_positions, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
            delta = proposed_score - chain.score
            accept = delta >= 0 or chain.rng.random() < pow(2.718281828, delta / temperature)
            if accept:
                chain.sequence, chain.score = proposed_seq, proposed_score
            if chain.sequence not in best_score_by_sequence or chain.score > best_score_by_sequence[chain.sequence]:
                best_score_by_sequence[chain.sequence] = chain.score

    ranked = sorted(best_score_by_sequence.items(), key=lambda kv: kv[1], reverse=True)

    if refold_every is None:
        top = ranked[:candidate_num]
        return [MCMCCandidate(sequence=seq, combined_score=score, passed_structural_check=None) for seq, score in top]

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
        )
        for r in validated[:candidate_num]
    ]
