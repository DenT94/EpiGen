"""Metropolis-Hastings search over the edit window, using the two-expert oracle
as a cheap, position-independent energy function (no per-step model calls).

Starting points and the optional structural safety net both reuse stage 1
(`epigen.pipeline.fold_invert_refold.run`) rather than reimplementing folding
or ProteinMPNN sampling.
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
from epigen.pipeline.oracle.scoring import CANONICAL_AA, PositionScores


def _contribution(
    position: int,
    aa: str,
    esm2_scores: PositionScores,
    pmpnn_scores: PositionScores,
    weight_esm2: float,
    weight_pmpnn: float,
) -> float:
    esm2_ll = esm2_scores[position - 1].get(aa, float("-inf"))
    pmpnn_ll = pmpnn_scores[position - 1].get(aa, float("-inf"))
    return weight_esm2 * esm2_ll + weight_pmpnn * pmpnn_ll


def combined_score(
    sequence: str,
    window_positions: list[int],
    esm2_scores: PositionScores,
    pmpnn_scores: PositionScores,
    *,
    weight_esm2: float = 0.5,
    weight_pmpnn: float = 0.5,
) -> float:
    """Sum of the combined per-position score over `window_positions` for `sequence`.

    Positions outside the window don't contribute -- the search never touches
    them, so they're irrelevant to ranking candidates against each other.
    """
    return sum(
        _contribution(pos, sequence[pos - 1], esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
        for pos in window_positions
    )


@dataclass
class ChainResult:
    """One MCMC chain's trajectory: every (sequence, combined_score) visited, including the start."""

    visited: list[tuple[str, float]] = field(default_factory=list)


def run_mcmc_chain(
    start_sequence: str,
    window_positions: list[int],
    esm2_scores: PositionScores,
    pmpnn_scores: PositionScores,
    *,
    steps: int = 500,
    temperature: float = 1.0,
    weight_esm2: float = 0.5,
    weight_pmpnn: float = 0.5,
    seed: int | None = None,
) -> ChainResult:
    """Single-position-substitution Metropolis-Hastings over `window_positions`.

    The edit-retention constraint is structural: proposals are only ever
    drawn from `window_positions`, so residues outside the window can never
    change. Score updates are incremental (only the changed position's
    contribution is recomputed) since the energy is position-independent by
    construction.
    """
    rng = random.Random(seed)
    sequence = list(start_sequence)
    score = combined_score("".join(sequence), window_positions, esm2_scores, pmpnn_scores, weight_esm2=weight_esm2, weight_pmpnn=weight_pmpnn)
    result = ChainResult(visited=[("".join(sequence), score)])

    for _ in range(steps):
        pos = rng.choice(window_positions)
        new_aa = rng.choice(CANONICAL_AA)
        old_aa = sequence[pos - 1]
        if new_aa == old_aa:
            result.visited.append(("".join(sequence), score))
            continue

        old_contribution = _contribution(pos, old_aa, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
        new_contribution = _contribution(pos, new_aa, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
        delta = new_contribution - old_contribution

        if delta >= 0 or rng.random() < pow(2.718281828, delta / temperature):
            sequence[pos - 1] = new_aa
            score += delta

        result.visited.append(("".join(sequence), score))

    return result


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
    steps: int = 500,
    temperature: float = 1.0,
    weight_esm2: float = 0.5,
    weight_pmpnn: float = 0.5,
    refold_every: int | None = None,
    candidate_num: int = 10,
    seed: int | None = None,
) -> list[MCMCCandidate]:
    """Run `num_starting_points * chains_per_start` MCMC chains and return the top
    `candidate_num` unique sequences by combined score.

    `esm2_scores`/`pmpnn_scores` are computed once from `folded` (two Modal
    calls total) and reused by every chain. Starting points are `folded`'s
    own sequence plus `num_starting_points - 1` diverse warm starts sampled
    via stage 1's `propose_compensatory_mutations` (reused, not
    reimplemented).

    `refold_every` doesn't gate individual MCMC steps (that would defeat the
    point of a Modal-call-free search) -- instead it's applied once, as a
    post-hoc structural safety net: the top `candidate_num * 3` pre-filter
    candidates are refolded and TM-gated via stage 1's `refold_and_gate`, and
    only ones that pass are kept. `passed_structural_check` is `None` when
    this step is skipped.
    """
    from epigen.pipeline.oracle.scoring import position_scores_esm2, position_scores_proteinmpnn

    esm2_scores = position_scores_esm2(folded.sequence)
    pmpnn_scores = position_scores_proteinmpnn(folded.structure, folded.sequence)

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

    best_score_by_sequence: dict[str, float] = {}
    for i, start_sequence in enumerate(starting_sequences):
        for j in range(chains_per_start):
            chain_seed = None if seed is None else seed + i * chains_per_start + j
            chain_result = run_mcmc_chain(
                start_sequence,
                window_positions,
                esm2_scores,
                pmpnn_scores,
                steps=steps,
                temperature=temperature,
                weight_esm2=weight_esm2,
                weight_pmpnn=weight_pmpnn,
                seed=chain_seed,
            )
            for sequence, score in chain_result.visited:
                if sequence not in best_score_by_sequence or score > best_score_by_sequence[sequence]:
                    best_score_by_sequence[sequence] = score

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
