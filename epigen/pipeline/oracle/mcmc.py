"""Metropolis-Hastings search over the edit window.

Round-synchronized across all chains: every chain proposes one substitution
per round, and all proposals are scored together in a single ESM2 call, a
single ProteinMPNN call, and (when a starting nt sequence is supplied) a
single Evo2 call -- so a `steps`-round search costs `steps * (2 or 3)` Modal
calls total, not `steps * num_chains * (2 or 3)`, while still recomputing
each expert's score fresh every round (no stale precomputed table, unlike a
position-independent design).

ESM2/ProteinMPNN score per-position AA log-probs, summed over the window
(see `window_score`). Evo2 gets the same window-restricted treatment via a
different route: `oracle.evo2_scoring.window_log_prob_batch` pulls Evo2's
own per-nucleotide log-probabilities (`Evo2ScoringConfig.return_logits=True`)
and sums just the codons at the window's AA positions, rather than averaging
over the whole nt sequence the way `nt_sequence_score_batch`'s
`avg_log_likelihood` does -- see oracle/evo2_scoring.py and oracle/codon.py
for why and how the nt sequence is carried alongside the AA sequence, and
for why this per-candidate table isn't reusable across sequences the way
ESM2/ProteinMPNN's are (Evo2 is autoregressive).

Meant to be called from inside `oracle/modal_app.py`'s Modal function, so
these per-round calls to the already-deployed esm2/proteinmpnn/evo2 services
happen container-to-container instead of laptop-to-Modal -- see that module.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from epigen.pipeline.fold_invert_refold.run import (
    CompensatoryCandidate,
    FoldedStructure,
    propose_compensatory_mutations,
    refold_and_gate,
)
from epigen.pipeline.oracle import checkpoint as checkpoint_store
from epigen.pipeline.oracle.checkpoint import ChainCheckpoint, CheckpointState
from epigen.pipeline.oracle.codon import apply_aa_substitution_to_nt
from epigen.pipeline.oracle.evo2_scoring import window_log_prob_batch
from epigen.pipeline.oracle.scoring import (
    CANONICAL_AA,
    PositionScores,
    position_scores_esm2_batch,
    position_scores_proteinmpnn_batch,
)


def window_score(
    sequence: str,
    window_positions: list[int],
    esm2_scores: PositionScores,
    pmpnn_scores: PositionScores,
    weight_esm2: float,
    weight_pmpnn: float,
) -> float:
    """Sum of the weighted per-position AA score over `window_positions`, for the AA
    actually present at each position in `sequence`.

    Public (not `_window_score`) so a caller with its own `esm2_scores`/`pmpnn_scores`
    tables -- e.g. the ones `orchestrate.py` already computes once for the oracle
    sanity checks -- can score an arbitrary sequence (WT, an MCMC chain's starting or
    ending point, ...) at zero extra Modal cost, using the ESM2+ProteinMPNN portion of
    what the search itself optimizes. Evo2's contribution isn't in this function --
    see `oracle.evo2_scoring.window_log_prob_batch` for the equivalent window-restricted
    Evo2 term -- because unlike `esm2_scores`/`pmpnn_scores`, it can't be computed once
    and reused across arbitrary sequences (Evo2 is autoregressive; its table is only
    valid for the exact sequence it was computed on), so there's no single cached table
    a caller here could reuse for free the way this function's other two arguments are.
    """
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


@dataclass(frozen=True)
class MCMCSearchResult:
    """Everything `run_mcmc_search` produces: the ranked candidates, plus each individual
    chain's starting and ending sequence -- for comparing the *distribution* of where chains
    started vs. where they ended up (e.g. against a WT/edit-only baseline), not just the
    handful of best-scoring sequences `candidates` keeps."""

    candidates: list[MCMCCandidate]  # top candidate_num unique sequences, existing behavior
    starting_sequences: list[str]  # one per chain (num_starting_points * chains_per_start), pre-round-loop
    ending_sequences: list[str]  # one per chain, same order, after all `steps` rounds
    wt_score: float  # edit-only baseline `window_score`, the per-chain freeze threshold (see run_mcmc_search)
    rounds_run: int  # rounds actually executed before hitting `steps` or every chain freezing, whichever first
    converged_chain_count: int  # chains that froze (score > wt_score) before `steps` was reached


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
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 5,
    on_checkpoint: Callable[[], None] | None = None,
) -> MCMCSearchResult:
    """Run `num_starting_points * chains_per_start` round-synchronized MCMC chains and
    return the top `candidate_num` unique sequences by combined score, plus every chain's
    individual starting/ending sequence (`MCMCSearchResult.starting_sequences`/
    `.ending_sequences`) for comparing score distributions, not just the best few.

    Starting points: `folded`'s own sequence plus `num_starting_points - 1`
    diverse warm starts sampled via stage 1's `propose_compensatory_mutations`
    (reused, not reimplemented).

    Each chain freezes (stops proposing further mutations) once its score exceeds
    `folded.sequence`'s own `window_score` -- the edit-only baseline -- since it's already
    recovered at least that much window fitness; the round loop itself exits early, before
    `steps`, once every chain has frozen. See `MCMCSearchResult.wt_score`/`.rounds_run`/
    `.converged_chain_count` for what actually happened.

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

    `checkpoint_dir`: when given, every `checkpoint_every` rounds (and once
    right after round 0) all chain state is snapshotted there via
    `oracle.checkpoint.save` -- if that directory already has a checkpoint
    from a previous call with the *same* `window_positions`/chain
    count/`use_evo2`, this call resumes from it (same chains, same RNG
    state, same running best-candidates tables) instead of starting over,
    so a crash mid-search only loses up to `checkpoint_every` rounds, not
    the whole run. A checkpoint from an incompatible config raises rather
    than silently continuing a different search. `on_checkpoint`, if given,
    is called (no args) right after every save -- e.g. a Modal Volume's
    `.commit()`, since a Volume's writes are only durable outside the
    writing container once committed. `checkpoint_dir` is untouched by
    default (`None`) -- no filesystem writes, no behavior change.
    """
    use_evo2 = nt_sequence is not None
    num_chains = num_starting_points * chains_per_start

    resumed = checkpoint_store.load(checkpoint_dir) if checkpoint_dir else None
    if resumed is not None and (
        resumed.window_positions != window_positions or resumed.num_chains != num_chains or resumed.use_evo2 != use_evo2
    ):
        raise ValueError(
            f"checkpoint at {checkpoint_dir!r} doesn't match this call's config (window_positions/"
            "chain count/use_evo2 differ) -- use a different checkpoint_dir to start a new search."
        )

    if resumed is not None:
        chains = [
            _ChainState(sequence=c.sequence, score=c.score, nt_sequence=c.nt_sequence, rng=_rng_from_state(c.rng_state))
            for c in resumed.chains
        ]
        active = [c.active for c in resumed.chains]
        starting_sequences_per_chain = resumed.starting_sequences_per_chain
        best_score_by_sequence = resumed.best_score_by_sequence
        best_nt_by_sequence = resumed.best_nt_by_sequence
        wt_score = resumed.wt_score
        start_round = resumed.round
    else:
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
        # One entry per chain (not deduped -- `chains_per_start` copies of the same warm start
        # each count separately), captured before the round loop mutates `chain.sequence`. Lets a
        # caller compare the *distribution* of where chains started vs. ended without any extra
        # Modal calls (see `run_mcmc_search`'s docstring / `MCMCSearchResult`).
        starting_sequences_per_chain = [c.sequence for c in chains]

        # Round 0: score every chain's starting sequence once, to seed `score`.
        esm2_batch = position_scores_esm2_batch([c.sequence for c in chains])
        pmpnn_batch = position_scores_proteinmpnn_batch(folded.structure, [c.sequence for c in chains])
        evo2_batch = (
            window_log_prob_batch([c.nt_sequence for c in chains], window_positions) if use_evo2 else [0.0] * len(chains)
        )
        for chain, esm2_scores, pmpnn_scores, evo2_score in zip(chains, esm2_batch, pmpnn_batch, evo2_batch, strict=True):
            chain.score = (
                window_score(chain.sequence, window_positions, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
                + weight_evo2 * evo2_score
            )

        best_score_by_sequence = {c.sequence: c.score for c in chains}
        best_nt_by_sequence = {c.sequence: c.nt_sequence for c in chains}

        # Edit-only baseline: `folded.sequence`'s own `window_score`, from the same round-0
        # batch above (it's always one of `starting_sequences`, so its score was just computed
        # alongside every chain's). Once a chain's score exceeds this, it's already recovered at
        # least as much window fitness as the un-mutated edit -- freeze it (stop proposing
        # further mutations) rather than let it keep wandering, and once *every* chain has
        # frozen, stop the round loop entirely instead of burning the remaining `steps` rounds'
        # Modal calls on nothing.
        wt_score = next(c.score for c in chains if c.sequence == folded.sequence)
        active = [c.score <= wt_score for c in chains]
        start_round = 0

        if checkpoint_dir:
            _save_checkpoint(checkpoint_dir, 0, chains, active, best_score_by_sequence, best_nt_by_sequence, starting_sequences_per_chain, wt_score, window_positions, use_evo2)
            if on_checkpoint is not None:
                on_checkpoint()

    rounds_run = start_round
    for step_idx in range(start_round, steps):
        active_indices = [i for i, is_active in enumerate(active) if is_active]
        if not active_indices:
            break
        rounds_run = step_idx + 1

        proposals = []
        nt_proposals = []
        for i in active_indices:
            chain = chains[i]
            pos = chain.rng.choice(window_positions)
            new_aa = chain.rng.choice(CANONICAL_AA)
            proposals.append(chain.sequence[: pos - 1] + new_aa + chain.sequence[pos:])
            if use_evo2:
                nt_proposals.append(apply_aa_substitution_to_nt(chain.nt_sequence, pos, new_aa))

        esm2_batch = position_scores_esm2_batch(proposals)
        pmpnn_batch = position_scores_proteinmpnn_batch(folded.structure, proposals)
        evo2_batch = window_log_prob_batch(nt_proposals, window_positions) if use_evo2 else [0.0] * len(active_indices)

        for i, proposed_seq, esm2_scores, pmpnn_scores, evo2_score, proposed_nt in zip(
            active_indices, proposals, esm2_batch, pmpnn_batch, evo2_batch, nt_proposals or [None] * len(active_indices), strict=True
        ):
            chain = chains[i]
            proposed_score = (
                window_score(proposed_seq, window_positions, esm2_scores, pmpnn_scores, weight_esm2, weight_pmpnn)
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
            if chain.score > wt_score:
                active[i] = False

        all_frozen = not any(active)
        if checkpoint_dir and (rounds_run % checkpoint_every == 0 or rounds_run == steps or all_frozen):
            _save_checkpoint(checkpoint_dir, rounds_run, chains, active, best_score_by_sequence, best_nt_by_sequence, starting_sequences_per_chain, wt_score, window_positions, use_evo2)
            if on_checkpoint is not None:
                on_checkpoint()

    ending_sequences_per_chain = [c.sequence for c in chains]
    converged_chain_count = sum(1 for is_active in active if not is_active)
    ranked = sorted(best_score_by_sequence.items(), key=lambda kv: kv[1], reverse=True)

    if refold_every is None:
        top = ranked[:candidate_num]
        return MCMCSearchResult(
            candidates=[
                MCMCCandidate(sequence=seq, combined_score=score, passed_structural_check=None, nt_sequence=best_nt_by_sequence[seq])
                for seq, score in top
            ],
            starting_sequences=starting_sequences_per_chain,
            ending_sequences=ending_sequences_per_chain,
            wt_score=wt_score,
            rounds_run=rounds_run,
            converged_chain_count=converged_chain_count,
        )

    pre_filter = ranked[: candidate_num * 3]
    pseudo_candidates = [CompensatoryCandidate(sequence=seq, perplexity=0.0, sequence_recovery=0.0) for seq, _ in pre_filter]
    refolded = refold_and_gate(pseudo_candidates, folded, seed=seed)
    validated = [r for r in refolded if r.passed_self_consistency_gate]
    validated.sort(key=lambda r: best_score_by_sequence[r.candidate.sequence], reverse=True)
    return MCMCSearchResult(
        candidates=[
            MCMCCandidate(
                sequence=r.candidate.sequence,
                combined_score=best_score_by_sequence[r.candidate.sequence],
                passed_structural_check=True,
                nt_sequence=best_nt_by_sequence[r.candidate.sequence],
            )
            for r in validated[:candidate_num]
        ],
        starting_sequences=starting_sequences_per_chain,
        ending_sequences=ending_sequences_per_chain,
        wt_score=wt_score,
        rounds_run=rounds_run,
        converged_chain_count=converged_chain_count,
    )


def _rng_from_state(rng_state: tuple) -> random.Random:
    """Rebuild a `random.Random` whose future draws continue exactly where a checkpointed
    chain's did -- `random.Random.getstate()`/`.setstate()` round-trip the full Mersenne
    Twister state, not just the seed, so this isn't an approximation."""
    rng = random.Random()
    rng.setstate(rng_state)
    return rng


def _save_checkpoint(
    checkpoint_dir: str,
    round_idx: int,
    chains: list[_ChainState],
    active: list[bool],
    best_score_by_sequence: dict[str, float],
    best_nt_by_sequence: dict[str, str | None],
    starting_sequences_per_chain: list[str],
    wt_score: float,
    window_positions: list[int],
    use_evo2: bool,
) -> None:
    state = CheckpointState(
        round=round_idx,
        window_positions=window_positions,
        num_chains=len(chains),
        use_evo2=use_evo2,
        chains=[
            ChainCheckpoint(
                sequence=chain.sequence,
                score=chain.score,
                nt_sequence=chain.nt_sequence,
                rng_state=chain.rng.getstate(),
                active=is_active,
            )
            for chain, is_active in zip(chains, active, strict=True)
        ],
        best_score_by_sequence=best_score_by_sequence,
        best_nt_by_sequence=best_nt_by_sequence,
        starting_sequences_per_chain=starting_sequences_per_chain,
        wt_score=wt_score,
    )
    checkpoint_store.save(checkpoint_dir, state)


def _reverse_translate_matching(aa_sequence: str) -> str:
    """Reverse-translate a warm-start AA sequence with the same preferred-codon table used
    for the reference nt sequence, so per-position codons stay comparable across chains."""
    from epigen.pipeline.oracle.codon import reverse_translate

    return reverse_translate(aa_sequence)
