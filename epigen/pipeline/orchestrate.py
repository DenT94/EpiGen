"""End-to-end orchestration: WT -> edit -> oracle/MCMC search -> diffs.

Substitution-only MVP (see todo.md for insertion notes). The edit itself
(`edit_start`/`edit_sequence`, a same-length multi-residue substitution --
e.g. residues 20:27 replaced with "WHSPRAL") is a fixed constraint threaded
through the whole run -- MCMC only searches `window_positions`, which must
not overlap the edit, implementing mypipelinethoughts.md's "constraint =
retain edit sequence." This is the function the Streamlit app calls; kept
separate from the UI so it's directly testable/scriptable.
"""

from __future__ import annotations

from dataclasses import dataclass

from epigen.pipeline.alignment import identity_map
from epigen.pipeline.contact_diff.diff import NeighborDelta, diff_all_changed_positions
from epigen.pipeline.explain.evidence import (
    CandidateEvidence,
    build_candidate_evidence,
    with_annotation_conflicts,
)
from epigen.pipeline.fold_invert_refold.run import (
    CompensatoryCandidate,
    FoldedStructure,
    RefoldedCandidate,
    fold_sequence,
    refold_and_gate,
)
from epigen.pipeline.fold_invert_refold.structure_source import get_structure
from epigen.pipeline.literature import AnnotationRange, flag_positions, get_annotations
from epigen.pipeline.oracle.codon import apply_aa_substitutions_to_nt, reverse_translate
from epigen.pipeline.oracle.correlation import expert_agreement, fraction_below_wt
from epigen.pipeline.oracle.mcmc import MCMCCandidate, run_mcmc_search, window_score
from epigen.pipeline.oracle.modal_app import run_mcmc_search_on_modal
from epigen.pipeline.oracle.scoring import position_scores_esm2, position_scores_proteinmpnn
from epigen.pipeline.sae_diff.run import ThreeStateSAEDiff, diff_many_candidates


@dataclass
class EndToEndResult:
    """Everything a caller (e.g. the Streamlit app) needs to display one run."""

    original: FoldedStructure  # WT, per get_structure() (PDB if a good match exists, else ESMFold2)
    edit_only: FoldedStructure  # WT + the fixed edit, no compensation yet
    expert_correlation: float  # Pearson correlation between ESM2 and ProteinMPNN over the window
    fraction_below_wt: float  # fraction of window substitutions scoring below WT
    mcmc_candidates: list[MCMCCandidate]  # top candidate_num compensatory sequences, by combined score
    top_candidate: RefoldedCandidate | None  # the winning candidate, refolded + TM-gated
    contact_deltas: list[NeighborDelta]  # edit_only vs top_candidate, every changed position
    sae_diffs: dict[str, ThreeStateSAEDiff]  # keyed by candidate sequence, one entry per mcmc_candidates
    annotation_ranges: list[AnnotationRange]  # all known functional/structural ranges (literature.get_annotations)
    annotation_conflicts: list[AnnotationRange]  # subset overlapping window_positions (not the fixed edit itself)
    top_candidate_evidence: CandidateEvidence | None  # stage-4 input bundle for top_candidate; None if no candidates
    wt_score: float  # edit_only's own window_score -- the baseline every chain score below is measured against
    chain_starting_scores: list[float]  # one per MCMC chain, window_score at its starting sequence
    chain_ending_scores: list[float]  # one per MCMC chain, same order, window_score at its final sequence


def run_end_to_end(
    wt_sequence: str,
    *,
    edit_start: int,
    edit_sequence: str,
    window_positions: list[int],
    pdb_id: str | None = None,
    chain_id: str = "A",
    num_starting_points: int = 2,
    chains_per_start: int = 2,
    steps: int = 200,
    temperature: float = 1.0,
    candidate_num: int = 5,
    seed: int | None = 0,
    wt_nt_sequence: str | None = None,
    use_evo2: bool = True,
    weight_evo2: float = 0.34,
    annotation_ranges: list[AnnotationRange] | None = None,
    use_modal_mcmc: bool = False,
    checkpoint_every: int = 5,
) -> EndToEndResult:
    """Run the full substitution-MVP loop and return everything needed to display it.

    Args:
        wt_sequence: Wild-type sequence.
        edit_start: 1-indexed position of the first residue of the fixed
            disruptive edit.
        edit_sequence: Amino acid sequence the edit substitutes in, e.g.
            "WHSPRAL" -- replaces `len(edit_sequence)` consecutive residues
            starting at `edit_start` (residues `edit_start` through
            `edit_start + len(edit_sequence) - 1`). A single-letter string
            is a normal one-residue substitution.
        window_positions: 1-indexed positions MCMC may search for compensatory
            mutations. Must not overlap the edit's positions -- the edit is a
            fixed constraint, not something MCMC can undo.
        pdb_id: Optional known PDB ID; skips the RCSB search if given.
        num_starting_points/chains_per_start/steps/temperature/candidate_num:
            forwarded to `oracle.mcmc.run_mcmc_search`.
        seed: Shared seed threaded through every stochastic step, for
            reproducible runs.
        wt_nt_sequence: Real coding sequence for `wt_sequence`, if known.
            When omitted and `use_evo2=True`, one is generated via
            `oracle.codon.reverse_translate` (a deterministic preferred-codon
            table, not necessarily the real construct's codons -- see that
            module's docstring). The fixed edit's codon is substituted the
            same way MCMC substitutes codons for its own proposals.
        use_evo2: Whether to score candidates with Evo2 (a third,
            DNA-level expert) alongside ESM2/ProteinMPNN. Requires the
            `evo2` proto-tools app to be deployed to Modal.
        weight_evo2: Weight for Evo2's score in the combined MCMC energy,
            forwarded to `oracle.mcmc.run_mcmc_search`.
        annotation_ranges: Known functional/structural ranges for `wt_sequence`,
            in its own numbering (see `literature.get_annotations`). Defaults
            to fetching them automatically via Paperclip, keyed off `pdb_id`
            when given; pass `[]` explicitly to skip the lookup (e.g. offline).
            Advisory only -- surfaced via `EndToEndResult.annotation_conflicts`,
            never blocks the run, since literature coverage is incomplete for
            most constructs.
        use_modal_mcmc: Run the MCMC search inside `oracle.modal_app`'s
            deployed whole-loop Modal function instead of orchestrating each
            round from the laptop. Per-round esm2/proteinmpnn/evo2 calls
            become container-to-container (near-zero latency, no repeated
            client-side lookup/reconnect per round) instead of one laptop
            round trip per round -- see that module's docstring. Requires
            `modal deploy -e proto-env epigen/pipeline/oracle/modal_app.py`
            to have been run already; falls through to the laptop-orchestrated
            path's own errors if not deployed.
        checkpoint_every: Only used when `use_modal_mcmc=True` -- how often
            (in rounds) the Modal-side search checkpoints its chain state to
            a Volume, so a crash/timeout only loses up to this many rounds
            instead of the whole run. See `oracle.modal_app.run_mcmc_search_remote`
            and `oracle.mcmc.run_mcmc_search`'s `checkpoint_dir`. Retrying with
            identical arguments (including this one) resumes automatically;
            no effect on the laptop-orchestrated path, which has no
            checkpointing.
    """
    edit_positions = list(range(edit_start, edit_start + len(edit_sequence)))
    if set(edit_positions) & set(window_positions):
        raise ValueError(
            f"edit positions {edit_positions} must not overlap window_positions {window_positions} -- "
            "the edit is a fixed constraint; MCMC only searches the compensatory window."
        )

    if annotation_ranges is None:
        annotation_ranges = get_annotations(wt_sequence, pdb_id=pdb_id)
    # window_positions only, not edit_positions -- the edit is already a fixed, done decision
    # (not something being chosen among), so it overlapping an annotation isn't an actionable
    # warning the way the compensatory window (residues MCMC is actually free to mutate)
    # overlapping one is.
    annotation_conflicts = flag_positions(annotation_ranges, window_positions)

    original = get_structure(wt_sequence, pdb_id=pdb_id, chain_id=chain_id, seed=seed)

    edit_end = edit_start + len(edit_sequence) - 1
    edit_only_sequence = wt_sequence[: edit_start - 1] + edit_sequence + wt_sequence[edit_end:]
    edit_only = fold_sequence(edit_only_sequence, seed=seed)

    esm2_scores = position_scores_esm2(edit_only.sequence)
    pmpnn_scores = position_scores_proteinmpnn(edit_only.structure, edit_only.sequence)
    correlation = expert_agreement(esm2_scores, pmpnn_scores, edit_only.sequence, window_positions)
    below_wt = fraction_below_wt(esm2_scores, pmpnn_scores, edit_only.sequence, window_positions)

    edit_only_nt_sequence = None
    if use_evo2:
        wt_nt = wt_nt_sequence or reverse_translate(wt_sequence)
        edit_only_nt_sequence = apply_aa_substitutions_to_nt(wt_nt, edit_start, edit_sequence)

    if use_modal_mcmc:
        raw_result = run_mcmc_search_on_modal(
            edit_only,
            window_positions,
            chain_id=chain_id,
            num_starting_points=num_starting_points,
            chains_per_start=chains_per_start,
            steps=steps,
            temperature=temperature,
            candidate_num=candidate_num,
            seed=seed,
            nt_sequence=edit_only_nt_sequence,
            weight_evo2=weight_evo2,
            checkpoint_every=checkpoint_every,
        )
        mcmc_candidates = [
            MCMCCandidate(
                sequence=c["sequence"],
                combined_score=c["combined_score"],
                passed_structural_check=c["passed_structural_check"],
                nt_sequence=c["nt_sequence"],
            )
            for c in raw_result["candidates"]
        ]
        chain_starting_sequences = raw_result["starting_sequences"]
        chain_ending_sequences = raw_result["ending_sequences"]
    else:
        mcmc_result = run_mcmc_search(
            edit_only,
            window_positions,
            chain_id=chain_id,
            num_starting_points=num_starting_points,
            chains_per_start=chains_per_start,
            steps=steps,
            temperature=temperature,
            candidate_num=candidate_num,
            seed=seed,
            nt_sequence=edit_only_nt_sequence,
            weight_evo2=weight_evo2,
        )
        mcmc_candidates = mcmc_result.candidates
        chain_starting_sequences = mcmc_result.starting_sequences
        chain_ending_sequences = mcmc_result.ending_sequences

    # The fixed edit must survive every candidate and every chain's start/end untouched --
    # `run_mcmc_search`'s round loop only ever proposes substitutions at `window_positions`,
    # and `propose_compensatory_mutations`'s warm starts fix everything outside
    # `window_positions` (which includes edit_positions, since orchestrate.py already
    # rejects the two ranges overlapping above). Both guarantee this by construction; this
    # is a cheap, local (no Modal call) assertion that it actually held, not just an assumed
    # invariant -- fail loudly rather than silently ship a candidate that lost the edit.
    for label, sequences in (
        ("candidate", (c.sequence for c in mcmc_candidates)),
        ("chain start", chain_starting_sequences),
        ("chain end", chain_ending_sequences),
    ):
        for seq in sequences:
            if any(seq[p - 1] != edit_only.sequence[p - 1] for p in edit_positions):
                raise RuntimeError(
                    f"Fixed edit was not preserved in a {label} sequence -- expected "
                    f"{edit_sequence!r} at positions {edit_positions}, got {seq!r}. This "
                    "should be structurally impossible; see orchestrate.py's comment here."
                )

    # Free (no extra Modal call): score WT/chain starts/chain ends with the same
    # ESM2+ProteinMPNN per-position tables already spent above on the oracle sanity
    # checks. 0.5/0.5 matches run_mcmc_search's own (unexposed-here) default weights,
    # so this is directly comparable to what the search itself optimized -- Evo2's
    # whole-sequence term isn't included, since it isn't a per-position table (see
    # oracle.mcmc.window_score's docstring).
    wt_score = window_score(edit_only.sequence, window_positions, esm2_scores, pmpnn_scores, 0.5, 0.5)
    chain_starting_scores = [
        window_score(seq, window_positions, esm2_scores, pmpnn_scores, 0.5, 0.5) for seq in chain_starting_sequences
    ]
    chain_ending_scores = [
        window_score(seq, window_positions, esm2_scores, pmpnn_scores, 0.5, 0.5) for seq in chain_ending_sequences
    ]

    top_candidate: RefoldedCandidate | None = None
    contact_deltas: list[NeighborDelta] = []
    sae_diffs: dict[str, ThreeStateSAEDiff] = {}
    top_candidate_evidence: CandidateEvidence | None = None
    if mcmc_candidates:
        pseudo = [CompensatoryCandidate(sequence=mcmc_candidates[0].sequence, perplexity=0.0, sequence_recovery=0.0)]
        top_candidate = refold_and_gate(pseudo, edit_only, seed=seed)[0]
        contact_deltas = diff_all_changed_positions(
            edit_only, top_candidate.folded, chain_id=chain_id, position_map=identity_map()
        )
        # Broad pass over every candidate (not just the winner), batched into 3 Modal
        # calls total regardless of candidate count -- see diff_many_candidates' docstring.
        # Needed for the cross-candidate PCA scatter, not just the single top_candidate diff.
        candidate_diffs = diff_many_candidates(
            wt_sequence,
            edit_only_sequence,
            [c.sequence for c in mcmc_candidates],
            position_map=identity_map(),
        )
        sae_diffs = {c.sequence: diff for c, diff in zip(mcmc_candidates, candidate_diffs, strict=True)}

        # Stage 4's input bundle -- cheap, deterministic assembly of what's already
        # computed above (no LLM call here; that's explain.agent.explain_candidate,
        # triggered on demand since it costs real Anthropic API time/money).
        top_sae_diff = sae_diffs[top_candidate.candidate.sequence]
        evidence = build_candidate_evidence(
            edit_only,
            top_candidate,
            contact_deltas,
            top_sae_diff,
            edit_positions=edit_positions,
            chain_id=chain_id,
        )
        top_candidate_evidence = with_annotation_conflicts(evidence, annotation_conflicts)

    return EndToEndResult(
        original=original,
        edit_only=edit_only,
        expert_correlation=correlation,
        fraction_below_wt=below_wt,
        mcmc_candidates=mcmc_candidates,
        top_candidate=top_candidate,
        contact_deltas=contact_deltas,
        sae_diffs=sae_diffs,
        annotation_ranges=annotation_ranges,
        annotation_conflicts=annotation_conflicts,
        top_candidate_evidence=top_candidate_evidence,
        wt_score=wt_score,
        chain_starting_scores=chain_starting_scores,
        chain_ending_scores=chain_ending_scores,
    )
