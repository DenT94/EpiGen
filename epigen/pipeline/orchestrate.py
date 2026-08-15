"""End-to-end orchestration: WT -> edit -> oracle/MCMC search -> diffs.

Substitution-only MVP (see todo.md for insertion notes). The edit itself
(`edit_position`/`edit_residue`) is a fixed constraint threaded through the
whole run -- MCMC only searches `window_positions`, which must exclude
`edit_position`, implementing mypipelinethoughts.md's "constraint = retain
edit sequence." This is the function the Streamlit app calls; kept separate
from the UI so it's directly testable/scriptable.
"""

from __future__ import annotations

from dataclasses import dataclass

from epigen.pipeline.alignment import identity_map
from epigen.pipeline.contact_diff.diff import NeighborDelta, diff_all_changed_positions
from epigen.pipeline.fold_invert_refold.run import (
    CompensatoryCandidate,
    FoldedStructure,
    RefoldedCandidate,
    fold_sequence,
    refold_and_gate,
)
from epigen.pipeline.fold_invert_refold.structure_source import get_structure
from epigen.pipeline.literature import AnnotationRange, flag_positions, get_annotations
from epigen.pipeline.oracle.codon import apply_aa_substitution_to_nt, reverse_translate
from epigen.pipeline.oracle.correlation import expert_agreement, fraction_below_wt
from epigen.pipeline.oracle.mcmc import MCMCCandidate, run_mcmc_search
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
    annotation_conflicts: list[AnnotationRange]  # subset overlapping edit_position or window_positions


def run_end_to_end(
    wt_sequence: str,
    *,
    edit_position: int,
    edit_residue: str,
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
) -> EndToEndResult:
    """Run the full substitution-MVP loop and return everything needed to display it.

    Args:
        wt_sequence: Wild-type sequence.
        edit_position: 1-indexed position of the fixed disruptive edit.
        edit_residue: One-letter amino acid the edit substitutes in.
        window_positions: 1-indexed positions MCMC may search for compensatory
            mutations. Must not include `edit_position` -- the edit is a
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
    """
    if edit_position in window_positions:
        raise ValueError(
            f"edit_position {edit_position} must not be in window_positions {window_positions} -- "
            "the edit is a fixed constraint; MCMC only searches the compensatory window."
        )

    if annotation_ranges is None:
        annotation_ranges = get_annotations(wt_sequence, pdb_id=pdb_id)
    annotation_conflicts = flag_positions(annotation_ranges, [edit_position, *window_positions])

    original = get_structure(wt_sequence, pdb_id=pdb_id, chain_id=chain_id, seed=seed)

    edit_only_sequence = wt_sequence[: edit_position - 1] + edit_residue + wt_sequence[edit_position:]
    edit_only = fold_sequence(edit_only_sequence, seed=seed)

    esm2_scores = position_scores_esm2(edit_only.sequence)
    pmpnn_scores = position_scores_proteinmpnn(edit_only.structure, edit_only.sequence)
    correlation = expert_agreement(esm2_scores, pmpnn_scores, edit_only.sequence, window_positions)
    below_wt = fraction_below_wt(esm2_scores, pmpnn_scores, edit_only.sequence, window_positions)

    edit_only_nt_sequence = None
    if use_evo2:
        wt_nt = wt_nt_sequence or reverse_translate(wt_sequence)
        edit_only_nt_sequence = apply_aa_substitution_to_nt(wt_nt, edit_position, edit_residue)

    mcmc_candidates = run_mcmc_search(
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

    top_candidate: RefoldedCandidate | None = None
    contact_deltas: list[NeighborDelta] = []
    sae_diffs: dict[str, ThreeStateSAEDiff] = {}
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
    )
