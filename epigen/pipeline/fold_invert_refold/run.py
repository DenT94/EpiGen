"""proto_tools-backed implementation of the fold -> invert -> refold loop.

All model calls run on Modal (`device="modal"`, `proto-env` environment; see
configs/modal.md) via proto-tools. Two confidence gates guard the loop per
CLAUDE.md: a pLDDT gate before trusting a fold enough to invert on, and a
TM-score self-consistency gate on refolded candidates before they're passed
to the contact/SAE diff stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from proto_tools import (
    ESMFold2Config,
    ESMFold2Input,
    InverseFoldingStructureInput,
    ProteinMPNNSampleConfig,
    ProteinMPNNSampleInput,
    TMalignConfig,
    TMalignInput,
    run_esmfold2,
    run_proteinmpnn_sample,
    run_tmalign,
)
from proto_tools.entities.structures import ResidueSelection, Structure

logger = logging.getLogger(__name__)

DEVICE = "modal"
PLDDT_GATE = 0.7
TM_SCORE_GATE = 0.5  # TMalign's own documented same-fold threshold.


@dataclass
class FoldedStructure:
    """One structure for `sequence`, plus its confidence-gate verdict.

    `source`/`pdb_id` record where the structure came from -- ESMFold2 or a
    real PDB entry via `structure_source.get_structure()` -- since a real
    structure is strictly more trustworthy than a prediction and callers may
    want to know which they got.
    """

    sequence: str
    structure: Structure
    plddt: float
    avg_pae: float
    passed_confidence_gate: bool
    source: str = "esmfold2"  # "esmfold2" | "pdb"
    pdb_id: str | None = None


def fold_sequence(
    sequence: str,
    *,
    include_pae_matrix: bool = False,
    seed: int | None = None,
) -> FoldedStructure:
    """Fold one sequence with ESMFold2 (Modal) and apply the pLDDT confidence gate."""
    config = ESMFold2Config(device=DEVICE, include_pae_matrix=include_pae_matrix, seed=seed)
    output = run_esmfold2(ESMFold2Input(complexes=[sequence]), config)
    structure = output.structures[0]
    plddt = structure.metrics.plddt
    return FoldedStructure(
        sequence=sequence,
        structure=structure,
        plddt=plddt,
        avg_pae=structure.metrics.avg_pae,
        passed_confidence_gate=plddt >= PLDDT_GATE,
    )


@dataclass
class CompensatoryCandidate:
    """One ProteinMPNN-proposed sequence, restricted to the edit window."""

    sequence: str
    perplexity: float
    sequence_recovery: float


def propose_compensatory_mutations(
    folded: FoldedStructure,
    *,
    window_positions: list[int],
    chain_id: str = "A",
    num_sequences: int = 8,
    temperature: float = 0.2,
    seed: int | None = None,
) -> list[CompensatoryCandidate]:
    """Sample ProteinMPNN sequences that vary only within `window_positions`.

    Every residue outside the window is held fixed, so proposals are
    compensatory mutations around the inserted motif rather than a full
    redesign of the scaffold.

    Raises:
        ValueError: If `folded` failed the pLDDT confidence gate -- inverting
            on a low-confidence structure isn't trustworthy (CLAUDE.md's
            explicit gating requirement).
    """
    if not folded.passed_confidence_gate:
        raise ValueError(
            f"Refusing to invert-fold a low-confidence structure "
            f"(pLDDT={folded.plddt:.3f} < {PLDDT_GATE})."
        )

    all_positions = set(folded.structure.get_chain_positions(chain_id))
    fixed = sorted(all_positions - set(window_positions))

    inp = InverseFoldingStructureInput(
        structure=folded.structure,
        chains_to_redesign=[chain_id],
        fixed_positions=ResidueSelection(chains={chain_id: fixed}) if fixed else None,
    )
    config = ProteinMPNNSampleConfig(
        num_sequences_per_structure=num_sequences,
        temperature=temperature,
        seed=seed,
        device=DEVICE,
    )
    output = run_proteinmpnn_sample(ProteinMPNNSampleInput(inputs=[inp]), config)
    design_set = output.design_sets[0]
    return [
        CompensatoryCandidate(
            sequence=design.chains[0].sequence,  # single-chain monomer: chain 0 is the redesign.
            perplexity=design.metrics.perplexity,
            sequence_recovery=design.metrics.sequence_recovery,
        )
        for design in design_set.complexes
    ]


@dataclass
class RefoldedCandidate:
    """A candidate after refolding, carrying its self-consistency verdict."""

    candidate: CompensatoryCandidate
    folded: FoldedStructure
    tm_score: float
    passed_self_consistency_gate: bool


def refold_and_gate(
    candidates: list[CompensatoryCandidate],
    reference: FoldedStructure,
    *,
    seed: int | None = None,
) -> list[RefoldedCandidate]:
    """Refold each candidate and gate on TM-score self-consistency vs `reference`.

    A candidate that fails the pLDDT confidence gate on refold is scored
    tm_score=0.0 and fails the self-consistency gate automatically, rather
    than running TMalign against an untrustworthy structure.
    """
    results = []
    for i, candidate in enumerate(candidates):
        refolded = fold_sequence(candidate.sequence, seed=None if seed is None else seed + i)
        tm_score = 0.0
        if refolded.passed_confidence_gate:
            tm_output = run_tmalign(
                TMalignInput(query_structure=refolded.structure, reference_structure=reference.structure),
                TMalignConfig(),
            )
            # Normalize by the reference (target scaffold) length, per TMalign's own
            # guidance for ranking candidates against a fixed structure.
            tm_score = tm_output.metrics.tm_score_chain_2
        results.append(
            RefoldedCandidate(
                candidate=candidate,
                folded=refolded,
                tm_score=tm_score,
                passed_self_consistency_gate=tm_score >= TM_SCORE_GATE,
            )
        )
    return results
