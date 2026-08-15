"""Evidence bundle for the stage-4 explanation agent.

CLAUDE.md: the agent "receives: contact deltas, SAE feature deltas,
self-consistency TM-score, motif accessibility check" and "must ground every
claim in the numeric deltas provided." This module assembles exactly that
bundle for one compensatory candidate -- the top MCMC candidate, since that's
the only one with a real refolded structure (`orchestrate.run_end_to_end`
only refolds/contact-diffs the winner; see its docstring) -- and renders it
into the compact, numbers-first text the agent prompt is built from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from epigen.pipeline.contact_diff.accessibility import (
    MotifAccessibility,
    motif_accessibility_check,
)
from epigen.pipeline.contact_diff.diff import NeighborDelta
from epigen.pipeline.fold_invert_refold.run import FoldedStructure, RefoldedCandidate
from epigen.pipeline.literature import AnnotationRange
from epigen.pipeline.sae_diff.run import FeatureDelta, ThreeStateSAEDiff, top_k_deltas


@dataclass(frozen=True)
class CandidateEvidence:
    """Everything the explanation agent is allowed to ground claims in, for one candidate."""

    candidate_sequence: str
    edit_positions: list[int]  # 1-indexed, the fixed disruptive edit
    changed_positions: list[int]  # positions where this candidate differs from edit-only
    plddt: float
    tm_score: float  # vs edit-only, per RefoldedCandidate
    passed_self_consistency_gate: bool
    contact_deltas: list[NeighborDelta]  # edit-only vs this candidate, every changed position
    net_delta_contact_energy: float  # sum of delta_contact_energy -- negative = net favorable
    sae_top_deltas: list[FeatureDelta]  # compensated_vs_edit, top-k by |delta|
    sae_feature_descriptions: dict[int, dict[str, Any]] | None  # feature_index -> label/description, if fetched
    motif_accessibility: MotifAccessibility
    annotation_conflicts: list[AnnotationRange]  # edit/window positions overlapping known functional/structural sites


def build_candidate_evidence(
    edit_only: FoldedStructure,
    top_candidate: RefoldedCandidate,
    contact_deltas: list[NeighborDelta],
    sae_diff: ThreeStateSAEDiff,
    *,
    edit_positions: list[int],
    chain_id: str = "A",
    sae_top_k: int = 15,
    sae_feature_descriptions: dict[int, dict[str, Any]] | None = None,
) -> CandidateEvidence:
    """Assemble one candidate's evidence bundle from what `orchestrate.run_end_to_end` already computed.

    Args:
        edit_only: The pre-compensation folded structure (edit applied, no fix yet).
        top_candidate: The winning MCMC candidate, refolded + TM-gated.
        contact_deltas: `contact_diff.diff.diff_all_changed_positions(edit_only, top_candidate.folded, ...)`.
        sae_diff: `sae_diff.run` diff for this candidate's sequence (from `EndToEndResult.sae_diffs`).
        edit_positions: The fixed edit's positions -- the motif whose accessibility we're checking.
        sae_feature_descriptions: Optional human-readable labels for `sae_diff`'s top features
            (from `sae_diff.describe.describe_candidate`), if that on-demand pass was run.
    """
    changed_positions = sorted({d.edit_position for d in contact_deltas})
    net_delta_contact_energy = sum(d.delta_contact_energy for d in contact_deltas)
    motif_accessibility = motif_accessibility_check(top_candidate.folded.structure, chain_id, edit_positions)
    return CandidateEvidence(
        candidate_sequence=top_candidate.candidate.sequence,
        edit_positions=list(edit_positions),
        changed_positions=changed_positions,
        plddt=top_candidate.folded.plddt,
        tm_score=top_candidate.tm_score,
        passed_self_consistency_gate=top_candidate.passed_self_consistency_gate,
        contact_deltas=contact_deltas,
        net_delta_contact_energy=net_delta_contact_energy,
        sae_top_deltas=top_k_deltas(sae_diff.compensated_vs_edit, k=sae_top_k),
        sae_feature_descriptions=sae_feature_descriptions,
        motif_accessibility=motif_accessibility,
        annotation_conflicts=[],
    )


def with_annotation_conflicts(evidence: CandidateEvidence, annotation_conflicts: list[AnnotationRange]) -> CandidateEvidence:
    """Return a copy of `evidence` with `annotation_conflicts` attached.

    Kept separate from `build_candidate_evidence` because `EndToEndResult.annotation_conflicts`
    is computed once per run (over edit + window positions), not per candidate.
    """
    fields = {f: getattr(evidence, f) for f in evidence.__dataclass_fields__}
    fields["annotation_conflicts"] = annotation_conflicts
    return CandidateEvidence(**fields)


def _format_contact_deltas(deltas: list[NeighborDelta], *, k: int = 15) -> str:
    if not deltas:
        return "(none -- no neighbor deltas computed)"
    ranked = sorted(deltas, key=lambda d: abs(d.delta_contact_energy), reverse=True)[:k]
    lines = []
    for d in ranked:
        plddt_s = f"{d.delta_plddt:+.3f}" if d.delta_plddt is not None else "n/a"
        pae_s = f"{d.delta_pae:+.3f}" if d.delta_pae is not None else "n/a"
        lines.append(
            f"  edit_pos={d.edit_position} neighbor_pos={d.position} "
            f"({d.original_residue}->{d.candidate_residue}) "
            f"delta_distance_a={d.delta_distance_a:+.2f} "
            f"delta_contact_energy={d.delta_contact_energy:+.3f} "
            f"delta_plddt={plddt_s} delta_pae={pae_s}"
        )
    return "\n".join(lines)


def _format_sae_deltas(deltas: list[FeatureDelta], descriptions: dict[int, dict[str, Any]] | None) -> str:
    if not deltas:
        return "(none -- no SAE feature deltas above threshold)"
    lines = []
    for d in deltas:
        label = ""
        if descriptions and d.feature_index in descriptions:
            desc = descriptions[d.feature_index]
            label = f" [{desc.get('label', '')}: {desc.get('description', '')}]"
        lines.append(f"  pos={d.position} feature={d.feature_index} delta={d.delta:+.3f}{label}")
    return "\n".join(lines)


def _format_motif_accessibility(m: MotifAccessibility) -> str:
    lines = [f"  threshold: {m.threshold_a2:.1f} A^2 absolute per-residue SASA"]
    for pos in m.positions:
        lines.append(f"  pos={pos} sasa_a2={m.sasa_a2[pos]:.1f} exposed={m.exposed[pos]}")
    lines.append(f"  overall motif_accessible={m.motif_accessible}")
    return "\n".join(lines)


def format_evidence_for_prompt(evidence: CandidateEvidence) -> str:
    """Render `evidence` as the numbers-first text block the agent prompt is built around.

    Every number the agent is allowed to cite lives here; nothing else about the
    candidate is given to it (see `agent.explain_candidate`'s system prompt).
    """
    conflicts = (
        "\n".join(f"  {r.label} (residues {r.start}-{r.end}, {r.kind})" for r in evidence.annotation_conflicts)
        or "  (none)"
    )
    return f"""\
CANDIDATE SEQUENCE: {evidence.candidate_sequence}

EDIT POSITIONS (fixed, the inserted/substituted motif): {evidence.edit_positions}
POSITIONS CHANGED BY THIS COMPENSATORY CANDIDATE (vs edit-only): {evidence.changed_positions}

STRUCTURAL CONFIDENCE / SELF-CONSISTENCY
  pLDDT (refolded candidate): {evidence.plddt:.3f}
  TM-score (candidate vs edit-only structure): {evidence.tm_score:.3f}
  passed_self_consistency_gate: {evidence.passed_self_consistency_gate}

CONTACT MICROENVIRONMENT DELTAS (edit-only -> candidate, top by |delta_contact_energy|; negative delta_contact_energy = more favorable)
{_format_contact_deltas(evidence.contact_deltas)}
  net_delta_contact_energy (sum over all rows above and below the cutoff): {evidence.net_delta_contact_energy:+.3f}

SAE FEATURE DELTAS (compensated vs edit-only, top by |delta|; watch for solvent-exposed/disordered-loop features near the motif)
{_format_sae_deltas(evidence.sae_top_deltas, evidence.sae_feature_descriptions)}

MOTIF ACCESSIBILITY CHECK (must the lactocepin motif stay surface-exposed, i.e. cleavable, in the candidate structure)
{_format_motif_accessibility(evidence.motif_accessibility)}

LITERATURE ANNOTATION CONFLICTS (edit or compensatory-window positions overlapping known functional/structural sites)
{conflicts}
"""
