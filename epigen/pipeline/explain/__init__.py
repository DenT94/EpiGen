"""Stage 4: Agent explanation layer.

Receives contact deltas, SAE feature deltas, self-consistency TM-score, and
motif accessibility check. Must ground every claim in the numeric deltas
provided -- flag internally if a claimed compensation contradicts the
numbers (e.g. claims rescue but delta_contact_energy is unfavorable and
TM-score dropped). Output: plain-language explanation of whether/how the
compensatory mutation rescues stability, keyed to specific evidence.

`evidence.py` assembles the numeric evidence bundle from what
`orchestrate.run_end_to_end` already computed (stage 2/3 outputs + a motif
accessibility check); `agent.py` is the Claude call plus a deterministic
grounding cross-check -- see that module's docstring for the design.
"""

from __future__ import annotations

from epigen.pipeline.explain.agent import (
    CandidateExplanation,
    ClaimedGrounding,
    GroundingCheckResult,
    check_grounding,
    explain_candidate,
)
from epigen.pipeline.explain.evidence import (
    CandidateEvidence,
    build_candidate_evidence,
    format_evidence_for_prompt,
    with_annotation_conflicts,
)

__all__ = [
    "CandidateEvidence",
    "CandidateExplanation",
    "ClaimedGrounding",
    "GroundingCheckResult",
    "build_candidate_evidence",
    "check_grounding",
    "explain_candidate",
    "format_evidence_for_prompt",
    "with_annotation_conflicts",
]
