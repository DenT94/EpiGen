"""Stage 4: the grounded natural-language explanation agent.

CLAUDE.md: "Agent receives: contact deltas, SAE feature deltas,
self-consistency TM-score, motif accessibility check. Must ground every
claim in the numeric deltas provided -- flag internally if a claimed
compensation contradicts the numbers (e.g. claims rescue but
Delta_contact_energy is unfavorable and TM-score dropped). Output:
plain-language explanation of whether/how the compensatory mutation rescues
stability, keyed to specific evidence."

Design: Claude does the language (why the numbers mean what they mean), but
the grounding check itself is deterministic Python, not the model
self-reporting. The model is required to state four specific yes/no claims
alongside its narrative (`ClaimedGrounding`); `check_grounding` recomputes
the same four booleans straight from `CandidateEvidence` and diffs them
against what the model claimed. A model that says "rescues" while the
recomputed booleans disagree gets flagged, regardless of how convincing the
prose reads -- this is the internal contradiction check CLAUDE.md asks for,
made auditable rather than trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from epigen.config import settings
from epigen.pipeline.explain.evidence import CandidateEvidence, format_evidence_for_prompt

MODEL = "claude-opus-5"

Verdict = Literal["rescues", "partial_rescue", "does_not_rescue", "inconclusive"]

SYSTEM_PROMPT = """\
You are the explanation layer of EpiGen, a protein \
design tool. A disruptive edit (an inserted/substituted motif) \
was made to a scaffold protein, and a compensatory mutation candidate was \
generated to restore stability while keeping the motif surface-accessible \
(cleavable). You are given the complete numeric evidence for one candidate: \
contact-microenvironment deltas, SAE interpretability feature deltas, a \
self-consistency TM-score, and a motif solvent-accessibility check.

Ground every claim you make in a specific number from the evidence block -- \
name the position, the delta, or the SASA value you're pointing at. Do not \
invent evidence, do not reason from general protein-chemistry priors alone, \
and do not soften an unfavorable verdict into a favorable one. If the \
evidence is genuinely mixed or thin, say so and use verdict "inconclusive" \
rather than picking a side. Reserve "rescues" for cases where the structural \
self-consistency gate passed AND the net contact-energy trend is favorable \
AND the motif stays accessible -- a single strong compensating contact does \
not outweigh a failed TM-score gate.

You must also answer four specific yes/no questions about the evidence \
(`grounding`) -- answer them by reading the numbers directly, independent of \
what verdict you choose. These are cross-checked programmatically against \
the raw numbers after you respond, so answer them honestly even if they seem \
to undercut your narrative."""


class ClaimedGrounding(BaseModel):
    """Four numeric facts the agent must read off the evidence, checked against ground truth after the fact."""

    self_consistency_passed: bool = Field(description="Did tm_score meet the self-consistency gate (passed_self_consistency_gate)?")
    contact_energy_net_favorable: bool = Field(
        description="Is net_delta_contact_energy negative (favorable) summed across the contact deltas shown?"
    )
    motif_accessible: bool = Field(description="Does the motif accessibility check report motif_accessible=True?")
    sae_signals_disorder_risk: bool = Field(
        description="Do the top SAE feature deltas plausibly indicate increased "
        "solvent-exposed/disordered-loop character near the motif (a cleavability risk), "
        "based on their labels/positions? Answer False if no such signal is evident or no labels were provided."
    )


class CandidateExplanation(BaseModel):
    verdict: Verdict
    headline: str = Field(description="One sentence, plain language, the verdict and its main reason.")
    narrative: str = Field(description="2-4 sentences grounding the verdict in specific numbers from the evidence.")
    grounding: ClaimedGrounding
    caveats: str = Field(description="What would change this verdict, or what's uncertain -- 1-2 sentences.")


@dataclass(frozen=True)
class GroundingCheckResult:
    """Deterministic comparison of the agent's claimed grounding against the actual evidence numbers."""

    actual: ClaimedGrounding
    claimed: ClaimedGrounding
    mismatches: list[str]  # human-readable, one per disagreeing field (excludes sae_signals_disorder_risk -- judgment call)

    @property
    def contradicts_evidence(self) -> bool:
        return bool(self.mismatches)


def _recompute_grounding(evidence: CandidateEvidence) -> ClaimedGrounding:
    """The ground-truth answers to the four grounding questions, computed straight from `evidence`'s numbers."""
    return ClaimedGrounding(
        self_consistency_passed=evidence.passed_self_consistency_gate,
        contact_energy_net_favorable=evidence.net_delta_contact_energy < 0,
        motif_accessible=evidence.motif_accessibility.motif_accessible,
        # No deterministic ground truth for an interpretability read -- excluded from mismatch checks below.
        sae_signals_disorder_risk=False,
    )


def check_grounding(evidence: CandidateEvidence, explanation: CandidateExplanation) -> GroundingCheckResult:
    """Diff the agent's claimed grounding against the actual numbers, plus a verdict-level sanity check.

    Only the three numerically-checkable fields (`self_consistency_passed`,
    `contact_energy_net_favorable`, `motif_accessible`) are compared --
    `sae_signals_disorder_risk` is an interpretability judgment call with no
    single ground-truth boolean, so it's reported but never flagged as a
    mismatch. A "rescues"/"partial_rescue" verdict paired with any of the
    three failing is also flagged, even if the agent's own claimed booleans
    were (incorrectly) reported as passing -- this catches a model that
    fabricates the boolean answers along with the verdict.
    """
    actual = _recompute_grounding(evidence)
    claimed = explanation.grounding
    mismatches = []
    for field_name in ("self_consistency_passed", "contact_energy_net_favorable", "motif_accessible"):
        actual_value = getattr(actual, field_name)
        claimed_value = getattr(claimed, field_name)
        if actual_value != claimed_value:
            mismatches.append(f"claimed {field_name}={claimed_value} but the evidence shows {actual_value}")

    if explanation.verdict in ("rescues", "partial_rescue"):
        failing = [f for f in ("self_consistency_passed", "contact_energy_net_favorable", "motif_accessible") if not getattr(actual, f)]
        if failing:
            mismatches.append(
                f"verdict={explanation.verdict!r} but the evidence fails: {', '.join(failing)}"
            )

    return GroundingCheckResult(actual=actual, claimed=claimed, mismatches=mismatches)


def explain_candidate(evidence: CandidateEvidence, *, client: anthropic.Anthropic | None = None) -> tuple[CandidateExplanation, GroundingCheckResult]:
    """Generate a grounded plain-language explanation for one candidate, plus its contradiction check.

    Returns the raw explanation and the deterministic grounding check together --
    callers (e.g. the Streamlit app) decide how to surface a contradiction
    (CLAUDE.md says "flag internally", not block the run).
    """
    if client is None:
        client = anthropic.Anthropic(api_key=settings.require_anthropic_api_key())

    response = client.messages.parse(
        model=MODEL,
        # Was 2048 -- too tight combined with thinking={"type": "adaptive"} (no token cap
        # on this model) + output_config.effort="high" (deliberately thorough reasoning):
        # thinking alone could burn through most/all of a 2048 budget before the model even
        # started writing the final JSON, truncating CandidateExplanation mid-string
        # ("EOF while parsing a string" from pydantic's json_invalid -- the response hit
        # max_tokens before the JSON closed, not a malformed-output bug). Raised generously
        # since thinking effort here isn't itself capped.
        max_tokens=8192,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": format_evidence_for_prompt(evidence)}],
        output_format=CandidateExplanation,
    )
    explanation = response.parsed_output
    return explanation, check_grounding(evidence, explanation)
