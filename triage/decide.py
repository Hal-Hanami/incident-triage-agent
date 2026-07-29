"""Stage: decide — the deterministic PROPOSE / ABSTAIN contract (design §6).

**No model.** This is the safety-critical join, so it is a plain, fully-testable
rule over the stage outputs (classification confidence, retrieval score, the
drafter's grounded action) plus the running cost — never an LLM. The agent emits
exactly one of two outcomes (`schema.Outcome`): PROPOSE a cited recommendation, or
ABSTAIN and escalate to a human. There is no "execute."

PROPOSE is allowed only when *all* of design §6.1 hold: classification confidence
≥ threshold, retrieval surfaced a runbook above the score threshold, and the draft
produced a citation-backed action. Otherwise the agent ABSTAINs and escalates — the
default-safe outcome, a first-class success (§6.2). Two overrides make ABSTAIN win
regardless:

  - **§6.3 the SEV1 rule** — anything classified SEV1 goes to a human even with a
    runbook in hand (high-blast-radius incidents are a human's call by policy).
  - **a runbook that itself directs escalation** — when the grounded action is
    `ESCALATE` (e.g. RB-db-failover), the runbook content escalates, *independent of
    the predicted severity*. This is the second line of defense for the INC-0003
    case where classify misread a SEV1 primary-DB outage as SEV2 (docs/EVALUATION.md).

The cost budget (§8) trips an ABSTAIN too, and it is enforced *between*
stages: `run_triage` checks the running cost after each paid stage, and once the
budget is crossed no further paid stage runs — in particular the Opus draft is
never bought — so `decide()` accepts `draft=None` for that path. Thresholds are
configuration (`Thresholds`), tuned on the eval set against the false-abstention /
missed-escalation trade-off (§6, §10) — not hardcoded magic numbers. The primary
abstention driver is still the drafter's grounding judgment (§5); these numeric
gates are the secondary safety net.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import observe
from .classify import Classification, classify_incident
from .draft import Draft, draft_response
from .retrieve import Retrieval, retrieve_runbooks
from .schema import (Citation, ESCALATE, Incident, IncidentType, Outcome,
                     Severity, TriageResult)


@dataclass(frozen=True)
class Thresholds:
    """Decision gates (design §6.1). Configuration, tuned on the eval set — not
    hardcoded magic numbers (§6, §10). Defaults are deliberately permissive: the
    drafter's grounding (§5) is the primary abstention mechanism, and these gates
    are the secondary safety net, so they should not over-fire and cause false
    abstentions on answerable incidents. See docs/EVALUATION.md for the measurements."""

    min_classification_confidence: float = 0.35
    min_retrieval_score: float = 0.10


DEFAULT_THRESHOLDS = Thresholds()

# Escalation target by predicted incident type (design §6.2) — who the human handoff
# names. Matches the on-call rotations the fixtures' notes reference.
_TARGETS: dict[IncidentType, str] = {
    IncidentType.SECURITY_SUSPECTED: "security on-call",
    IncidentType.DATA_PIPELINE: "data on-call",
    IncidentType.DATABASE: "database on-call",
}


def escalation_target_for(incident_type: IncidentType) -> str:
    """The on-call rotation an abstention escalates to, by predicted type (design §6.2)."""
    return _TARGETS.get(incident_type, "on-call SRE")


@dataclass
class Decision:
    """The decider's verdict: a PROPOSE (cited action) or an ABSTAIN (escalation with
    a named target + machine-readable reason). Exactly the fields the TriageResult
    needs from §6 — assembled with classification/retrieval context by `triage_result`."""

    outcome: Outcome
    proposed_action: str | None = None
    citations: list[Citation] = field(default_factory=list)
    escalation_target: str | None = None
    escalation_reason: str | None = None


def decide(classification: Classification, retrieval: Retrieval, draft: Draft | None,
           *, thresholds: Thresholds = DEFAULT_THRESHOLDS,
           budget_exceeded: bool = False) -> Decision:
    """Turn the stage outputs + running cost into the final Outcome (design §6).

    Order is safety-first: budget (§8) and the SEV1 rule (§6.3) and a runbook-directed
    escalation all force ABSTAIN before any PROPOSE path; then the §6.1 gates
    (retrieval score, classification confidence, a citation-backed draft). The chosen
    `escalation_reason` is the first failing condition — a human-readable why for the
    handoff; the *outcome* is the same ABSTAIN regardless of which fired.

    `draft=None` means the stage never ran — a budget trip skips it (§8);
    a run that somehow skipped it without the trip has nothing citation-backed
    to propose and abstains on grounding."""
    target = escalation_target_for(classification.type)

    def abstain(reason: str) -> Decision:
        return Decision(Outcome.ABSTAIN, escalation_target=target, escalation_reason=reason)

    # §8: a triage that would overspend degrades to an escalation rather than running away.
    if budget_exceeded:
        return abstain("cost_budget_exceeded")
    # §6.3: SEV1 is always a human's call, even with a runbook and high confidence.
    if classification.severity is Severity.SEV1:
        return abstain("sev1_human_review")
    # A runbook that itself directs escalation (RB-db-failover) → escalate regardless of
    # the predicted severity. Catches a SEV1 that was misclassified lower (INC-0003).
    if draft is not None and draft.grounded and draft.action_key == ESCALATE:
        return abstain("runbook_directs_escalation")
    # §6.2: retrieval surfaced no supporting runbook above the score threshold (out of scope).
    if retrieval.top_score < thresholds.min_retrieval_score:
        return abstain("no_supporting_runbook")
    # §6.2: classification too uncertain to commit to an action.
    if classification.min_confidence < thresholds.min_classification_confidence:
        return abstain("low_confidence")
    # §5/§6.1: the drafter didn't ground a citation-backed action (or abstained outright).
    if draft is None or not draft.is_citation_backed:
        return abstain("insufficient_grounding")
    # All gates pass → a cited recommendation for a human to apply (§6.1).
    return Decision(Outcome.PROPOSE, proposed_action=draft.action_key,
                    citations=list(draft.citations))


def triage_result(incident_id: str, classification: Classification,
                  retrieval: Retrieval, decision: Decision) -> TriageResult:
    """Assemble the final `TriageResult` from the stage outputs + the decision (§6)."""
    return TriageResult(
        incident_id=incident_id,
        severity=classification.severity,
        severity_confidence=classification.severity_confidence,
        type=classification.type,
        type_confidence=classification.type_confidence,
        outcome=decision.outcome,
        proposed_action=decision.proposed_action,
        citations=decision.citations,
        escalation_reason=decision.escalation_reason,
        escalation_target=decision.escalation_target,
        retrieved_runbooks=retrieval.runbook_ids,
    )


def run_triage(incident: Incident, classifier, retriever, drafter,
               *, thresholds: Thresholds = DEFAULT_THRESHOLDS,
               budget: float = observe.DEFAULT_INCIDENT_BUDGET_USD,
               trace: observe.Trace | None = None) -> TriageResult:
    """Run the whole measurable pipeline for one incident → a TriageResult (design §2).

    classify → retrieve → draft → decide, sharing one Trace so the per-incident
    observability footer (§7) and the cost budget (§8) see every stage. The budget
    is enforced *between* stages: the running cost is checked after each paid
    stage, and once it crosses the budget no further paid stage runs — in
    particular the Opus draft is never bought — and the decider turns the trip
    into ABSTAIN(`cost_budget_exceeded`). Without a Trace there is no running
    cost to check, so nothing trips. This is the single-incident surface behind
    `python -m triage triage <INC-ID>`; the eval harness runs the same stages in
    its own loop so it can score each one (design §10)."""
    def over_budget() -> bool:
        return trace is not None and trace.cost() > budget

    view = incident.prompt_view()
    cls = classify_incident(view, classifier, trace=trace)
    exceeded = over_budget()
    ret = Retrieval(runbooks=[]) if exceeded else retrieve_runbooks(view, retriever, trace=trace)
    exceeded = exceeded or over_budget()
    drf = None if exceeded else draft_response(view, ret.chunks, drafter, trace=trace)
    exceeded = exceeded or over_budget()
    with observe.span(trace, "decide"):
        decision = decide(cls, ret, drf, thresholds=thresholds, budget_exceeded=exceeded)
    return triage_result(incident.id, cls, ret, decision)
