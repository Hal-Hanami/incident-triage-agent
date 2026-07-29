"""Offline tests for the decide stage — the PROPOSE/ABSTAIN contract (design §6).

decide() is the safety-critical join and deliberately has no model, so its whole
decision table is testable as plain code: every ABSTAIN override (budget §8,
SEV1 §6.3, runbook-directed escalation), every §6.1 gate, their precedence
order, and the PROPOSE happy path. `run_triage` is covered end to end with
fakes sharing one Trace — no key, no network.
"""

from __future__ import annotations

from triage import decide as decide_mod
from triage import observe
from triage.classify import Classification
from triage.decide import (DEFAULT_THRESHOLDS, Thresholds, decide,
                           escalation_target_for, run_triage, triage_result)
from triage.draft import INSUFFICIENT_EVIDENCE, Draft
from triage.retrieve import Retrieval, RunbookHit
from triage.schema import (ESCALATE, Citation, Incident, IncidentType, Outcome,
                           Severity)


def make_classification(sev=Severity.SEV2, typ=IncidentType.APP_ERROR,
                        conf=0.9) -> Classification:
    return Classification(sev, typ, severity_confidence=conf, type_confidence=conf)


def make_retrieval(score=0.8, runbook="RB-app-5xx") -> Retrieval:
    hit = RunbookHit(runbook_id=runbook, score=score,
                     section_path=f"{runbook} > First response", url="x", text="...")
    return Retrieval(runbooks=[hit], chunks=[{"source_url": runbook,
                                              "section_path": hit.section_path,
                                              "url": "x", "text": "..."}])


def make_draft(action="roll_back_last_deploy", cited=True) -> Draft:
    citations = [Citation(n=1, runbook_id="RB-app-5xx",
                          section="RB-app-5xx > First response", source="x")] if cited else []
    return Draft(action_key=action, recommendation="Roll back [1].", citations=citations)


# --- the decision table (§6.1–§6.3, §8) --------------------------------------

def test_all_gates_pass_yields_propose_with_the_drafts_citations():
    d = decide(make_classification(), make_retrieval(), make_draft())
    assert d.outcome is Outcome.PROPOSE
    assert d.proposed_action == "roll_back_last_deploy"
    assert [c.n for c in d.citations] == [1]
    assert d.escalation_target is None and d.escalation_reason is None


def test_budget_exceeded_forces_abstain_before_everything_else():
    d = decide(make_classification(sev=Severity.SEV1), make_retrieval(), make_draft(),
               budget_exceeded=True)
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "cost_budget_exceeded"


def test_sev1_always_escalates_even_with_runbook_and_high_confidence():
    d = decide(make_classification(sev=Severity.SEV1), make_retrieval(), make_draft())
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "sev1_human_review"


def test_runbook_directed_escalation_wins_regardless_of_predicted_severity():
    # the INC-0003 second line of defense: SEV1 misread as SEV2, but the grounded
    # action is ESCALATE (RB-db-failover) -> still a human handoff (design §6.3 note)
    d = decide(make_classification(sev=Severity.SEV2, typ=IncidentType.DATABASE),
               make_retrieval(runbook="RB-db-failover"), make_draft(action=ESCALATE))
    assert d.outcome is Outcome.ABSTAIN
    assert d.escalation_reason == "runbook_directs_escalation"
    assert d.escalation_target == "database on-call"


def test_low_retrieval_score_abstains_as_out_of_scope():
    d = decide(make_classification(), make_retrieval(score=0.01), make_draft())
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "no_supporting_runbook"


def test_empty_retrieval_scores_zero_and_abstains():
    d = decide(make_classification(), Retrieval(runbooks=[]), make_draft())
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "no_supporting_runbook"


def test_low_classification_confidence_abstains():
    d = decide(make_classification(conf=0.2), make_retrieval(), make_draft())
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "low_confidence"


def test_drafter_abstention_abstains_with_insufficient_grounding():
    d = decide(make_classification(), make_retrieval(),
               Draft(action_key=INSUFFICIENT_EVIDENCE, recommendation="out of scope"))
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "insufficient_grounding"


def test_uncited_draft_cannot_be_proposed():
    d = decide(make_classification(), make_retrieval(), make_draft(cited=False))
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "insufficient_grounding"


def test_skipped_draft_from_a_budget_trip_abstains_on_the_budget():
    # §8: a budget trip skips the draft stage entirely -> draft=None
    d = decide(make_classification(), make_retrieval(), None, budget_exceeded=True)
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "cost_budget_exceeded"


def test_missing_draft_without_a_budget_trip_abstains_on_grounding():
    # defensive: no caller produces this today, but a missing draft must never PROPOSE
    d = decide(make_classification(), make_retrieval(), None)
    assert d.outcome is Outcome.ABSTAIN and d.escalation_reason == "insufficient_grounding"


def test_thresholds_are_configuration_not_constants():
    strict = Thresholds(min_classification_confidence=0.95, min_retrieval_score=0.9)
    d = decide(make_classification(conf=0.9), make_retrieval(score=0.8), make_draft(),
               thresholds=strict)
    assert d.outcome is Outcome.ABSTAIN  # same inputs PROPOSE under DEFAULT_THRESHOLDS
    assert decide(make_classification(conf=0.9), make_retrieval(score=0.8), make_draft(),
                  thresholds=DEFAULT_THRESHOLDS).outcome is Outcome.PROPOSE


def test_escalation_target_by_type_with_sre_default():
    assert escalation_target_for(IncidentType.SECURITY_SUSPECTED) == "security on-call"
    assert escalation_target_for(IncidentType.DATABASE) == "database on-call"
    assert escalation_target_for(IncidentType.DATA_PIPELINE) == "data on-call"
    assert escalation_target_for(IncidentType.APP_ERROR) == "on-call SRE"


# --- TriageResult assembly ----------------------------------------------------

def test_triage_result_carries_stage_outputs_and_decision():
    cls, ret = make_classification(), make_retrieval()
    decision = decide(cls, ret, make_draft())
    result = triage_result("INC-0001", cls, ret, decision)
    assert result.incident_id == "INC-0001"
    assert (result.severity, result.type) == (cls.severity, cls.type)
    assert result.outcome is Outcome.PROPOSE
    assert result.proposed_action == "roll_back_last_deploy"
    assert result.retrieved_runbooks == ["RB-app-5xx"]


# --- run_triage end to end with fakes (one shared Trace, §7/§8) ---------------

INCIDENT = Incident(
    id="INC-T1", title="Elevated 5xx after deploy", body="errors up", source="pagerduty",
    gold_severity=Severity.SEV2, gold_type=IncidentType.APP_ERROR,
    gold_action="roll_back_last_deploy", in_scope=True, expected_runbook="RB-app-5xx")


class FakeClassifier:
    model = "fake-classifier"

    def __init__(self, classification, usage=None):
        self._c = classification
        self._c.usage = usage or {}

    def classify(self, view):
        return self._c


class FakeRetriever:
    def retrieve(self, view):
        return make_retrieval()


class FakeDrafter:
    model = "fake-drafter"

    def draft(self, view, sections):
        return make_draft()


def test_run_triage_traces_all_four_stages_and_proposes():
    trace = observe.Trace()
    result = run_triage(INCIDENT, FakeClassifier(make_classification()),
                        FakeRetriever(), FakeDrafter(), trace=trace)
    assert result.outcome is Outcome.PROPOSE
    assert [name for name, _ in trace.spans] == ["classify", "retrieve", "draft", "decide"]


def test_run_triage_trips_the_cost_budget_into_an_abstain():
    # a priced model + huge usage -> running cost far above the default $0.05 budget
    expensive = FakeClassifier(make_classification(),
                               usage={"input_tokens": 40_000_000, "output_tokens": 0})
    expensive.model = "claude-opus-4-8"  # $5/M input -> $200 spent
    trace = observe.Trace()
    result = run_triage(INCIDENT, expensive, FakeRetriever(), FakeDrafter(), trace=trace)
    assert result.outcome is Outcome.ABSTAIN
    assert result.escalation_reason == "cost_budget_exceeded"


class ExplodingDrafter:
    """Must never be called — pins that a tripped budget skips the Opus draft."""

    model = "fake-drafter"

    def draft(self, view, sections):
        raise AssertionError("draft stage ran despite an exhausted budget (§8)")


def test_run_triage_never_buys_further_stages_once_over_budget():
    # §8: classify alone blows the budget -> retrieve and draft are skipped,
    # the trace shows only the stages that actually ran, and the decision is the
    # budget abstain (with no retrieved runbooks to report).
    expensive = FakeClassifier(make_classification(),
                               usage={"input_tokens": 40_000_000, "output_tokens": 0})
    expensive.model = "claude-opus-4-8"

    class ExplodingRetriever:
        def retrieve(self, view):
            raise AssertionError("retrieve stage ran despite an exhausted budget")

    trace = observe.Trace()
    result = run_triage(INCIDENT, expensive, ExplodingRetriever(), ExplodingDrafter(),
                        trace=trace)
    assert result.escalation_reason == "cost_budget_exceeded"
    assert result.retrieved_runbooks == []
    assert [name for name, _ in trace.spans] == ["classify", "decide"]


def test_run_triage_without_a_trace_cannot_see_cost_and_still_decides():
    result = run_triage(INCIDENT, FakeClassifier(make_classification()),
                        FakeRetriever(), FakeDrafter(), trace=None)
    assert result.outcome is Outcome.PROPOSE
