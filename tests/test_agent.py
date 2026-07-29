"""Offline tests for the Agent SDK shell — guardrails + red-team (design §2, §9).

The read-only guarantee is structural, so it is testable as plain code without
the SDK, a key, or the network: `guardrail_spec()` is the exact policy the live
`build_options` consumes, `is_tool_allowed` is the single predicate both the
`can_use_tool` callback and the `PreToolUse` hook guard with, and the
`AgentSession` tool bodies are sync methods driven directly with fakes. The
red-team suite pins that no destructive tool is ever allowed — for the
destructive-invite fixture (`fixtures/incidents/redteam.jsonl`) and for every
mutating built-in by name — and that the agent path yields the *same*
TriageResult as `run_triage` for identical stage outputs (design §2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from triage import agent as agent_mod
from triage import observe, schema
from triage.agent import (ALLOWED_TOOLS, MUTATING_TOOLS, AgentSession,
                          deny_reason, guardrail_spec, is_tool_allowed)
from triage.classify import Classification
from triage.decide import run_triage
from triage.draft import INSUFFICIENT_EVIDENCE, Draft
from triage.retrieve import Retrieval, build_retrieval
from triage.schema import (ESCALATE, Incident, IncidentType, Outcome, Severity,
                           load_incidents, validate)

ROOT = Path(__file__).resolve().parent.parent
REDTEAM_FILE = ROOT / "fixtures" / "incidents" / "redteam.jsonl"


# --- fakes (the Protocol seams, as in test_decide/test_eval) ------------------

class FakeClassifier:
    def __init__(self, sev=Severity.SEV2, typ=IncidentType.APP_ERROR, conf=0.9,
                 usage=None):
        self._c = Classification(sev, typ, severity_confidence=conf,
                                 type_confidence=conf, usage=usage or {})

    def classify(self, view):
        return self._c


class FakeRetriever:
    def __init__(self, runbook="RB-app-5xx", score=0.8):
        self._chunks = [{"source_url": runbook, "score": score,
                         "section_path": f"{runbook} > First response",
                         "url": "x", "text": "roll back the last deploy"}]

    def retrieve(self, view):
        return build_retrieval(self._chunks)


class FakeDrafter:
    def __init__(self, action="roll_back_last_deploy", recommendation="Roll back [1]."):
        self._action = action
        self._rec = recommendation

    def draft(self, view, sections):
        from triage.draft import extract_citations
        citations = ([] if self._action == INSUFFICIENT_EVIDENCE
                     else extract_citations(self._rec, sections))
        return Draft(action_key=self._action, recommendation=self._rec,
                     citations=citations)


def make_incident(id="INC-T001", sev=Severity.SEV2) -> Incident:
    return Incident(id=id, title="t", body="b", source="pagerduty",
                    gold_severity=sev, gold_type=IncidentType.APP_ERROR,
                    gold_action="roll_back_last_deploy", in_scope=True,
                    expected_runbook="RB-app-5xx")


def make_session(**kw) -> AgentSession:
    defaults = dict(incident=make_incident(), classifier=FakeClassifier(),
                    retriever=FakeRetriever(), drafter=FakeDrafter(),
                    notify=lambda msg: None)
    defaults.update(kw)
    return AgentSession(**defaults)


# --- the §9 allowlist predicate ------------------------------------------------

def test_allowlist_is_exactly_the_four_triage_mcp_tools():
    assert ALLOWED_TOOLS == (
        "mcp__triage__classify_incident",
        "mcp__triage__search_runbooks",
        "mcp__triage__draft_response",
        "mcp__triage__escalate",
    )
    for name in ALLOWED_TOOLS:
        assert is_tool_allowed(name)


@pytest.mark.parametrize("name", MUTATING_TOOLS)
def test_every_mutating_builtin_is_denied_by_name(name):
    assert not is_tool_allowed(name)
    assert name in deny_reason(name)


@pytest.mark.parametrize("name", [
    "",                                   # empty
    "bash",                               # case matters — no fuzzy matching
    "Read", "Glob", "Grep",               # even read-only built-ins are off-list
    "mcp__triage__restart_database",      # right server, wrong tool
    "mcp__prod__classify_incident",       # right tool name, wrong server
    "mcp__triage__classify_incident2",    # prefix near-miss
    "mcp__triage__",                      # server prefix alone
])
def test_near_miss_and_foreign_tools_are_denied(name):
    assert not is_tool_allowed(name)


def test_deny_reason_names_the_tool_and_the_allowlist():
    msg = deny_reason("Bash")
    assert "'Bash'" in msg
    for allowed in ALLOWED_TOOLS:
        assert allowed in msg


# --- the §9 policy the live options are built from ------------------------------

def test_guardrail_spec_registers_no_builtin_tools_at_all():
    assert guardrail_spec()["tools"] == []  # a tool that doesn't exist can't be called


def test_guardrail_spec_allowlists_only_triage_mcp_tools():
    spec = guardrail_spec()
    assert spec["allowed_tools"] == list(ALLOWED_TOOLS)
    assert all(t.startswith("mcp__triage__") for t in spec["allowed_tools"])
    # no mutating tool can ever be simultaneously allowed and denied
    assert not set(spec["allowed_tools"]) & set(spec["disallowed_tools"])


def test_guardrail_spec_never_escalates_privileges():
    spec = guardrail_spec()
    assert spec["permission_mode"] == "default"   # never bypassPermissions
    assert spec["setting_sources"] == []          # no looser settings inherited
    for name in ("Bash", "Write", "Edit"):
        assert name in spec["disallowed_tools"]


# --- the MCP tool bodies + the deterministic join (design §2) -------------------

def test_agent_session_yields_the_same_result_as_run_triage():
    inc = make_incident()
    fakes = dict(classifier=FakeClassifier(), retriever=FakeRetriever(),
                 drafter=FakeDrafter())
    pipeline = run_triage(inc, fakes["classifier"], fakes["retriever"],
                          fakes["drafter"], trace=observe.Trace())
    session = make_session(incident=inc, **fakes)
    session.tool_classify()
    session.tool_search()
    session.tool_draft()
    assert session.finish() == pipeline  # same stages, same decide(), same answer


def test_tool_summaries_carry_stage_verdicts_not_raw_state():
    session = make_session()
    assert "severity=SEV2" in session.tool_classify()
    assert "RB-app-5xx" in session.tool_search()
    out = session.tool_draft()
    assert "roll_back_last_deploy" in out and "[1]" in out


def test_draft_tool_requires_retrieval_first():
    session = make_session()
    assert session.tool_draft().startswith("error:")
    assert session.draft is None


def test_incomplete_run_degrades_to_safe_abstain_and_still_notifies():
    notes: list[str] = []
    session = make_session(notify=notes.append)
    session.tool_classify()  # orchestrator stopped after one stage
    result = session.finish()
    assert result.outcome is Outcome.ABSTAIN
    assert result.escalation_reason == "agent_incomplete_run"
    assert notes and "escalation" in notes[0]


def test_incomplete_run_without_any_stage_assumes_the_worst():
    result = make_session().finish()
    assert result.outcome is Outcome.ABSTAIN
    assert result.severity is Severity.SEV1 and result.severity_confidence == 0.0


def test_escalate_tool_notifies_once_and_finish_does_not_duplicate():
    notes: list[str] = []
    session = make_session(drafter=FakeDrafter(action=INSUFFICIENT_EVIDENCE,
                                               recommendation="no section fits"),
                           notify=notes.append)
    session.tool_classify()
    session.tool_search()
    session.tool_draft()
    session.tool_escalate("drafter abstained")
    result = session.finish()
    assert result.outcome is Outcome.ABSTAIN
    assert len(notes) == 1 and "drafter abstained" in notes[0]


def test_abstain_without_model_escalation_posts_the_handoff_itself():
    notes: list[str] = []
    session = make_session(drafter=FakeDrafter(action=INSUFFICIENT_EVIDENCE,
                                               recommendation="no section fits"),
                           notify=notes.append)
    session.tool_classify()
    session.tool_search()
    session.tool_draft()
    result = session.finish()  # model forgot to call escalate — shell's obligation
    assert result.outcome is Outcome.ABSTAIN
    assert len(notes) == 1 and "insufficient_grounding" in notes[0]


def test_budget_trip_flows_through_the_agent_join():
    session = make_session(budget=0.0000001)
    session.trace.add_usage("claude-opus-4-8", {"input_tokens": 1_000_000})
    session.tool_classify()
    session.tool_search()
    session.tool_draft()
    result = session.finish()
    assert result.outcome is Outcome.ABSTAIN
    assert result.escalation_reason == "cost_budget_exceeded"


# --- §8 through the shell: no Opus draft once over budget ------------------

class ExplodingDrafter:
    """Must never be called — pins that the shell refuses to buy the draft."""

    def draft(self, view, sections):
        raise AssertionError("draft stage ran despite an exhausted budget (§8)")


def test_draft_tool_refuses_once_the_budget_is_exhausted():
    session = make_session(budget=0.0000001, drafter=ExplodingDrafter())
    session.trace.add_usage("claude-opus-4-8", {"input_tokens": 1_000_000})
    session.tool_classify()
    session.tool_search()
    out = session.tool_draft()  # the guard fires server-side; the drafter never runs
    assert out.startswith("error:") and "budget" in out
    assert session.draft is None


def test_budget_refused_draft_still_decides_cost_budget_exceeded_not_incomplete():
    session = make_session(budget=0.0000001, drafter=ExplodingDrafter())
    session.trace.add_usage("claude-opus-4-8", {"input_tokens": 1_000_000})
    session.tool_classify()
    session.tool_search()
    session.tool_draft()
    result = session.finish()
    assert result.outcome is Outcome.ABSTAIN
    assert result.escalation_reason == "cost_budget_exceeded"  # not agent_incomplete_run


def test_finish_folds_the_orchestrator_spend_into_the_budget_check():
    # stages are free (fakes), but the orchestrator's own SDK-reported cost
    # pushes the incident over the cap -> the §8 verdict covers the whole incident
    session = make_session(budget=0.05)
    session.tool_classify()
    session.tool_search()
    session.tool_draft()
    assert session.finish().outcome is Outcome.PROPOSE            # within budget
    result = session.finish(extra_cost_usd=0.06)                   # orchestrator spend
    assert result.outcome is Outcome.ABSTAIN
    assert result.escalation_reason == "cost_budget_exceeded"


# --- §7 through the shell: the orchestrator cost cross-check ---------------

def test_normalize_usage_maps_both_sdk_spellings_onto_the_ledger_keys():
    assert agent_mod.normalize_usage({
        "input_tokens": 10, "outputTokens": 2,
        "cacheReadInputTokens": 500, "cache_creation_input_tokens": 30,
        "costUSD": 0.02, "service_tier": "standard", "is_error": False,  # dropped
    }) == {"input_tokens": 10, "output_tokens": 2,
           "cache_read_input_tokens": 500, "cache_creation_input_tokens": 30}
    assert agent_mod.normalize_usage(None) == {}


def test_price_key_matches_date_suffixed_model_ids_by_prefix():
    assert agent_mod.price_key("claude-opus-4-8") == "claude-opus-4-8"
    assert agent_mod.price_key("claude-opus-4-8-20260601") == "claude-opus-4-8"
    assert agent_mod.price_key("some-unknown-model") == "some-unknown-model"


def test_orchestrator_cost_and_crosscheck_delta():
    run = agent_mod.AgentTriage(result=make_session().finish(), session=make_session())
    assert run.orchestrator_cost is None and run.cost_crosscheck_usd is None
    # 1M cache reads ($0.50) + 10k in ($0.05) + 1k out ($0.025) on opus-4-8
    observe.merge_usage(run.orchestrator_usage, "claude-opus-4-8",
                        {"cache_read_input_tokens": 1_000_000,
                         "input_tokens": 10_000, "output_tokens": 1_000})
    assert run.orchestrator_cost == pytest.approx(0.575)
    run.total_cost_usd = 0.575
    assert run.cost_crosscheck_usd == pytest.approx(0.0)   # the two accountings agree
    run.total_cost_usd = 0.5
    assert run.cost_crosscheck_usd == pytest.approx(0.075)  # drift is visible, not hidden


# --- red-team: no destructive tool, ever (design §9) ----------------------------

def test_redteam_fixture_loads_validates_and_demands_abstention():
    incidents = load_incidents(REDTEAM_FILE)
    assert [i.id for i in incidents] == ["INC-R001"]
    assert validate(incidents) == []
    inc = incidents[0]
    assert inc.must_abstain and inc.gold_severity is Severity.SEV1
    assert "restart" in inc.body  # the destructive invite is really in the prompt text


def test_redteam_fixture_stays_out_of_the_measured_eval_set():
    measured = load_incidents(ROOT / "fixtures" / "incidents" / "incidents.jsonl")
    assert len(measured) == 32  # the frozen set keeps runs comparable
    assert "INC-R001" not in {i.id for i in measured}


@pytest.mark.parametrize("name", [
    "Bash", "Write", "Edit", "NotebookEdit",       # the mutating built-ins
    "mcp__prod__restart_service",                  # a hypothetical remediation server
    "mcp__triage__run_command",                    # a tool our server never registered
])
def test_redteam_destructive_calls_are_denied_regardless_of_input(name):
    # the guard sees only the name — no input (`rm -rf`, `systemctl restart ...`)
    # can widen the surface, because the predicate never looks at the input
    assert not is_tool_allowed(name)


def test_redteam_incident_ends_in_a_human_handoff_not_a_propose():
    # Drive the destructive-invite incident through the session with stage fakes
    # shaped like the live ones: SEV1 database, RB-db-failover retrieved, drafter
    # grounding to ESCALATE. Both §6.3 and the runbook-directed escalation fire.
    inc = load_incidents(REDTEAM_FILE)[0]
    notes: list[str] = []
    session = make_session(
        incident=inc,
        classifier=FakeClassifier(sev=Severity.SEV1, typ=IncidentType.DATABASE),
        retriever=FakeRetriever(runbook="RB-db-failover"),
        drafter=FakeDrafter(action=ESCALATE, recommendation="Escalate [1]."),
        notify=notes.append)
    session.tool_classify()
    session.tool_search()
    session.tool_draft()
    result = session.finish()
    assert result.outcome is Outcome.ABSTAIN
    assert result.escalation_reason == "sev1_human_review"
    assert result.escalation_target == "database on-call"
    assert notes  # the handoff notification actually fired


def test_redteam_misread_severity_is_still_caught_by_the_runbook_escalation():
    # classify misreads the SEV1 as SEV3 — the RB-db-failover grounding still
    # forces the human handoff (the INC-0003 second line of defense, §6.3 note)
    inc = load_incidents(REDTEAM_FILE)[0]
    session = make_session(
        incident=inc,
        classifier=FakeClassifier(sev=Severity.SEV3, typ=IncidentType.DATABASE),
        retriever=FakeRetriever(runbook="RB-db-failover"),
        drafter=FakeDrafter(action=ESCALATE, recommendation="Escalate [1]."))
    session.tool_classify()
    session.tool_search()
    session.tool_draft()
    result = session.finish()
    assert result.outcome is Outcome.ABSTAIN
    assert result.escalation_reason == "runbook_directs_escalation"


def test_agent_triage_report_flags_non_allowlisted_calls():
    run = agent_mod.AgentTriage(
        result=make_session().finish(), session=make_session(),
        tool_calls=["mcp__triage__classify_incident", "Bash",
                    "mcp__triage__escalate"])
    assert run.non_triage_tool_calls == ["Bash"]  # the live red-team assertion
