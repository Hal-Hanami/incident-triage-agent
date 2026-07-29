"""Offline tests for the incident/triage data model and the committed fixtures.

No key, no network. Pins the schema invariants and asserts the synthetic incident
set is internally consistent, so the eval ground truth can be trusted.
"""

from __future__ import annotations

from pathlib import Path

from triage import schema

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "incidents" / "incidents.jsonl"


def test_fixture_loads_and_is_valid():
    incidents = schema.load_incidents(FIXTURE)
    assert len(incidents) >= 8
    assert schema.validate(incidents) == []  # committed fixtures must be clean


def test_fixture_has_both_answerable_and_abstention_cases():
    incidents = schema.load_incidents(FIXTURE)
    answerable = [i for i in incidents if not i.must_abstain]
    abstain = [i for i in incidents if i.must_abstain]
    # The abstention story needs both kinds present, like tech-docs-rag's in/out-of-corpus split.
    assert answerable, "need in-scope incidents to measure false-abstention"
    assert abstain, "need abstention tests (out-of-scope or SEV1/ESCALATE)"


def test_sev1_database_incident_must_abstain_even_with_a_runbook():
    incidents = {i.id: i for i in schema.load_incidents(FIXTURE)}
    inc = incidents["INC-0003"]
    assert inc.gold_severity is schema.Severity.SEV1
    assert inc.in_scope and inc.expected_runbook == "RB-db-failover"
    # SEV1 is escalated to a human even though a runbook exists (design §6.3).
    assert inc.must_abstain and inc.gold_action == schema.ESCALATE


def test_prompt_view_hides_gold_labels():
    inc = schema.load_incidents(FIXTURE)[0]
    view = inc.prompt_view()
    assert set(view) == {"id", "title", "body", "source"}
    # ground-truth labels must never leak into what the agent sees
    assert "gold_severity" not in view and "notes" not in view


def test_validate_flags_inconsistent_incident():
    bad = schema.Incident(
        id="X", title="t", body="b", source="ticket",
        gold_severity=schema.Severity.SEV3, gold_type=schema.IncidentType.DATA_PIPELINE,
        gold_action="some_fix", in_scope=False, expected_runbook=None,
    )
    problems = schema.validate([bad])
    # out-of-scope incident with a non-ESCALATE action is a fixture bug
    assert any("ESCALATE" in p for p in problems)


def test_validate_flags_unknown_action_key():
    bad = schema.Incident(
        id="Y", title="t", body="b", source="ticket",
        gold_severity=schema.Severity.SEV2, gold_type=schema.IncidentType.APP_ERROR,
        gold_action="reboot_everything", in_scope=True, expected_runbook="RB-app-5xx",
    )
    problems = schema.validate([bad])
    assert any("gold_action" in p for p in problems)


def test_validate_flags_action_runbook_mismatch():
    bad = schema.Incident(
        id="Z", title="t", body="b", source="ticket",
        gold_severity=schema.Severity.SEV2, gold_type=schema.IncidentType.APP_ERROR,
        gold_action="relieve_disk_pressure", in_scope=True, expected_runbook="RB-app-5xx",
    )
    problems = schema.validate([bad])
    # a real action key, but it belongs to a different runbook than the one expected
    assert any("does not match" in p for p in problems)


def test_runbook_actions_are_all_valid_keys():
    # the per-runbook actions plus ESCALATE are exactly the allowed gold_action set
    assert set(schema.RUNBOOK_ACTIONS.values()) | {schema.ESCALATE} == set(schema.ACTION_KEYS)


def test_fixture_covers_every_type_and_severity():
    incidents = schema.load_incidents(FIXTURE)
    # enough volume + breadth to report (type × severity) accuracy with meaning (design §3)
    assert len(incidents) >= 30
    assert {i.gold_type for i in incidents} == set(schema.IncidentType)
    assert {i.gold_severity for i in incidents} == set(schema.Severity)
