"""Offline tests for the draft stage (design §4.3, §5).

Pure-function coverage (prompt assembly, citation extraction, verdict parsing,
the grounded / citation-backed contract) plus the `draft_response` orchestration
with a scripted fake Drafter — no key, no network. The abstention sentinel and
the "no in-range [n] -> not proposable" rule are what the decider (§6.1) gates
on, so they get the tight assertions.
"""

from __future__ import annotations

import json

import pytest

from triage import draft as draft_mod
from triage import observe
from triage.draft import (INSUFFICIENT_EVIDENCE, Draft, build_draft_user,
                          draft_response, extract_citations, format_sections,
                          parse_draft)
from triage.schema import ESCALATE

SECTIONS = [
    {"source_url": "RB-app-5xx", "section_path": "RB-app-5xx > Rollback",
     "url": "fixtures/runbooks/RB-app-5xx.md#rollback", "text": "Roll back the last deploy."},
    {"source_url": "RB-db-failover", "section_path": "RB-db-failover > Policy",
     "url": "fixtures/runbooks/RB-db-failover.md#policy", "text": "Failover is human-only."},
]

VIEW = {"id": "INC-0001", "title": "Elevated 5xx after deploy",
        "body": "Error rate jumped from 0.1% to 4% right after release 2024-06-17.1.",
        "source": "pagerduty"}


# --- the Draft contract (what the decider gates on, §6.1) --------------------

def test_abstaining_draft_is_neither_grounded_nor_citation_backed():
    d = Draft(action_key=INSUFFICIENT_EVIDENCE, recommendation="No section covers this.")
    assert not d.grounded
    assert not d.is_citation_backed


def test_grounded_draft_without_citations_is_not_proposable():
    d = Draft(action_key="roll_back_last_deploy", recommendation="Roll back.")  # no [n]
    assert d.grounded
    assert not d.is_citation_backed


def test_grounded_cited_draft_is_citation_backed():
    d = Draft(action_key="roll_back_last_deploy", recommendation="Roll back [1].",
              citations=extract_citations("Roll back [1].", SECTIONS))
    assert d.grounded and d.is_citation_backed


# --- prompt assembly (§5: numbered sections, and nothing else) ---------------

def test_format_sections_numbers_from_1_with_path_and_text():
    text = format_sections(SECTIONS)
    assert "[1] RB-app-5xx > Rollback" in text
    assert "[2] RB-db-failover > Policy" in text
    assert "Roll back the last deploy." in text


def test_build_draft_user_carries_only_the_prompt_view_fields():
    prompt = build_draft_user(VIEW, SECTIONS)
    assert "INC-0001" in prompt and "pagerduty" in prompt
    assert "Elevated 5xx after deploy" in prompt
    assert "[1] RB-app-5xx > Rollback" in prompt
    # no gold labels / eval fields anywhere near the model
    assert "gold" not in prompt and "expected_runbook" not in prompt


# --- citation extraction (§5: in-range, dedup, order) ------------------------

def test_extract_citations_maps_n_to_its_section():
    (c,) = extract_citations("Roll back the last deploy [1].", SECTIONS)
    assert (c.n, c.runbook_id) == (1, "RB-app-5xx")
    assert c.section == "RB-app-5xx > Rollback"
    assert c.source == "fixtures/runbooks/RB-app-5xx.md#rollback"


def test_extract_citations_dedups_and_keeps_first_occurrence_order():
    cs = extract_citations("Do X [2], then Y [1], then Z [2].", SECTIONS)
    assert [c.n for c in cs] == [2, 1]


def test_extract_citations_drops_out_of_range_markers():
    assert extract_citations("Hallucinated [3] and [0].", SECTIONS) == []


# --- verdict parsing ----------------------------------------------------------

def test_parse_draft_happy_path_yields_cited_proposable_draft():
    text = json.dumps({"action": "roll_back_last_deploy",
                       "recommendation": "Roll back the last deploy [1]."})
    d = parse_draft(text, SECTIONS)
    assert d.action_key == "roll_back_last_deploy"
    assert d.is_citation_backed and [c.n for c in d.citations] == [1]


def test_parse_draft_escalate_action_is_grounded():
    text = json.dumps({"action": ESCALATE,
                       "recommendation": "Failover is a human call [2]."})
    d = parse_draft(text, SECTIONS)
    assert d.action_key == ESCALATE and d.grounded


def test_parse_draft_abstention_carries_no_citations():
    text = json.dumps({"action": INSUFFICIENT_EVIDENCE,
                       "recommendation": "No provided section addresses this [1]."})
    d = parse_draft(text, SECTIONS)
    assert d.action_key == INSUFFICIENT_EVIDENCE
    assert d.citations == [] and not d.is_citation_backed


def test_parse_draft_rejects_unknown_action():
    text = json.dumps({"action": "reboot_everything", "recommendation": "no [1]"})
    with pytest.raises(ValueError):
        parse_draft(text, SECTIONS)


# --- orchestration (trace/timing + per-model usage, §7) -----------------------

class FakeDrafter:
    model = "fake-drafter"

    def __init__(self, draft: Draft):
        self._draft = draft
        self.calls: list[tuple[dict, list]] = []

    def draft(self, view, sections):
        self.calls.append((view, sections))
        return self._draft


def test_draft_response_times_the_stage_and_files_usage_by_model():
    verdict = Draft(action_key="roll_back_last_deploy", recommendation="Roll back [1].",
                    citations=extract_citations("[1]", SECTIONS),
                    usage={"input_tokens": 11, "output_tokens": 5})
    drafter = FakeDrafter(verdict)
    trace = observe.Trace()
    result = draft_response(VIEW, SECTIONS, drafter, trace=trace)
    assert result is verdict
    assert drafter.calls == [(VIEW, SECTIONS)]
    assert [name for name, _ in trace.spans] == ["draft"]
    assert trace.usage_by_model == {"fake-drafter": {"input_tokens": 11, "output_tokens": 5}}
