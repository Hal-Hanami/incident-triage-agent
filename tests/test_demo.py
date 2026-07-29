"""Offline tests for the key-free cached demo (design §11).

Gold fakes drive the bake -> JSON -> restore -> render round trip with no
key/network, and the committed `demo/examples.json` is validated as-is: it must
parse, restore, and render through the same CLI renderers a clone would use —
and it must stay body-free (citations carry section paths, never runbook text).
"""

from __future__ import annotations

import json
from pathlib import Path

from triage import demo as demo_mod
from triage import observe, schema
from triage.__main__ import _format_triage
from triage.classify import Classification
from triage.schema import Outcome, Severity

ROOT = Path(__file__).resolve().parent.parent


class GoldClassifier:
    model = "fake-classifier"

    def __init__(self, incidents):
        self.by_id = {i.id: i for i in incidents.values()}

    def classify(self, view):
        inc = self.by_id[view["id"]]
        return Classification(inc.gold_severity, inc.gold_type, 0.9, 0.85,
                              usage={"input_tokens": 7, "output_tokens": 3})


class GoldRetriever:
    def __init__(self, incidents):
        self.by_id = {i.id: i for i in incidents.values()}

    def retrieve(self, view):
        from triage.retrieve import Retrieval, RunbookHit
        inc = self.by_id[view["id"]]
        if not inc.expected_runbook:
            return Retrieval(runbooks=[])
        rb = inc.expected_runbook
        hit = RunbookHit(runbook_id=rb, score=0.9,
                         section_path=f"{rb} > First response", url="x", text="...")
        return Retrieval(runbooks=[hit],
                         chunks=[{"source_url": rb, "section_path": hit.section_path,
                                  "url": "x", "text": "..."}])


class GoldDrafter:
    model = "fake-drafter"

    def __init__(self, incidents):
        self.by_id = {i.id: i for i in incidents.values()}

    def draft(self, view, sections):
        from triage.draft import INSUFFICIENT_EVIDENCE, Draft, extract_citations
        inc = self.by_id[view["id"]]
        if not sections or not inc.in_scope:
            return Draft(action_key=INSUFFICIENT_EVIDENCE, recommendation="not covered")
        rec = "Apply the runbook's first response [1]."
        return Draft(action_key=inc.gold_action, recommendation=rec,
                     citations=extract_citations(rec, sections),
                     usage={"input_tokens": 9, "output_tokens": 2})


def bake_with_fakes():
    incidents = demo_mod.incident_index()
    return demo_mod.bake_examples(GoldClassifier(incidents), GoldRetriever(incidents),
                                  GoldDrafter(incidents), log=lambda *_: None)


# --- the showcase itself ------------------------------------------------------

def test_showcase_covers_the_four_story_beats_with_real_fixture_ids():
    incidents = demo_mod.incident_index()
    assert all(spec["id"] in incidents for spec in demo_mod.SHOWCASE)
    stories = " ".join(spec["story"] for spec in demo_mod.SHOWCASE)
    for beat in ("§6.2", "§6.3", "§8", "§9"):  # abstain / SEV1 / budget / red-team
        assert beat in stories
    assert "INC-R001" in {s["id"] for s in demo_mod.SHOWCASE}  # the red-team ticket


# --- bake -> restore -> render round trip (fakes, offline) ---------------------

def test_bake_round_trips_through_json_and_the_cli_renderers():
    payload = json.loads(json.dumps(bake_with_fakes()))  # force JSON-serializable
    assert len(payload["examples"]) == len(demo_mod.SHOWCASE)
    for ex in payload["examples"]:
        result, trace, budget = demo_mod.restore(ex)
        assert result.incident_id == ex["incident"]["id"]
        rendered = _format_triage(ex["incident"]["title"], result)
        footer = "\n".join(observe.format_trace(trace, budget=budget))
        assert f"OUTCOME     {ex['result']['outcome']}" in rendered
        assert "--- trace ---" in footer and "design §8" in footer


def test_bake_captures_the_expected_outcomes():
    by_story = {}
    for ex in bake_with_fakes()["examples"]:
        by_story[ex["story"]] = ex
    outcomes = {ex["incident"]["id"]: ex["result"]["outcome"] for ex in by_story.values()}
    assert outcomes["INC-0008"] == Outcome.ABSTAIN.value   # out of scope
    assert outcomes["INC-R001"] == Outcome.ABSTAIN.value   # red-team -> human handoff
    propose = [ex for ex in by_story.values()
               if ex["result"]["outcome"] == Outcome.PROPOSE.value]
    assert propose and all(ex["result"]["citations"] for ex in propose)  # cited or not proposed


def test_restore_reprices_usd_from_tokens_not_the_baked_figure():
    ex = bake_with_fakes()["examples"][0]
    ex["trace"]["cost_by_model"] = {"claude-haiku-4-5":
                                    {"input_tokens": 1_000_000, "output_tokens": 0, "usd": 999.0}}
    _, trace, _ = demo_mod.restore(ex)
    # the stale baked "usd" is dropped; dollars come from PRICING over the tokens
    assert trace.cost() == observe.PRICING["claude-haiku-4-5"]["input"]


# --- the committed demo file (what a key-less clone actually replays) ----------

def test_committed_examples_file_is_valid_and_body_free():
    payload = demo_mod.load_examples()
    assert payload["examples"], "demo/examples.json must ship with baked entries"
    assert payload["generated_at"] and payload["bake_cost_usd"] > 0
    for ex in payload["examples"]:
        result, trace, budget = demo_mod.restore(ex)
        assert isinstance(result.severity, Severity)
        # body-free contract (§11): a citation is a pointer, never runbook text
        for c in ex["result"]["citations"]:
            assert set(c) == {"n", "runbook_id", "section", "source"}
        # renders through the real CLI surface without touching the network
        assert _format_triage(ex["incident"]["title"], result)
        assert observe.format_trace(trace, budget=budget)
    ids = {ex["incident"]["id"] for ex in payload["examples"]}
    assert {"INC-0001", "INC-0003", "INC-0008", "INC-R001"} <= ids
    # the staged §8 trip really tripped: an ABSTAIN(cost_budget_exceeded) entry exists
    assert any(ex["result"]["escalation_reason"] == "cost_budget_exceeded"
               for ex in payload["examples"])


def test_load_examples_missing_file_says_how_to_rebake(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="bake-demo"):
        demo_mod.load_examples(tmp_path / "nope.json")
