"""Offline tests for the eval harness skeleton (design §10).

A scripted fake Classifier (and a fake action Judge) drive the classify -> score
-> aggregate loop with no key/network. Asserts both the end-to-end run and the
`summarize()` math — including the metrics that stay n/a until their stage
(decide, retrieve, judge) so the harness can't silently fake them.
"""

from __future__ import annotations

from pathlib import Path

from triage import eval as eval_mod
from triage import schema
from triage.classify import Classification
from triage.schema import IncidentType, Outcome, Severity

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "incidents" / "incidents.jsonl"


class GoldClassifier:
    """A perfect fake: echoes each incident's gold labels. It needs the gold map,
    so it is a test-only oracle — the real classifier sees only the prompt view.
    `wrong_ids` forces a miss on chosen incidents to exercise <100% accuracy."""

    model = "fake-classifier"

    def __init__(self, incidents, *, wrong_ids=()):
        self.by_id = {i.id: i for i in incidents}
        self.wrong_ids = set(wrong_ids)

    def classify(self, view):
        inc = self.by_id[view["id"]]
        sev, typ = inc.gold_severity, inc.gold_type
        if inc.id in self.wrong_ids:
            sev = Severity.SEV4 if sev is not Severity.SEV4 else Severity.SEV1
            typ = IncidentType.UNKNOWN if typ is not IncidentType.UNKNOWN else IncidentType.NETWORK
        return Classification(sev, typ, 0.9, 0.85, usage={"input_tokens": 7, "output_tokens": 3})


class FakeJudge:
    """Semantic judge fake: agrees with key-match, except on `flip_ids` where it
    returns the opposite verdict — the disagreement path the judge note surfaces."""

    model = "fake-judge"

    def __init__(self, flip_ids=()):
        self.flip_ids = set(flip_ids)
        self.calls: list[str] = []

    def judge_action(self, view, proposed_action, gold_action):
        self.calls.append(view["id"])
        correct = proposed_action == gold_action
        if view["id"] in self.flip_ids:
            correct = not correct
        return eval_mod.ActionVerdict(correct=correct, reason="fake",
                                      usage={"input_tokens": 4, "output_tokens": 1})


class GoldRetriever:
    """Test-only oracle: surfaces the expected runbook for in-scope incidents and
    nothing for out-of-scope ones (no supporting runbook exists to retrieve)."""

    def __init__(self, incidents):
        self.by_id = {i.id: i for i in incidents}

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
    """Test-only oracle: grounds the gold action with a [1] citation, or abstains
    when no section covers the incident (out of scope)."""

    model = "fake-drafter"

    def __init__(self, incidents):
        self.by_id = {i.id: i for i in incidents}

    def draft(self, view, sections):
        from triage.draft import INSUFFICIENT_EVIDENCE, Draft, extract_citations
        inc = self.by_id[view["id"]]
        if not sections or not inc.in_scope:
            return Draft(action_key=INSUFFICIENT_EVIDENCE, recommendation="not covered")
        rec = "Apply the runbook's first response [1]."
        return Draft(action_key=inc.gold_action, recommendation=rec,
                     citations=extract_citations(rec, sections),
                     usage={"input_tokens": 9, "output_tokens": 2})


# --- pure helpers -----------------------------------------------------------

def test_action_matches_is_case_and_space_insensitive():
    assert eval_mod.action_matches("roll_back_last_deploy", "roll_back_last_deploy")
    assert eval_mod.action_matches("  ROLL_back_last_deploy ", "roll_back_last_deploy")
    assert not eval_mod.action_matches("halt_rollout_pin_previous", "roll_back_last_deploy")
    assert not eval_mod.action_matches(None, "anything")


def test_select_filters_by_scope_and_limit():
    incidents = schema.load_incidents(FIXTURE)
    assert all(i.in_scope for i in eval_mod.select(incidents, scope="in"))
    assert all(not i.in_scope for i in eval_mod.select(incidents, scope="out"))
    assert len(eval_mod.select(incidents, limit=3)) == 3


# --- end-to-end classify loop with a fake -----------------------------------

def test_evaluate_perfect_classifier_scores_100_and_leaves_later_stages_na():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents))
    s = eval_mod.summarize(rows)
    assert s["n_total"] == len(incidents)
    assert s["severity_accuracy"] == 1.0 and s["type_accuracy"] == 1.0
    assert s["severity_accuracy_all"] == 1.0 and s["type_accuracy_all"] == 1.0
    # classify-only run: decision / retrieval / action metrics are not measured yet
    assert s["abstention_rate"] is None
    assert s["recall_at_k"] is None
    assert s["action_correctness"] is None
    # fake model is unpriced -> $0, but every incident is still timed
    assert s["cost"]["total"] == 0.0
    assert len(s["latencies"]) == len(rows)


def test_evaluate_counts_misclassifications_over_in_scope():
    incidents = schema.load_incidents(FIXTURE)
    wrong = [i.id for i in incidents if i.in_scope][:2]
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents, wrong_ids=wrong), scope="in")
    s = eval_mod.summarize(rows)
    n = s["n_in_scope"]
    assert n >= 4
    assert s["severity_accuracy"] == (n - 2) / n
    assert s["type_accuracy"] == (n - 2) / n


# --- full pipeline: classify + retrieve + draft -> decide ----------------

def test_evaluate_full_pipeline_with_gold_oracles_is_safe_across_the_set():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents),
                             GoldRetriever(incidents), GoldDrafter(incidents))
    s = eval_mod.summarize(rows)
    answerable = [i for i in incidents if not i.must_abstain]
    assert s["n_decided"] == len(incidents)
    assert s["n_proposed"] == len(answerable)
    assert s["abstention_rate"] == 1.0
    assert s["n_false_abstentions"] == 0
    assert s["n_missed_escalations"] == 0
    assert s["n_judged"] == len(answerable) and s["action_correctness"] == 1.0
    assert s["recall_at_1"] == 1.0
    # usage flows per model from every stage that reported tokens
    assert set(s["usage_by_model"]) == {"fake-classifier", "fake-drafter"}


def test_evaluate_misread_sev1_is_still_caught_by_the_runbook_escalation():
    # the INC-0003 scenario (design §6.3 note): classify drops SEV1 to something
    # lower, but the grounded action is ESCALATE (RB-db-failover) -> the decision
    # contract still hands off to a human, so the miss is NOT a missed escalation.
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents, wrong_ids={"INC-0003"}),
                             GoldRetriever(incidents), GoldDrafter(incidents))
    (row,) = [r for r in rows if r.incident.id == "INC-0003"]
    assert row.pred_severity is not Severity.SEV1      # the classifier got it wrong
    assert row.outcome is Outcome.ABSTAIN              # the contract still escalated
    assert row.escalation_reason == "runbook_directs_escalation"
    assert eval_mod.summarize(rows)["n_missed_escalations"] == 0


# --- cost ceiling through the harness (design §8) -------------------------

class ExpensiveClassifier(GoldClassifier):
    """Gold labels, but priced usage far above the per-incident budget."""

    model = "claude-opus-4-8"  # priced -> the trace sees real dollars

    def classify(self, view):
        c = super().classify(view)
        c.usage = {"input_tokens": 40_000_000}  # $200 at $5/M input
        return c


class ExplodingDrafter:
    """Must never be called — pins that a tripped budget skips the Opus draft."""

    model = "fake-drafter"

    def draft(self, view, sections):
        raise AssertionError("draft stage ran despite an exhausted budget (§8)")


def test_evaluate_budget_trip_skips_the_draft_and_abstains():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, ExpensiveClassifier(incidents),
                             GoldRetriever(incidents), ExplodingDrafter(),
                             max_total_usd=float("inf"), limit=2)
    assert len(rows) == 2  # the aggregate cap was lifted; only the per-incident cap fires
    for row in rows:
        assert row.outcome is Outcome.ABSTAIN
        assert row.escalation_reason == "cost_budget_exceeded"
        assert row.draft_action is None and "draft" not in row.stage_seconds


def test_evaluate_aggregate_cap_stops_the_run_as_a_fail_safe():
    incidents = schema.load_incidents(FIXTURE)
    # default cap = budget x selected; the first incident alone costs ~$200,
    # so the run must stop right after it — fewer rows than selected.
    rows = eval_mod.evaluate(incidents, ExpensiveClassifier(incidents), limit=5)
    assert len(rows) == 1


# --- per-stage observability through the harness (design §7) --------------

def test_evaluate_records_per_stage_latency_for_the_report():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents),
                             GoldRetriever(incidents), GoldDrafter(incidents), limit=3)
    for row in rows:
        assert set(row.stage_seconds) == {"classify", "retrieve", "draft", "decide"}
    s = eval_mod.summarize(rows)
    assert all(len(s["stage_latencies"][name]) == 3
               for name in ("classify", "retrieve", "draft", "decide"))
    report = eval_mod.format_report(rows, s)
    assert "· classify" in report and "per stage" in report


# --- aggregation of the later-stage metrics (hand-built rows / fakes) --------

def test_summarize_scores_abstention_when_decisions_are_present():
    incidents = {i.id: i for i in schema.load_incidents(FIXTURE)}

    def row(inc_id, outcome):
        inc = incidents[inc_id]
        return eval_mod.EvalRow(incident=inc, pred_severity=inc.gold_severity,
                                pred_type=inc.gold_type, outcome=outcome)

    rows = [
        row("INC-0001", Outcome.PROPOSE),    # answerable, proposed -> fine
        row("INC-0010", Outcome.ABSTAIN),    # answerable, abstained -> false abstention
        row("INC-0008", Outcome.ABSTAIN),    # must-abstain (out of scope) -> correct
        row("INC-0003", Outcome.PROPOSE),    # must-abstain (SEV1) but PROPOSE -> missed escalation
    ]
    s = eval_mod.summarize(rows)
    assert s["n_decided"] == 4 and s["n_must_abstain"] == 2
    assert s["abstention_rate"] == 0.5          # INC-0008 abstained, INC-0003 did not
    assert s["n_false_abstentions"] == 1        # INC-0010
    assert s["n_missed_escalations"] == 1       # INC-0003 (the dangerous error)


def test_summarize_scores_recall_when_retrieval_is_present():
    incidents = {i.id: i for i in schema.load_incidents(FIXTURE)}
    rows = [
        eval_mod.EvalRow(incident=incidents["INC-0001"], retrieved_runbooks=["RB-app-5xx", "RB-latency"]),
        eval_mod.EvalRow(incident=incidents["INC-0002"], retrieved_runbooks=["RB-latency"]),  # miss
    ]
    s = eval_mod.summarize(rows)
    assert s["n_recallable"] == 2 and s["recall_at_k"] == 0.5


def test_judge_seam_and_action_correctness_summary():
    incidents = {i.id: i for i in schema.load_incidents(FIXTURE)}
    judge = FakeJudge()
    v = judge.judge_action({"id": "INC-0001"}, "roll_back_last_deploy", "roll_back_last_deploy")
    assert v.correct and v.usage  # the seam returns a verdict + usage
    rows = [
        eval_mod.EvalRow(incident=incidents["INC-0001"], action_correct=True),
        eval_mod.EvalRow(incident=incidents["INC-0005"], action_correct=False),
    ]
    s = eval_mod.summarize(rows)
    assert s["n_judged"] == 2 and s["action_correctness"] == 0.5


def test_format_report_renders_table_and_metrics_block():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents), limit=4)
    report = eval_mod.format_report(rows, eval_mod.summarize(rows))
    assert "=== metrics ===" in report
    assert "severity accuracy" in report and "abstention rate" in report
    assert "INC-0001" in report


# --- the LLM judge through the harness (design §10) -----------------------

def test_evaluate_judge_grades_only_propose_outcomes():
    incidents = schema.load_incidents(FIXTURE)
    judge = FakeJudge()
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents),
                             GoldRetriever(incidents), GoldDrafter(incidents), judge=judge)
    proposed = [r for r in rows if r.outcome is Outcome.PROPOSE]
    abstained = [r for r in rows if r.outcome is Outcome.ABSTAIN]
    assert sorted(judge.calls) == sorted(r.incident.id for r in proposed)
    assert all(r.action_correct_judge is True for r in proposed)   # gold oracles -> agree
    assert all(r.action_correct_judge is None for r in abstained)  # nothing to grade
    # the judge's tokens are filed apart from the pipeline ledger (§10 honesty)
    assert all("fake-judge" not in r.usage_by_model for r in rows)
    assert all(set(r.judge_usage_by_model) == {"fake-judge"} for r in proposed)


def test_summarize_judge_note_stays_out_of_the_headline_ledger():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents),
                             GoldRetriever(incidents), GoldDrafter(incidents),
                             judge=FakeJudge())
    s = eval_mod.summarize(rows)
    assert s["n_judge"] == s["n_proposed"]
    assert s["judge_action_correctness"] == 1.0
    assert s["judge_agreement"] == 1.0 and s["judge_disagreement_ids"] == []
    # separate ledgers: pipeline cost has no judge tokens, judge cost is its own key
    assert "fake-judge" not in s["usage_by_model"]
    assert s["judge_usage_by_model"]["fake-judge"]["input_tokens"] > 0
    assert s["judge_cost"]["total"] == 0.0  # fake model is unpriced -> $0


def test_judge_disagreement_is_surfaced_in_summary_and_report():
    incidents = schema.load_incidents(FIXTURE)
    flip = "INC-0001"  # answerable; gold oracles propose the gold key -> judge flips to False
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents),
                             GoldRetriever(incidents), GoldDrafter(incidents),
                             judge=FakeJudge(flip_ids={flip}))
    s = eval_mod.summarize(rows)
    assert s["judge_disagreement_ids"] == [flip]
    assert s["judge_agreement"] == (s["n_judge"] - 1) / s["n_judge"]
    report = eval_mod.format_report(rows, s)
    assert "LLM-judge note" in report and "not a headline" in report
    assert flip in report.split("LLM-judge note")[1]  # named in the note


def test_report_has_no_judge_note_when_no_judge_ran():
    incidents = schema.load_incidents(FIXTURE)
    rows = eval_mod.evaluate(incidents, GoldClassifier(incidents),
                             GoldRetriever(incidents), GoldDrafter(incidents))
    report = eval_mod.format_report(rows, eval_mod.summarize(rows))
    assert "LLM-judge note" not in report  # never renders a metric it didn't measure


def test_parse_action_verdict_and_judge_prompt_assembly():
    v = eval_mod.parse_action_verdict('{"correct": true, "reason": "same effect"}')
    assert v.correct is True and v.reason == "same effect"
    assert not eval_mod.parse_action_verdict('{"correct": false, "reason": ""}').correct
    view = {"id": "INC-0001", "title": "T", "body": "B", "source": "pagerduty"}
    user = eval_mod.build_judge_user(view, "roll_back_last_deploy", "roll_back_last_deploy")
    assert "Proposed action: roll_back_last_deploy" in user
    assert "Reference action: roll_back_last_deploy" in user
    assert "INC-0001" in user and "B" in user
    # the closed verdict schema: a boolean + a reason, nothing else
    assert eval_mod.JUDGE_SCHEMA["required"] == ["correct", "reason"]
    assert eval_mod.JUDGE_SCHEMA["additionalProperties"] is False
