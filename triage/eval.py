"""Stage: offline eval harness — measure the pipeline against the synthetic gold
set (design §10), reported as one table, the same shape as tech-docs-rag's `rag eval`.

Every stage is optional and independent, because each one costs money to run and
a measurement should never require paying for a stage it does not score:

  - a `Classifier` alone scores **severity / type accuracy** (no index needed)
  - a `Retriever` alone scores **recall@1 / @3 / @k + MRR** (no Anthropic spend).
    Rank-sensitive metrics, so a Voyage rerank's lift is visible at all
  - all of them together score the decision contract: **abstention rate /
    false-abstention / missed-escalation** and key-match action correctness

Metrics whose stage did not run report `n/a` rather than a plausible-looking
zero — a number that was never measured must not be mistakable for one that was.

Every incident runs under its own `Trace` (design §7, §8), so the report carries
**per-stage p50/p95 latency** next to the end-to-end percentiles; the
per-incident budget is checked before the expensive draft stage (a tripped budget
skips the Opus spend and the decision is ABSTAIN `cost_budget_exceeded`); and the
whole run enforces an **aggregate cost cap** as a fail-safe — once crossed, no
further incident runs.

`ClaudeJudge` is the LLM-as-judge **semantic second opinion** on each PROPOSE's
action (`claude-haiku-4-5`, structured verdict, reference-guided by the gold
action). It exists for the case key-match cannot see: a drafter that *stretches*
a loosely-fitting runbook to an incident it doesn't really address. Its verdicts
render as a clearly-labeled note — never the headline — and its spend is filed
apart from the pipeline ledger, so cost figures stay comparable across runs.

The harness follows tech-docs-rag's `rag/eval.py`: every LLM seam sits behind a
Protocol, so the whole loop runs with fakes — no key, no network — keeping the
offline suite green per commit.
The deterministic metrics (accuracy, abstention, recall) are the honest headline;
the LLM-judge action score is noisy (tech-docs-rag measured ±~20pt single-shot) and
stays a clearly-labeled note (design §10 honesty discipline).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from . import classify as classify_mod
from . import decide as decide_mod
from . import draft as draft_mod
from . import observe
from . import retrieve as retrieve_mod
from .schema import Incident, IncidentType, Outcome, Severity

# design §2/§10: the action-faithfulness judge runs on the cheap model.
JUDGE_MODEL = "claude-haiku-4-5"


def action_matches(proposed: str | None, gold: str) -> bool:
    """Deterministic key-match for action correctness (design §10).

    The proposed action is correct iff its runbook action key equals the gold key
    (case/whitespace-insensitive). This is the cheap, reproducible half of the
    action metric; the LLM judge is the noisy semantic half layered on top.
    """
    if proposed is None:
        return False
    return proposed.strip().lower() == gold.strip().lower()


@dataclass
class ActionVerdict:
    """What a `Judge` returns for action correctness: a verdict + optional
    usage. The semantic check layered on top of the deterministic `action_matches`.
    """

    correct: bool
    reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class Judge(Protocol):
    """LLM-as-judge seam for action correctness — faked offline in tests.
    Mirrors tech-docs-rag's faithfulness `Judge`; the real `claude-haiku-4-5`
    implementation (`ClaudeJudge`) plugs in here.
    """

    def judge_action(self, view: dict[str, str], proposed_action: str,
                     gold_action: str) -> ActionVerdict: ...


@dataclass
class EvalRow:
    """Per-incident eval outcome. Fields are populated by whichever stages have
    run, and stay `None`/empty until that stage runs:

      - pred_severity / pred_type / *_confidence: classify
      - retrieved_runbooks / retrieval_top_score: retrieve
      - outcome / proposed_action / escalation_*: decide
      - draft_action / draft_grounded: draft   (diagnostics)
      - action_correct: key-match (deterministic)
      - action_correct_judge / judge_reason: LLM judge (semantic note)

    `usage_by_model` keys tokens by model for per-model cost (§7); `latency_s` is
    the end-to-end wall-clock for the incident, feeding the p50/p95 percentiles;
    `stage_seconds` is the per-stage split from the incident's Trace spans,
    feeding the per-stage p50/p95 (§7). `judge_usage_by_model` is filed
    apart from `usage_by_model` on purpose: the judge measures the pipeline, it
    is not part of it, so the pipeline's cost table stays comparable across runs.
    """

    incident: Incident
    pred_severity: Severity | None = None
    pred_type: IncidentType | None = None
    severity_confidence: float | None = None
    type_confidence: float | None = None
    retrieved_runbooks: list[str] = field(default_factory=list)
    retrieval_top_score: float | None = None
    outcome: Outcome | None = None
    proposed_action: str | None = None
    escalation_target: str | None = None
    escalation_reason: str | None = None
    draft_action: str | None = None
    draft_grounded: bool | None = None
    action_correct: bool | None = None
    action_correct_judge: bool | None = None
    judge_reason: str = ""
    judge_usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    latency_s: float = 0.0
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def severity_correct(self) -> bool:
        return self.pred_severity is not None and self.pred_severity == self.incident.gold_severity

    @property
    def type_correct(self) -> bool:
        return self.pred_type is not None and self.pred_type == self.incident.gold_type

    @property
    def expected_runbook_rank(self) -> int:
        """1-based rank of the expected runbook among those retrieved (0 = miss/none).
        The rank-sensitive recall signal a rerank moves (design §10)."""
        expected = self.incident.expected_runbook
        if not expected or expected not in self.retrieved_runbooks:
            return 0
        return self.retrieved_runbooks.index(expected) + 1


def select(incidents: Iterable[Incident], *, scope: str = "all", limit: int = 0) -> list[Incident]:
    """Filter the incident set by scope (in / out / all) and cap the count.

    `limit` bounds spend when the real classifier runs against the live API.
    """
    items = list(incidents)
    if scope == "in":
        items = [i for i in items if i.in_scope]
    elif scope == "out":
        items = [i for i in items if not i.in_scope]
    if limit > 0:
        items = items[:limit]
    return items


def evaluate(incidents: Iterable[Incident],
             classifier: classify_mod.Classifier | None = None,
             retriever: retrieve_mod.Retriever | None = None,
             drafter: draft_mod.Drafter | None = None,
             *, judge: Judge | None = None,
             thresholds: decide_mod.Thresholds = decide_mod.DEFAULT_THRESHOLDS,
             budget: float = observe.DEFAULT_INCIDENT_BUDGET_USD,
             max_total_usd: float | None = None,
             scope: str = "all", limit: int = 0) -> list[EvalRow]:
    """Run each incident through the wired stages and record a scored `EvalRow`.

    classify and retrieve are independent and optional: pass a classifier
    for accuracy, a retriever for recall, or both. The decision contract needs
    all three model stages, so when `classifier`, `retriever`, and `drafter` are all
    present each incident also runs draft → decide (§6) and records its outcome,
    proposed action / escalation, and key-match action correctness (§10).

    A `judge` grades each PROPOSE semantically after the decision — a second
    opinion on the action, reference-guided by the gold key. It runs outside the
    incident's Trace and latency window and its tokens are filed in
    `judge_usage_by_model`: the judge measures the pipeline rather than being a
    stage of it (§10 honesty — headline cost/latency stay comparable across runs).
    Judge spend still counts against the aggregate §8 fail-safe, because it is
    real spend of this run.

    Observability + cost ceiling (design §7/§8): each incident runs under its
    own Trace, so tokens are filed per model and every stage is timed
    (`stage_seconds`) alongside the end-to-end `latency_s`. The §8 per-incident
    budget is checked *before* the expensive draft stage — a tripped budget skips
    the Opus spend and decides ABSTAIN(`cost_budget_exceeded`); classify/retrieve
    still run because they are what this harness measures (the production path,
    `run_triage`, skips every paid stage once over budget). The run also enforces
    an aggregate cap as a fail-safe: once the summed cost crosses `max_total_usd`
    (default: the per-incident budget × the selection size) no further incident
    runs, so the caller sees fewer rows than it selected.
    """
    rows: list[EvalRow] = []
    selected = select(incidents, scope=scope, limit=limit)
    if max_total_usd is None:
        max_total_usd = budget * len(selected)  # §8: the aggregate fail-safe cap
    total_spent = 0.0
    full_pipeline = classifier is not None and retriever is not None and drafter is not None
    for inc in selected:
        if total_spent > max_total_usd:
            break  # §8 aggregate cap tripped — stop buying stages entirely
        view = inc.prompt_view()
        t0 = time.perf_counter()
        trace = observe.Trace()
        row = EvalRow(incident=inc)
        cls = ret = None

        if classifier is not None:
            cls = classify_mod.classify_incident(view, classifier, trace=trace)
            row.pred_severity = cls.severity
            row.pred_type = cls.type
            row.severity_confidence = cls.severity_confidence
            row.type_confidence = cls.type_confidence

        if retriever is not None:
            ret = retrieve_mod.retrieve_runbooks(view, retriever, trace=trace)
            row.retrieved_runbooks = ret.runbook_ids
            row.retrieval_top_score = ret.top_score

        if full_pipeline:
            # §8: checked before the expensive stage — over budget = no Opus draft.
            exceeded = trace.cost() > budget
            drf = None
            if not exceeded:
                drf = draft_mod.draft_response(view, ret.chunks, drafter, trace=trace)
                row.draft_action = drf.action_key
                row.draft_grounded = drf.grounded
                exceeded = trace.cost() > budget
            with trace.span("decide"):
                decision = decide_mod.decide(cls, ret, drf, thresholds=thresholds,
                                             budget_exceeded=exceeded)
            row.outcome = decision.outcome
            row.proposed_action = decision.proposed_action
            row.escalation_target = decision.escalation_target
            row.escalation_reason = decision.escalation_reason
            if decision.outcome is Outcome.PROPOSE:  # key-match action correctness (§10); the judge is separate
                row.action_correct = action_matches(decision.proposed_action, inc.gold_action)

        row.usage_by_model = trace.usage_by_model
        row.stage_seconds = dict(trace.spans)
        row.latency_s = time.perf_counter() - t0  # pipeline only — the judge below is measurement
        total_spent += trace.cost()

        if judge is not None and row.outcome is Outcome.PROPOSE:
            verdict = judge.judge_action(view, row.proposed_action, inc.gold_action)
            row.action_correct_judge = verdict.correct
            row.judge_reason = verdict.reason
            if verdict.usage:
                observe.merge_usage(row.judge_usage_by_model,
                                    getattr(judge, "model", JUDGE_MODEL), verdict.usage)
            total_spent += observe.cost_usd(row.judge_usage_by_model)["total"]
        rows.append(row)
    return rows


def summarize(rows: Sequence[EvalRow]) -> dict[str, Any]:
    """Aggregate per-incident rows into the headline metrics (pure -> testable).

    Accuracy is over in-scope incidents (design §10) with an overall figure
    alongside. Decision metrics (abstention/false-abstention/missed-escalation)
    need the decider, recall needs retrieval, and action correctness
    needs the drafter/judge; each reads `n/a` (None) until its stage has
    run on the rows, so the harness never reports a metric it didn't measure.
    """
    def ratio(num: int, den: int) -> float | None:
        return (num / den) if den else None

    classified = [r for r in rows if r.pred_severity is not None]
    in_scope = [r for r in classified if r.incident.in_scope]

    # --- classification accuracy ---
    sev_acc_in = ratio(sum(r.severity_correct for r in in_scope), len(in_scope))
    type_acc_in = ratio(sum(r.type_correct for r in in_scope), len(in_scope))
    sev_acc_all = ratio(sum(r.severity_correct for r in classified), len(classified))
    type_acc_all = ratio(sum(r.type_correct for r in classified), len(classified))

    # --- decision-dependent metrics: only over incidents a decision ran on ---
    decided = [r for r in rows if r.outcome is not None]
    must_abstain = [r for r in decided if r.incident.must_abstain]
    answerable = [r for r in decided if not r.incident.must_abstain]
    abstained_correctly = [r for r in must_abstain if r.outcome is Outcome.ABSTAIN]
    false_abstentions = [r for r in answerable if r.outcome is Outcome.ABSTAIN]
    missed_escalations = [r for r in must_abstain if r.outcome is Outcome.PROPOSE]

    # --- retrieval recall: rank of the expected runbook among those retrieved,
    # over in-scope incidents that name one. recall@k (any rank) saturates fast with
    # only a handful of runbooks, so recall@1/@3 + MRR are where a Voyage rerank's lift
    # shows — tech-docs-rag's honesty pattern (design §10). ---
    recallable = [r for r in rows
                  if r.retrieved_runbooks and r.incident.in_scope and r.incident.expected_runbook]
    ranks = [r.expected_runbook_rank for r in recallable]
    n_rec = len(recallable)

    def recall_at(cutoff: int) -> float | None:
        return ratio(sum(1 for x in ranks if 0 < x <= cutoff), n_rec)

    retrieval_mrr = (sum(1.0 / x for x in ranks if x) / n_rec) if n_rec else None

    # --- action correctness: deterministic key-match over PROPOSE outcomes ---
    judged = [r for r in rows if r.action_correct is not None]

    # --- the LLM-judge note (§10): semantic second opinion — never a headline.
    # Aggregated separately (own token ledger, own cost) so the pipeline figures
    # above stay comparable across runs. ---
    judge_rows = [r for r in rows if r.action_correct_judge is not None]
    judge_both = [r for r in judge_rows if r.action_correct is not None]
    judge_usage_by_model: dict[str, dict[str, int]] = {}
    for r in judge_rows:
        for model, usage in r.judge_usage_by_model.items():
            observe.merge_usage(judge_usage_by_model, model, usage)

    usage_by_model: dict[str, dict[str, int]] = {}
    for r in rows:
        for model, usage in r.usage_by_model.items():
            observe.merge_usage(usage_by_model, model, usage)
    latencies = [r.latency_s for r in rows if r.latency_s]

    # per-stage latency samples (§7) — the split behind the per-stage p50/p95
    stage_latencies: dict[str, list[float]] = {}
    for r in rows:
        for name, secs in r.stage_seconds.items():
            stage_latencies.setdefault(name, []).append(secs)

    return {
        "n_total": len(rows),
        "n_classified": len(classified),
        "n_in_scope": len(in_scope),
        "severity_accuracy": sev_acc_in,
        "type_accuracy": type_acc_in,
        "severity_accuracy_all": sev_acc_all,
        "type_accuracy_all": type_acc_all,
        "n_decided": len(decided),
        "n_proposed": sum(1 for r in decided if r.outcome is Outcome.PROPOSE),
        "n_abstained": sum(1 for r in decided if r.outcome is Outcome.ABSTAIN),
        "n_must_abstain": len(must_abstain),
        "abstention_rate": ratio(len(abstained_correctly), len(must_abstain)),
        "n_false_abstentions": len(false_abstentions),
        "false_abstention_ids": [r.incident.id for r in false_abstentions],
        "n_missed_escalations": len(missed_escalations),
        "missed_escalation_ids": [r.incident.id for r in missed_escalations],
        "n_recallable": n_rec,
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_k": recall_at(10**9),  # any rank > 0 = retrieved within top-k
        "retrieval_mrr": retrieval_mrr,
        "n_judged": len(judged),
        "action_correctness": ratio(sum(1 for r in judged if r.action_correct), len(judged)),
        "n_judge": len(judge_rows),
        "judge_action_correctness": ratio(sum(1 for r in judge_rows if r.action_correct_judge),
                                          len(judge_rows)),
        "judge_agreement": ratio(sum(1 for r in judge_both
                                     if r.action_correct_judge == r.action_correct),
                                 len(judge_both)),
        "judge_disagreement_ids": [r.incident.id for r in judge_both
                                   if r.action_correct_judge != r.action_correct],
        "judge_usage_by_model": judge_usage_by_model,
        "judge_cost": observe.cost_usd(judge_usage_by_model),
        "usage_by_model": usage_by_model,
        "cost": observe.cost_usd(usage_by_model),
        "latencies": latencies,
        "stage_latencies": stage_latencies,
    }


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x * 100:5.1f}%"


def _mrr(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:6.3f}"


def _decision_cells(r: EvalRow) -> tuple[str, str]:
    """The (outcome, detail) columns for one row: the proposed action (+ key-match
    mark) for a PROPOSE, or the named escalation target + reason for an ABSTAIN."""
    if r.outcome is None:
        return "-", ""
    if r.outcome is Outcome.PROPOSE:
        mark = "" if r.action_correct is None else ("  ok" if r.action_correct else "  X")
        return "PROPOSE", f"{r.proposed_action}{mark}"
    return "ABSTAIN", f"-> {r.escalation_target} ({r.escalation_reason})"


def format_report(rows: Sequence[EvalRow], summary: dict[str, Any]) -> str:
    """Render the per-incident table + the headline metrics block (design §10)."""
    lines: list[str] = []
    lines.append(f"{'id':<10} {'gold sev/type':<24} {'pred sev/type':<24} {'sv':>2} {'ty':>2}  "
                 f"{'outcome':<8} action / escalation")
    lines.append("-" * 104)
    for r in rows:
        gold = f"{r.incident.gold_severity.value}/{r.incident.gold_type.value}"
        if r.pred_severity is not None:
            pred = f"{r.pred_severity.value}/{r.pred_type.value}"
            sev = "ok" if r.severity_correct else "X"
            typ = "ok" if r.type_correct else "X"
        else:
            pred, sev, typ = "-", "-", "-"
        outcome, detail = _decision_cells(r)
        lines.append(f"{r.incident.id:<10} {gold:<24} {pred:<24} {sev:>2} {typ:>2}  "
                     f"{outcome:<8} {detail}")

    s = summary
    lines.append("")
    lines.append("=== metrics ===")
    lines.append(f"  severity accuracy     {_pct(s['severity_accuracy'])}   "
                 f"(in-scope; {s['n_in_scope']} incidents)")
    lines.append(f"  type accuracy         {_pct(s['type_accuracy'])}   (in-scope)")
    lines.append(f"  severity accuracy*    {_pct(s['severity_accuracy_all'])}   "
                 f"(*all {s['n_classified']} incidents, incl. out-of-scope)")
    lines.append(f"  type accuracy*        {_pct(s['type_accuracy_all'])}   (*all)")
    lines.append(f"  abstention rate       {_pct(s['abstention_rate'])}   "
                 f"({s['n_must_abstain']} must-abstain incidents correctly escalated)")
    fa = f"   ({', '.join(s['false_abstention_ids'])})" if s.get("false_abstention_ids") else ""
    lines.append(f"  false abstentions     {s['n_false_abstentions']:>5}   (answerable wrongly abstained){fa}")
    me = f"   ({', '.join(s['missed_escalation_ids'])})" if s.get("missed_escalation_ids") else ""
    lines.append(f"  missed escalations    {s['n_missed_escalations']:>5}   (must-abstain got PROPOSE — dangerous){me}")
    lines.append(f"  retrieval recall@1    {_pct(s['recall_at_1'])}   "
                 f"({s['n_recallable']} recallable; expected runbook retrieved)")
    lines.append(f"  retrieval recall@3    {_pct(s['recall_at_3'])}")
    lines.append(f"  retrieval recall@k    {_pct(s['recall_at_k'])}")
    lines.append(f"  retrieval MRR         {_mrr(s['retrieval_mrr'])}   "
                 f"(mean reciprocal rank of the expected runbook)")
    lines.append(f"  action correctness    {_pct(s['action_correctness'])}   "
                 f"({s['n_judged']} PROPOSE outcomes, key-match — deterministic)")
    if s.get("n_decided"):
        lines.append(f"  decisions             {s['n_proposed']:>5} PROPOSE / {s['n_abstained']} ABSTAIN"
                     f"   (of {s['n_decided']} decided)")
    if s.get("n_judge"):
        # §10 honesty: the semantic judge is noisy — a clearly-labeled note,
        # never the headline, with its own spend kept out of the pipeline ledger.
        dis = (f"   (disagrees on {', '.join(s['judge_disagreement_ids'])})"
               if s["judge_disagreement_ids"] else "   (no disagreements)")
        lines.append("")
        lines.append("=== LLM-judge note — noisy; not a headline (design §10) ===")
        lines.append(f"  judge action score    {_pct(s['judge_action_correctness'])}   "
                     f"({s['n_judge']} PROPOSE outcomes, semantic, {JUDGE_MODEL})")
        lines.append(f"  agrees w/ key-match   {_pct(s['judge_agreement'])}{dis}")
        lines.append(f"  judge spend           ${s['judge_cost']['total']:>6.4f}   "
                     f"(measurement overhead; excluded from the pipeline cost below)")
    if s["usage_by_model"]:
        lines.append("")
        lines.append("=== cost / latency ===")
        lines.extend(observe.format_cost_block(
            s["usage_by_model"], indent="  ", latencies=s["latencies"], n=s["n_total"]))
        for name in ("classify", "retrieve", "draft", "decide"):  # per-stage p50/p95 (§7)
            vals = s.get("stage_latencies", {}).get(name)
            if vals:
                lines.append(f"  {'· ' + name:<18} p50={observe.percentile(vals, 50):.2f}s  "
                             f"p95={observe.percentile(vals, 95):.2f}s  (per stage, n={len(vals)})")
    return "\n".join(lines)


# --- the real claude-haiku-4-5 judge ------------------------------------

JUDGE_SYSTEM = (
    "You are grading one proposed first-response action from an incident-triage system.\n"
    "You are given the incident text, the ACTION the system proposed, and the REFERENCE "
    "first-response action an SRE authored for this incident.\n"
    "Rules:\n"
    "- Judge correct=true when the proposed action addresses THIS incident and would have "
    "the same safe first-response effect as the reference. The same action key as the "
    "reference is normally correct; a different key is correct only if it clearly "
    "accomplishes the same first response.\n"
    "- Judge correct=false when the proposed action does not actually address this incident, "
    "stretches a loosely-related procedure to fit it, or acts where the reference hands the "
    "incident to a human (ESCALATE).\n"
    "- Judge only from the given text. Do not invent details.\n"
    "- `reason` is one short sentence."
)

# Structured-output schema for the verdict. Booleans need no client-side clamping;
# `reason` length is prompt-bounded (json_schema has no string-length constraint).
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}


def build_judge_user(view: dict[str, str], proposed_action: str, gold_action: str) -> str:
    """The incident prompt view + the two action keys — nothing else reaches the judge."""
    return (f"Incident {view['id']} (source: {view['source']})\n"
            f"Title: {view['title']}\n\n{view['body']}\n\n"
            f"Proposed action: {proposed_action}\n"
            f"Reference action: {gold_action}")


def parse_action_verdict(text: str) -> ActionVerdict:
    """Parse the judge's JSON verdict (strict — structured outputs guarantee shape)."""
    data = json.loads(text)
    return ActionVerdict(correct=bool(data["correct"]), reason=data.get("reason", ""))


class ClaudeJudge:
    """Action-correctness judge on claude-haiku-4-5 via the official SDK with
    structured outputs (design §10). The cheap model is deliberate: the judge is
    measurement overhead run once per PROPOSE, and its verdict is a note, not a
    headline, so it earns no Opus. Reference-guided: the gold action key anchors
    the comparison, which keeps the noisy part (semantic equivalence) small. The
    lazy `anthropic` import + key check mirror `ClaudeClassifier`, so the offline
    suite needs neither the SDK nor a key.
    """

    def __init__(self, model: str = JUDGE_MODEL, max_tokens: int = 512) -> None:
        try:
            import anthropic  # lazy: only the live judge path needs it
        except ModuleNotFoundError as e:
            raise SystemExit(
                "the `anthropic` SDK is not installed (it ships as the optional 'live' extra).\n"
                "  Run:  uv run --with anthropic --with sqlite-vec python -m triage eval --judge\n"
                "  or:   pip install -e '.[live]'"
            ) from e

        if not (os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit(
                "ANTHROPIC_API_KEY not set.\n"
                "  Put  ANTHROPIC_API_KEY=...  in incident-triage-agent/.env  (gitignored), "
                "or export it."
            )
        self.model = model
        self.max_tokens = max_tokens  # a boolean + one sentence; 512 is ample headroom
        self._client = anthropic.Anthropic()

    def judge_action(self, view: dict[str, str], proposed_action: str,
                     gold_action: str) -> ActionVerdict:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user",
                       "content": build_judge_user(view, proposed_action, gold_action)}],
            output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        result = parse_action_verdict(text)
        result.usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        return result
