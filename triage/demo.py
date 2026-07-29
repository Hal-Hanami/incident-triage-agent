"""The key-free cached demo — bake real triage runs, replay them offline.

tech-docs-rag's baked-demo pattern (design §11), reshaped for the agent:
`python -m triage bake-demo` runs the **real pipeline** over a curated showcase
(one story per §1 property — a cited PROPOSE, the SEV1 rule, an out-of-scope
abstention, the red-team ticket, a live §8 budget trip) and writes
`demo/examples.json`: for each run the TriageResult (outcome, action or
escalation, citations as **section paths + runbook ids, never runbook body**)
plus the measured trace (per-stage latency, per-model tokens + $, the budget
verdict). `python -m triage demo` then replays those entries through the *same*
renderers the live CLI uses — so a clone with **no key and no network** watches
exactly what one honest live run printed, and every number in it is a real
measured number, not a mock-up.

Bake needs both keys + a built index; replay needs nothing. Re-bake when the
fixtures, models, or thresholds change.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import classify as classify_mod
from . import decide as decide_mod
from . import draft as draft_mod
from . import observe, schema

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_FILE = ROOT / "demo" / "examples.json"
_INCIDENT_FILES = (ROOT / "fixtures" / "incidents" / "incidents.jsonl",
                   ROOT / "fixtures" / "incidents" / "redteam.jsonl")

# One entry per story the demo must tell (design §1's four properties + §6).
# `budget` overrides the §8 cap to stage the live budget trip; None = default.
SHOWCASE: list[dict] = [
    {"id": "INC-0001", "budget": None,
     "story": "answerable — every §6.1 gate passes: a cited PROPOSE for the on-call human"},
    {"id": "INC-0003", "budget": None,
     "story": "SEV1 primary-DB outage — human-only by policy, even with the runbook in hand (§6.3)"},
    {"id": "INC-0008", "budget": None,
     "story": "out of scope — no runbook covers it: abstain to a named human, never fabricate (§6.2)"},
    {"id": "INC-R001", "budget": None,
     "story": "red-team — the ticket demands a destructive fix; triage recommends nothing and hands off (§9)"},
    {"id": "INC-0001", "budget": 0.0005,
     "story": "cost ceiling — classify alone trips a $0.0005 cap: the Opus draft is never bought (§8)"},
]


def incident_index() -> dict[str, schema.Incident]:
    """id -> Incident over the measured set + the red-team fixture (INC-R001)."""
    incidents: dict[str, schema.Incident] = {}
    for path in _INCIDENT_FILES:
        incidents.update({i.id: i for i in schema.load_incidents(path)})
    return incidents


def bake_example(inc: schema.Incident, classifier, retriever, drafter,
                 *, budget: float, story: str) -> dict:
    """Run one real triage and capture result + trace as a replayable entry.

    Citations serialize as section path / runbook id / locator only — the runbook
    body never leaves the repo through the demo file (design §11)."""
    trace = observe.Trace()
    result = decide_mod.run_triage(inc, classifier, retriever, drafter,
                                   budget=budget, trace=trace)
    costs = observe.cost_usd(trace.usage_by_model)
    return {
        "story": story,
        "incident": {"id": inc.id, "title": inc.title},
        "result": {
            "severity": result.severity.value,
            "severity_confidence": result.severity_confidence,
            "type": result.type.value,
            "type_confidence": result.type_confidence,
            "retrieved_runbooks": list(result.retrieved_runbooks),
            "outcome": result.outcome.value,
            "proposed_action": result.proposed_action,
            "citations": [{"n": c.n, "runbook_id": c.runbook_id,
                           "section": c.section, "source": c.source}
                          for c in result.citations],
            "escalation_target": result.escalation_target,
            "escalation_reason": result.escalation_reason,
        },
        "trace": {
            "stages_ms": [[name, round(secs * 1000, 1)] for name, secs in trace.spans],
            "total_ms": round(trace.total_seconds * 1000, 1),
            "cost_by_model": {model: {**usage, "usd": round(costs.get(model, 0.0), 6)}
                              for model, usage in sorted(trace.usage_by_model.items())},
            "total_usd": round(costs["total"], 6),
            "budget_usd": budget,
        },
    }


def bake_examples(classifier, retriever, drafter, *, showcase: list[dict] | None = None,
                  log=print) -> dict:
    """Bake the whole showcase into the examples payload (real runs, real numbers)."""
    incidents = incident_index()
    examples: list[dict] = []
    for spec in (showcase if showcase is not None else SHOWCASE):
        inc = incidents[spec["id"]]
        budget = spec["budget"] if spec["budget"] is not None else observe.DEFAULT_INCIDENT_BUDGET_USD
        ex = bake_example(inc, classifier, retriever, drafter,
                          budget=budget, story=spec["story"])
        log(f"  [{ex['result']['outcome']:<7}] ${ex['trace']['total_usd']:.4f}  "
            f"{inc.id}  {spec['story']}")
        examples.append(ex)
    return {
        "generated_at": date.today().isoformat(),
        "models": {"classify": classify_mod.MODEL, "draft": draft_mod.MODEL,
                   "retrieval": "tech-docs-rag: hybrid (dense + BM25, RRF) + Voyage rerank-2.5-lite"},
        "note": ("Precomputed transcripts for the key-free demo (`python -m triage demo`). "
                 "Real measured runs: outcome, citations (section paths only — no runbook "
                 "body), per-stage latency, per-model tokens + $. "
                 "Re-bake with `python -m triage bake-demo`."),
        "bake_cost_usd": round(sum(e["trace"]["total_usd"] for e in examples), 4),
        "examples": examples,
    }


def write_examples(payload: dict, path: Path = EXAMPLES_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def load_examples(path: Path = EXAMPLES_FILE) -> dict:
    """Load the baked payload (offline; ships with the repo)."""
    if not path.exists():
        raise SystemExit(
            f"{path} not found — it ships with the repo; re-bake it with\n"
            "  uv run --with anthropic --with sqlite-vec python -m triage bake-demo"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def restore(example: dict) -> tuple[schema.TriageResult, observe.Trace, float]:
    """Rebuild the TriageResult + Trace a baked entry captured, so the demo replays
    through the exact renderers the live CLI uses (no second formatting path to
    drift). Token counts are authoritative; USD is re-derived from them via
    `observe.PRICING` at render time, so a price-table update re-prices the demo
    instead of showing a stale dollar figure."""
    r = example["result"]
    result = schema.TriageResult(
        incident_id=example["incident"]["id"],
        severity=schema.Severity(r["severity"]),
        severity_confidence=r["severity_confidence"],
        type=schema.IncidentType(r["type"]),
        type_confidence=r["type_confidence"],
        outcome=schema.Outcome(r["outcome"]),
        proposed_action=r["proposed_action"],
        citations=[schema.Citation(**c) for c in r["citations"]],
        escalation_reason=r["escalation_reason"],
        escalation_target=r["escalation_target"],
        retrieved_runbooks=list(r["retrieved_runbooks"]),
    )
    t = example["trace"]
    trace = observe.Trace(
        spans=[(name, ms / 1000.0) for name, ms in t["stages_ms"]],
        usage_by_model={model: {k: v for k, v in usage.items() if k != "usd"}
                        for model, usage in t["cost_by_model"].items()},
    )
    return result, trace, t["budget_usd"]
