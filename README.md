# incident-triage-agent

**Read-only first-pass incident triage, on the Claude Agent SDK + MCP.**
Give it an alert or incident ticket → it classifies **severity + type**, retrieves
the matching **runbook**, and proposes a **cited first-response recommendation** —
or, when confidence is low or the situation is out of scope, it **abstains and
escalates to a human**. It never executes remediation.

[![tests](https://github.com/Hal-Hanami/incident-triage-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Hal-Hanami/incident-triage-agent/actions/workflows/ci.yml)

An agent that can act is only as trustworthy as its willingness *not* to. So the
thing this repository measures is not how often the agent is right — it is how
reliably it declines. On a frozen set of 32 synthetic incidents, over **four**
full-pipeline runs, reported as ranges because a single run would hide how much
these numbers move:

| | measured (4 runs) | what it means |
|---|---|---|
| abstention rate | **100%** (15/15 must-abstain), every run | it never proposed an action on an incident that needed a human |
| **missed escalations** | **0**, every run | the dangerous error has not occurred in any measured run |
| false abstentions | **1–2 of 17** answerable | it *over*-escalates — see below |
| action correctness | **93.3–93.8%** (key-match) | the LLM judge agreed with it in every case, both times it ran |
| cost | **$0.0135–$0.0142** per incident | the Opus drafter is ≈93% of it |
| latency | **p50 5.2–6.8s / p95 8.2–15.2s** | draft is the lever |

**The two headline numbers are the only ones that did not move.** Abstention and
missed escalations were identical in all four runs; everything else has a spread,
and severity classification has a 13-point one. Ranges are honest here in a way
a single column would not be.

**The false abstentions are the honest part.** One or two answerable incidents
per run were escalated that did not need to be — LLM variance in the drafter, not
a fixed property. Both errors are in the conservative direction, which is the
direction to be wrong in for this job, but the design target was 0 and no run met
it. [`docs/EVALUATION.md`](docs/EVALUATION.md) has every run, its date, its
reproduce command, and what it fails to show.

Runbook search is **not reimplemented here**: it reuses the retrieval package from
[tech-docs-rag](https://github.com/Hal-Hanami/tech-docs-rag), a source-grounded
RAG that declines to answer what its sources don't support. This project carries
that posture one rung up the autonomy ladder — from *don't answer* to **don't
act** — where the cost of being wrong is an action rather than a sentence.

- **Measured** — severity/type accuracy, retrieval recall, and **abstention rate /
  false-abstention / missed-escalation** on a versioned synthetic incident set.
- **Guardrailed** — **read-only by construction**: the agent is given no tool that
  can mutate a system; its only side effect is escalating to a person.
- **Cost-capped** — a hard **per-incident USD budget** that trips an abstain rather
  than overspending.
- **Observable** — **per-stage latency + per-model \$** on every triage, p50/p95
  across the set.

## How it works

```
incident → classify → retrieve → draft → decide → TriageResult (+ trace)
           sev+type   runbook    cited    PROPOSE | ABSTAIN→escalate
           haiku-4-5  (reused)   opus-4-8  (deterministic rule)
```

The four stages are a measurable pipeline; the **Claude Agent SDK** wraps them as
**in-process MCP tools** and orchestrates them under a read-only permission policy.
The safety-critical join — PROPOSE or ABSTAIN — runs **no model at all**: it is a
deterministic rule, so the decision cannot drift run to run. Full spec, including
the PROPOSE/ABSTAIN contract and the SEV1 rule:
[`docs/design.md`](docs/design.md).

## Quickstart (offline — no key needed)

```sh
# list the synthetic incident set (22 in-scope + 10 abstention tests)
python -m triage incidents
python -m triage incidents --scope out   # just the must-abstain cases

# replay five real measured triage runs — no key, no network
python -m triage demo

# check fixture integrity
python -m triage validate

# the offline test suite — 178 tests: pricing math (incl. cache tokens), schema
# invariants, fixtures, the full PROPOSE/ABSTAIN decision table, the budget/skip
# wiring, the guardrail policy *and the callbacks that enforce it*, the retrieval
# adapter's import contract, and the pipeline eval loop with fakes.
# 173 of those run anywhere; the other 5 check the stand-in rag package against
# the real one and skip unless a tech-docs-rag checkout sits beside this repo.
uv run --with pytest python -m pytest -q       # or: pip install -e '.[dev]' && pytest -q
```

## Measure (needs keys)

```sh
# classify: severity/type accuracy + per-model cost + p50/p95 (claude-haiku-4-5)
export ANTHROPIC_API_KEY=...                     # or put it in .env (gitignored)
uv run --with anthropic python -m triage eval --classify-only   # --limit N bounds spend

# retrieve: build the runbook index, then measure recall@1/@3/@k + MRR
export VOYAGE_API_KEY=...                         # shares .env with the Anthropic key
uv run --with sqlite-vec python -m triage index-runbooks
uv run --with sqlite-vec python -m triage eval --retrieval-only  # --no-rerank ablates the rerank

# the full pipeline (both keys + a built index): classify → retrieve → draft
# (claude-opus-4-8) → decide, scoring abstention / false-abstention / missed-escalation
uv run --with anthropic --with sqlite-vec python -m triage eval   # --no-draft = no Opus spend

# one incident end to end, with the per-stage latency + per-model $ trace + §8 budget verdict
uv run --with anthropic --with sqlite-vec python -m triage triage INC-0003

# the §8 cost ceiling, live: a tiny budget makes classify trip the cap, the Opus
# draft is never bought, and the decision degrades to ABSTAIN(cost_budget_exceeded)
uv run --with anthropic --with sqlite-vec python -m triage triage INC-0001 --budget 0.0005

# the same incident through the Agent SDK shell: read-only MCP tools, the
# per-run tool/deny audit, and the orchestrator cost cross-check (SDK vs token-priced)
uv run --with claude-agent-sdk --with anthropic --with sqlite-vec python -m triage agent INC-0001
```

Retrieval reuses the sibling tech-docs-rag checkout for `import rag` (default
`../tech-docs-rag`, override with `TECH_DOCS_RAG_PATH`).

## Layout

```
docs/design.md        the spec (numbered §, cited from code)
docs/EVALUATION.md    every measured number: date, set, reproduce command, limits
triage/schema.py      Incident / TriageResult / Severity / IncidentType / Outcome
triage/classify.py    classify stage — Classifier Protocol + claude-haiku-4-5
triage/runbooks.py    runbook corpus adapter — chunks fixtures/runbooks → rag index rows
triage/retrieve.py    retrieve stage — Retriever Protocol + RagRetriever (reuses tech-docs-rag)
triage/draft.py       draft stage — Drafter Protocol + claude-opus-4-8 cited recommendation
triage/decide.py      decide stage — deterministic PROPOSE/ABSTAIN rule; no model
triage/agent.py       Agent SDK shell — stages as in-process MCP tools, read-only guardrails
triage/eval.py        eval harness — accuracy + recall@1/@3/@k + MRR + abstention / missed-escalation + per-stage p50/p95
triage/observe.py     sourced per-model pricing (cache-aware), Trace, the §8 cost ceiling
triage/__main__.py    CLI — incidents / validate / index-runbooks / eval / triage <INC-ID> / agent <INC-ID> / demo
fixtures/incidents/   synthetic incident set (JSONL) — no real/secret data
fixtures/runbooks/    synthetic runbooks the RAG indexes
tests/                offline test suite (no key, no network)
```

## Non-goals

Not an auto-remediation system (it recommends + escalates, never executes). No real
incidents, telemetry, or secrets — everything is synthetic, so it is solo-buildable
and safe to make public. See [`docs/design.md`](docs/design.md) §11.

## Contributing

Commit conventions and the scope list are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Licence: [MIT](LICENSE).
