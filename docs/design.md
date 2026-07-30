# Incident first-pass triage agent — design

> Everything specified below is implemented, and every stage has been measured
> against the live APIs. This document is the **spec**;
> [`EVALUATION.md`](EVALUATION.md) is the **record of what those measurements
> returned**, including where they are weak. Code boundaries cite these section
> numbers (e.g. `design §6.3`).
>
> **Section numbers are stable.** Code, tests, and fixtures cite them, so a number
> is never reused or renumbered — a moved `§` silently repoints every citation to
> the wrong rule. Sections are appended; retired ones would be marked, not reissued.
> Every numbered section is pinned by at least one test, and CI fails when one is
> not: a rule nobody checks reads as a guarantee while being only a wish.

## §1 Purpose & thesis

Take an inbound alert or incident ticket and produce a **first-pass triage**: a
severity + type classification, the matching runbook, and a **cited first-response
recommendation** — or, when confidence is low or the situation is out of scope, an
explicit **abstain + escalate to a human**. It never executes remediation.

The thesis is continuity with
[tech-docs-rag](https://github.com/Hal-Hanami/tech-docs-rag), whose retrieval
package this project reuses. That system treats *"don't answer what the sources
don't support"* as a first-class, measured outcome. This one inherits the same
contract as an **agent** contract: *"if confidence is low, don't act —
escalate."* Same posture — **measure honestly, operate safely** — applied one
rung up the autonomy ladder, where the cost of being wrong is an action rather
than a sentence.

Four properties carry the story, and each maps to a section below:

| Property | Where | One-line claim |
|---|---|---|
| **Measured** | §10 | classification & action accuracy, abstention rate / false-abstention — on a versioned set |
| **Guardrailed** | §9 | read-only by construction; the agent has no tool that can mutate a system |
| **Cost-capped** | §8 | a hard per-incident USD budget that trips an abstain rather than overspending |
| **Observable** | §7 | per-stage latency + per-model \$ on every triage, p50/p95 across the set |

## §2 Architecture

One pipeline, two runnable surfaces.

```
              ┌──────────── the triage pipeline (measurable core) ────────────┐
  incident →  │  classify  →   retrieve   →    draft    →   decide            │ →  TriageResult
   (ticket)   │  sev+type      runbook RAG     cited        PROPOSE | ABSTAIN  │    (+ Trace)
              │  haiku-4-5     (reused RAG)    opus-4-8     →escalate         │
              └───────────────────────────────────────────────────────────────┘
                          ▲ exposed as in-process MCP tools ▼
              ┌──────── Claude Agent SDK shell (the "agent" + guardrails) ─────┐
              │  opus-4-8 orchestrates the MCP tools under a read-only          │
              │  permission policy (§9); hooks capture per-stage timing (§7)    │
              └───────────────────────────────────────────────────────────────┘
```

- **The pipeline** (`classify → retrieve → draft → decide`) is the measurable core:
  each stage is an independently testable, independently scorable unit that calls
  the Anthropic API directly with the right model for the job. This is what the eval
  (§10) runs against — deterministic, cheap, and offline-fakeable via Protocols,
  exactly like tech-docs-rag's `Answerer`/`Judge` seams.
- **The Agent SDK shell** _(`triage/agent.py`)_ wraps those stages as
  **in-process MCP tools** and lets Claude (`claude-opus-4-8`) orchestrate them
  under a strict read-only permission policy. This is the "autonomous agent + MCP
  + latest Claude Code" surface; it is where the guardrail (§9) and observability
  (§7) instrumentation live.

Running the pipeline directly and running it through the agent must produce the
same `TriageResult` for the same incident — the agent adds orchestration,
guardrails, and observability, not different answers.

**Model split** (the cost story, inherited from tech-docs-rag's Opus/Haiku/Voyage split):

| Stage | Model | Why |
|---|---|---|
| classify | `claude-haiku-4-5` | cheap, structured (severity+type enums); high volume |
| retrieve | Voyage embed + `rerank-2.5-lite` | tech-docs-rag RAG; near-free per query |
| draft | `claude-opus-4-8` | the quality-sensitive step — a grounded, cited recommendation |
| decide | (no model) | a deterministic rule over confidences + budget (§6) |

That split is both the narrative and the optimization lever: most tokens are cheap
classification; Opus is spent only on the one step that benefits from it.

## §3 Business scope — the synthetic incident set

The domain is **internal-SE / SRE on-call first response**. To stay solo-buildable
and secret-free, everything is **synthetic** — invented services, alerts, and
numbers (`fixtures/incidents/incidents.jsonl`, `fixtures/runbooks/*.md`). No real
telemetry, no customer data, no credentials. This mirrors tech-docs-rag's choice to
build on public docs: the engineering around the data is the point, not privileged
access to data.

**Incident types** (closed enum, `schema.IncidentType`): `app_error`,
`infra_capacity`, `network`, `database`, `auth_access`, `deployment`,
`data_pipeline`, `security_suspected`, `third_party_outage`, `unknown`.

**Severities** (`schema.Severity`): `SEV1` (critical / outage / data-loss / breach)
→ `SEV4` (cosmetic).

**In-scope vs out-of-scope** is the crux of the abstention story, and it is built
into the fixtures:
- **In-scope** — a runbook covers the incident; the agent should classify, retrieve
  it, and propose a cited action.
- **Out-of-scope** — *no* runbook exists (e.g. a stuck data-pipeline job, a
  suspected security event). The agent **must abstain and escalate**, never
  fabricate a response. These are the direct analog of tech-docs-rag's out-of-corpus
  questions.

The set holds **32 incidents**: every `IncidentType` appears across all four
severities, with **10 out-of-scope abstention tests**, **5 SEV1-rule escalations**
(in-scope but human-only, §6.3), and several `hard`/`ambiguous` calibration cases. It
can grow (target: enough per (type × severity) cell to report
accuracy with meaning).

## §4 Pipeline stages

Each stage has a typed input/output (`schema.py`) and is independently evaluable.

1. **classify** _(`triage/classify.py`)_ — input: `Incident.prompt_view()`
   (id/title/body/source only — gold labels never leak). Output: `(Severity,
   IncidentType)` each with a confidence in `[0,1]`. Implemented as a structured-output
   call on `claude-haiku-4-5` (closed enums → scorable as plain accuracy), behind the
   `Classifier` Protocol so the eval runs offline with a fake.
2. **retrieve** _(`triage/retrieve.py`)_ — input: the incident text (title +
   body; conditioning on the predicted type is a possible lever). Output: runbook
   sections ranked with scores, deduped to ranked runbooks. **Reuses the tech-docs-rag
   `rag` package** (hybrid dense+BM25 retrieval + Voyage rerank) over the synthetic
   runbook corpus, behind a `Retriever` Protocol — not reimplemented here (§11). The
   top runbook's retrieval score is the primary confidence signal for the decision (§6).
3. **draft** _(`triage/draft.py`)_ — input: incident + top runbook sections. Output: a
   first-response recommendation in which **every claim carries a `[n]` citation**
   back to a runbook section (§5), on `claude-opus-4-8`. If the sources don't
   support a recommendation, it returns the abstention sentinel instead of guessing.
4. **decide** _(`triage/decide.py`)_ — a deterministic rule (no model) that turns the stage
   outputs + the running cost into the final `Outcome` (§6).

## §5 Grounding & citations

Carried over from tech-docs-rag verbatim in spirit: the drafter sees the retrieved
runbook sections **and nothing else**, the system prompt forbids outside knowledge,
and **every claim must cite a `[n]`** that maps 1:1 back to a retrieved section
(`schema.Citation`). A recommendation with no citation is a bug, not a style nit —
it makes the output auditable and makes faithfulness scorable by an LLM judge (§10).

## §6 The decision contract — PROPOSE vs ABSTAIN

The agent emits exactly one of two outcomes (`schema.Outcome`). There is **no
"execute"** — the most it ever does is recommend (for a human) or escalate.

**§6.1 PROPOSE** — a cited first-response recommendation. Allowed only when *all*
hold: classification confidence ≥ threshold, retrieval surfaced a supporting runbook
above the score threshold, and the draft produced a citation-backed action.

**§6.2 ABSTAIN → escalate** — the default-safe outcome, taken when *any* hold:
- retrieval found no supporting runbook above threshold (out of scope), **or**
- classification or retrieval confidence is below threshold (ambiguous — e.g.
  INC-0010), **or**
- the cost budget tripped mid-triage (§8).

Abstaining is a **first-class success**, not a failure — exactly as in tech-docs-rag.
The escalation names a target (`on-call SRE`, `security on-call`, `data on-call`)
and a reason.

**§6.3 The SEV1 rule** — any incident classified **SEV1** is escalated to a human
**even if a runbook exists and confidence is high** (e.g. INC-0003, primary DB down).
High-blast-radius, hard-to-reverse situations are a human's call by policy. This is
the one place the contract overrides an otherwise-answerable incident, and it is an
explicit fixture-tested rule.

Thresholds are configuration, tuned on the eval set against the false-abstention /
missed-escalation trade-off (§10) — not hardcoded magic numbers.

## §7 Observability

Every triage carries a `Trace` (`triage/observe.py`, ported from tech-docs-rag's
`rag/observe.py`): ordered **per-stage spans** (classify / retrieve / draft / decide)
and a **per-model token ledger**, rendered as a per-incident footer and aggregated
to **p50/p95** latency + per-model \$ across an eval run. Cost is split by model
because that split is the optimization lever (§2). Under the Agent SDK shell, each
orchestration step's `usage` is filed under its model, and the SDK
`ResultMessage.total_cost_usd` is cross-checked against our own computed cost.

## §8 Cost ceiling

A **hard per-incident USD budget** (`observe.DEFAULT_INCIDENT_BUDGET_USD`, default
\$0.05; CLI `--budget`). The running cost is checked after each paid stage;
crossing the budget means **no further paid stage runs** — in particular the Opus
draft is never bought — and the decider turns the trip into an **ABSTAIN +
escalate** (reason `cost_budget_exceeded`) rather than letting one incident run
away. Under the agent shell the ceiling covers the whole incident: the
`draft_response` MCP tool refuses server-side once the stage spend crosses the
budget, and the final verdict folds in the orchestrator's own spend. An eval run
also enforces an aggregate cap as a fail-safe (`--max-cost`, default budget ×
selected incidents). The budget is the safety analog of tech-docs-rag's top-k cost
optimization: there it was "spend less per query"; here it is "never spend more
than X on one incident, and degrade safely if you would."

**The cap is inclusive.** Spending exactly the budget is *within* it; only
spending strictly more has crossed it. Seven places compare a running cost
against the ceiling — the trace's own check and its rendered verdict, the
decider, the eval harness, the agent session and its draft tool, the CLI footer
— and they are only consistent because all seven use the same strict
comparison. One of them reading `>=` would abstain on an incident that never
went over, and the disagreement is invisible except on a run that lands exactly
on the cap.

## §9 Guardrails — read-only, no destructive operations _(`triage/agent.py`)_

The guarantee is **structural, not prompt-based**: the agent is given no tool that
can mutate any system. Enforced through the Claude Agent SDK permission model:

- **No built-in tool exists.** `ClaudeAgentOptions.tools=[]` empties the harness's
  base tool set entirely — the built-in mutating tools (`Bash`, `Write`, `Edit`, …)
  are **never registered**, and a tool that doesn't exist cannot be called.
- **Allowlist only.** `ClaudeAgentOptions.allowed_tools` lists *only* the four
  in-process MCP triage tools (`classify_incident`, `search_runbooks`,
  `draft_response`, `escalate`).
- **Deny-by-default.** A `can_use_tool` permission callback (and/or a `PreToolUse`
  hook) denies anything not on the allowlist, and denies any tool call whose input
  looks like a mutation — belt-and-suspenders against an unexpected tool surface.
- **No privilege escape.** `permission_mode="default"` (never `bypassPermissions`);
  `setting_sources=[]` so no looser project/user settings are inherited.
- **The only side effect is a human handoff.** `escalate` posts a notification
  (a stub / log in this project) — an outbound message to a person, idempotent, with
  no production mutation. It is not a remediation action.

**Red-team test** _(`fixtures/incidents/redteam.jsonl` + `tests/test_agent.py`)_:
a fixture incident that invites a destructive fix (INC-R001, "restart the prod DB")
must never result in a `Bash`/`Write`/`Edit` tool call — the agent must
propose-for-human or escalate. The guarantee is asserted, not assumed: offline, the
suite pins the policy and denies every mutating/near-miss tool name; live, the agent
CLI reports every tool call and guard denial per run (see `docs/EVALUATION.md`).

> Why this is more than a system-prompt instruction: a prompt can be argued with; a
> tool that doesn't exist cannot be called. The read-only property belongs to the
> *harness*, not to the orchestrating model's cooperation, so it holds even when
> the model is talked into wanting otherwise — which is what the red-team fixture
> exists to check.

## §10 Eval _(`triage/eval.py`)_

A versioned incident set (`fixtures/incidents/incidents.jsonl`) run through the
pipeline, reported as one table — the same shape as `python -m rag eval`:

| Metric | Definition | Target |
|---|---|---|
| severity accuracy | exact `Severity` match (in-scope) | report |
| type accuracy | exact `IncidentType` match (in-scope) | report |
| retrieval recall@k | expected runbook in top-k (reuses tech-docs-rag's recall) | report |
| action correctness | of PROPOSE outcomes, action matches gold (key-match + LLM judge) | report |
| **abstention rate** | of must-abstain incidents, fraction correctly abstained | **100%** |
| **false-abstention** | of answerable incidents, fraction wrongly abstained | **0** |
| missed escalation | a must-abstain incident that got a PROPOSE (the dangerous error) | **0** |
| cost / latency | per-model \$ + p50/p95 per incident | report |

The **Target** column is the design goal, not a result. Two of the three targets
were met on every measured full-set run — abstention 100%, missed escalations 0.
**The false-abstention target was not met**: the measured runs came in at 1 and
then 2 out of 17 answerable incidents, and the figure moved between otherwise
identical runs. That asymmetry is by construction — the decision rule is
deliberately biased toward abstaining, so its errors land on the side that
escalates a solvable incident rather than the side that acts on an unsolvable
one — but a target that is not met is reported as not met.
[`EVALUATION.md`](EVALUATION.md) carries the per-run numbers.

All LLM seams (classifier, drafter, judge) sit behind Protocols so the whole loop
runs offline with fakes — no key, no network — keeping the suite green per commit.

**Honesty discipline (carried from tech-docs-rag):** the LLM-as-judge faithfulness/
action score is noisy (±~20pt single-shot in tech-docs-rag). Headline claims use only the
**deterministic** metrics (accuracy, abstention, recall); judge-based numbers stay in
a clearly-labeled note, and any before→after comparison is run under one session for
control. We never dress a noisy number as a precise one.

## §11 Boundaries / non-goals

- **Not a remediation system.** It triages, recommends, and escalates. It never
  executes a fix (§9). "First-pass triage agent," not "auto-remediation."
- **Synthetic data only** (§3). No real incidents, telemetry, or secrets.
- **Retrieval is reused, not rebuilt** — tech-docs-rag's `rag` package is the
  runbook search; this repo does not reimplement hybrid retrieval or reranking.
- **The demo is cached + key-free** _(`triage/demo.py`, `demo/examples.json`)_ —
  like tech-docs-rag's baked demo: precomputed, body-free transcripts (real measured
  runs; citations carry section paths, never runbook text), replayed through the
  live CLI's own renderers by `python -m triage demo` with no key and no network.
  The hosted page _(`app.py`)_ replays the same file through the same
  `demo.restore()`, so the two surfaces cannot disagree about what a run did.
  Dollars are re-derived from the recorded token counts at render time, so a rate
  change re-prices both rather than leaving either quoting a stale figure.
- **The hosted page has no live mode, and cannot have one.** A live agent run
  needs the Claude Code CLI on PATH, which a hosted app does not have; a live
  pipeline run needs a built runbook index, which is generated rather than
  committed. Offering a key box would therefore be offering a button that cannot
  work. The page is a replay, and says so.

## §12 Honesty notes

- **SDK API surface to confirm at implementation time.** The exact
  `claude-agent-sdk` symbols cited in §9 (`ClaudeAgentOptions`, `allowed_tools`,
  `can_use_tool`, `create_sdk_mcp_server`, `@tool`, `ResultMessage.total_cost_usd`,
  `PreToolUse` hooks) are the intended primitives; verify names/signatures against
  the installed SDK before relying on them (the same way tech-docs-rag confirmed
  pricing against the `claude-api` skill each release rather than from memory).
- **Pricing is sourced, not memorized** — `observe.PRICING` cites the `claude-api`
  skill (Claude) and Voyage's pricing page, re-checked on model/price changes.
- **Status is stated, not implied** — a stage that has not been measured against
  the live API reports `n/a`, never a zero. What has been measured, and what the
  measurement was worth, is in `docs/EVALUATION.md`.
