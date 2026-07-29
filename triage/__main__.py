"""CLI for the triage pipeline.

    python -m triage incidents [--scope in|out|all]   # list the synthetic incident set
    python -m triage validate                          # check fixture integrity (offline)
    python -m triage index-runbooks                    # build the runbook search index (reuses tech-docs-rag)
    python -m triage eval [--scope ...] [--limit N] [-k K] [--classify-only|--retrieval-only|--no-draft] [--no-rerank] [--budget $] [--max-cost $] [--judge]
    python -m triage triage <INC-ID> [-k K] [--no-rerank] [--budget $]   # run one full triage end to end
    python -m triage agent <INC-ID> [-k K] [--no-rerank] [--budget $]    # same triage via the Agent SDK shell
    python -m triage demo [INC-ID]                     # replay the baked key-free demo (offline)
    python -m triage bake-demo                         # re-bake demo/examples.json from live runs

`incidents`, `validate`, and `demo` are offline and need no key — `demo` replays
`demo/examples.json`, real measured transcripts baked by `bake-demo` (which needs
both keys + a built index), through the same renderers the live commands use.
`eval --judge` additionally grades each PROPOSE with a claude-haiku-4-5
LLM judge; the verdict renders as a clearly-labeled note, never a headline
(design §10). `index-runbooks` chunks the
runbook corpus and builds the `rag` vector+BM25 index over it (needs
VOYAGE_API_KEY; run via `uv run --with sqlite-vec`). `eval` measures the pipeline:
classify (claude-haiku-4-5 → severity/type accuracy), retrieve (tech-docs-rag rag →
recall@1/@3/@k + MRR), and — by default — draft (claude-opus-4-8) + decide (§6) →
abstention / false-abstention / missed-escalation + key-match action correctness,
plus per-model cost and p50/p95 latency. `--classify-only` needs no index;
`--retrieval-only` measures recall with no Anthropic spend; `--no-draft` runs only
classify+retrieve only; default runs the full pipeline (needs ANTHROPIC_API_KEY
+ VOYAGE_API_KEY + a built index; `uv run --with anthropic --with sqlite-vec`). `triage
<INC-ID>` runs the full classify->retrieve->draft->decide pipeline for one incident and
prints the TriageResult + the per-stage observability footer (§7). `agent <INC-ID>` runs
the same incident through the Agent SDK shell (design §2/§9): claude-opus-4-8
orchestrates the stages as in-process MCP tools under the read-only permission policy,
and the footer additionally reports every tool call, every guard denial, and the
orchestrator's own cost — SDK-reported and cross-checked against our own token-priced
estimate (§7) — (`uv run --with claude-agent-sdk --with anthropic --with
sqlite-vec`; also needs the Claude Code CLI). Both single-incident commands can also
address the red-team fixture (INC-R001, fixtures/incidents/redteam.jsonl), which stays
out of the measured eval set.

Cost ceiling (§8): `--budget` sets the per-incident USD cap (default $0.05).
Crossing it mid-triage skips the remaining paid stages (notably the Opus draft) and the
decision degrades to ABSTAIN(cost_budget_exceeded) — demonstrable live with e.g.
`triage INC-0001 --budget 0.0005`. `eval` additionally enforces `--max-cost`, an
aggregate cap for the whole run (default: budget × selected incidents) — once crossed,
no further incident is run. Measurements: docs/EVALUATION.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import observe, schema

ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_FILE = ROOT / "fixtures" / "incidents" / "incidents.jsonl"
REDTEAM_FILE = ROOT / "fixtures" / "incidents" / "redteam.jsonl"


def _incident_index() -> dict[str, schema.Incident]:
    """id -> Incident across the measured set AND the red-team fixture, so the
    single-incident commands (`triage` / `agent`) can address INC-R001 while the
    eval set itself stays the frozen 32 (docs/EVALUATION.md)."""
    incidents = {i.id: i for i in schema.load_incidents(INCIDENTS_FILE)}
    incidents.update({i.id: i for i in schema.load_incidents(REDTEAM_FILE)})
    return incidents


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE pairs from a local .env (gitignored) into the environment if
    present, without overriding vars already set — just enough to make the documented
    `.env` key path work for `eval`. No third-party dependency."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cmd_incidents(args: argparse.Namespace) -> None:
    incidents = schema.load_incidents(INCIDENTS_FILE)
    if args.scope == "in":
        incidents = [i for i in incidents if i.in_scope]
    elif args.scope == "out":
        incidents = [i for i in incidents if not i.in_scope]
    print(f"{len(incidents)} incidents (scope={args.scope})\n")
    for inc in incidents:
        flag = "abstain" if inc.must_abstain else "answer "
        tags = f"  [{', '.join(inc.tags)}]" if inc.tags else ""
        print(f"  {inc.id}  {inc.gold_severity.value}  {inc.gold_type.value:<18} {flag}  {inc.title}{tags}")


def cmd_validate(args: argparse.Namespace) -> None:
    incidents = schema.load_incidents(INCIDENTS_FILE)
    problems = schema.validate(incidents)
    if problems:
        print(f"INVALID — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    n_out = sum(1 for i in incidents if not i.in_scope)
    print(f"OK — {len(incidents)} incidents valid "
          f"({len(incidents) - n_out} in-scope, {n_out} abstention tests)")


def _format_triage(title: str, result: schema.TriageResult) -> str:
    """Render one TriageResult — the predicted labels, the outcome, and either the
    cited proposed action or the named human escalation (design §6). Takes the
    title (not the Incident) so the baked demo can replay through this exact
    renderer."""
    lines = [f"=== triage {result.incident_id}: {title} ==="]
    lines.append(f"  severity    {result.severity.value}  (confidence {result.severity_confidence:.2f})")
    lines.append(f"  type        {result.type.value}  (confidence {result.type_confidence:.2f})")
    lines.append(f"  retrieved   {', '.join(result.retrieved_runbooks) or '(none)'}")
    lines.append(f"  OUTCOME     {result.outcome.value}")
    if result.outcome is schema.Outcome.PROPOSE:
        lines.append(f"  action      {result.proposed_action}  (for the on-call human to apply)")
        lines.append("  citations:")
        for c in result.citations:
            lines.append(f"    [{c.n}] {c.section}  ({c.source})")
    else:
        lines.append(f"  escalate -> {result.escalation_target}  (reason: {result.escalation_reason})")
    return "\n".join(lines)


def cmd_triage(args: argparse.Namespace) -> None:
    from . import classify as classify_mod
    from . import decide as decide_mod
    from . import draft as draft_mod
    from . import retrieve as retrieve_mod

    inc = _incident_index().get(args.incident_id)
    if inc is None:
        print(f"unknown incident {args.incident_id!r} — try `python -m triage incidents`")
        sys.exit(1)
    # Each constructor raises SystemExit with a fix-it hint if a key/index is missing.
    classifier = classify_mod.ClaudeClassifier()
    retriever = retrieve_mod.RagRetriever(k=args.k, rerank=not args.no_rerank)
    drafter = draft_mod.ClaudeDrafter()
    trace = observe.Trace()
    result = decide_mod.run_triage(inc, classifier, retriever, drafter,
                                   budget=args.budget, trace=trace)
    print(_format_triage(inc.title, result))
    print()
    print("\n".join(observe.format_trace(trace, budget=args.budget)))


def cmd_agent(args: argparse.Namespace) -> None:
    from . import agent as agent_mod
    from . import classify as classify_mod
    from . import draft as draft_mod
    from . import retrieve as retrieve_mod

    inc = _incident_index().get(args.incident_id)
    if inc is None:
        print(f"unknown incident {args.incident_id!r} — try `python -m triage incidents`")
        sys.exit(1)
    classifier = classify_mod.ClaudeClassifier()
    retriever = retrieve_mod.RagRetriever(k=args.k, rerank=not args.no_rerank)
    drafter = draft_mod.ClaudeDrafter()
    run = agent_mod.run_agent_triage(inc, classifier, retriever, drafter,
                                     budget=args.budget)
    print(_format_triage(inc.title, run.result))
    print()
    print("--- agent shell (design §2/§9) ---")
    print(f"  tool calls  {', '.join(run.tool_calls) or '(none)'}")
    print(f"  denied      {', '.join(run.denied_calls) or '(none)'}")
    offlist = run.non_triage_tool_calls
    print(f"  red-team    non-allowlisted calls: {offlist or 'NONE (read-only held)'}")

    def usd(x: float | None) -> str:
        return f"${x:.4f}" if x is not None else "n/a"

    # §7: the orchestrator's spend, twice — the SDK's own accounting and our
    # token-priced estimate from ResultMessage per-model usage. They should agree.
    delta = run.cost_crosscheck_usd
    print(f"  orchestrator SDK={usd(run.total_cost_usd)}  "
          f"computed={usd(run.orchestrator_cost)}  "
          f"(delta {f'{delta:+.4f}' if delta is not None else 'n/a'}; §7 cross-check)  "
          f"turns={run.num_turns}  {run.duration_ms / 1000:.1f}s")
    # §8: the budget verdict over the whole incident — stages + orchestrator.
    stages = run.session.trace.cost()
    orch = run.total_cost_usd if run.total_cost_usd is not None else (run.orchestrator_cost or 0.0)
    total = stages + orch
    state = "EXCEEDED" if total > run.session.budget else "within"
    print(f"  incident $  {total:.4f} = stages {stages:.4f} + orchestrator {orch:.4f}  "
          f"(budget ${run.session.budget:.4f} — {state}, design §8)")
    print()
    print("\n".join(observe.format_trace(run.session.trace)))


def cmd_index_runbooks(args: argparse.Namespace) -> None:
    from . import retrieve as retrieve_mod

    print("chunking runbooks + embedding with Voyage (input_type=document) ...")
    stats = retrieve_mod.index_runbooks()
    print(f"\nindexed {stats['count']} chunks from {stats['runbooks']} runbooks  "
          f"dim={stats['dim']}  model={stats['model']}  embed={stats['embed_secs']}s\n"
          f"  -> {stats['db_path']}")


def cmd_eval(args: argparse.Namespace) -> None:
    from . import classify as classify_mod
    from . import draft as draft_mod
    from . import eval as eval_mod
    from . import retrieve as retrieve_mod

    incidents = schema.load_incidents(INCIDENTS_FILE)
    # classify + draft need ANTHROPIC_API_KEY; retrieve needs VOYAGE_API_KEY + a built
    # index. Each constructor raises SystemExit with a fix-it hint if its prerequisite
    # is missing. draft (and thus decide, which needs all three stages) runs by default
    # and is skipped by any of the stage flags.
    classifier = None if args.retrieval_only else classify_mod.ClaudeClassifier()
    retriever = None if args.classify_only else retrieve_mod.RagRetriever(
        k=args.k, rerank=not args.no_rerank)
    drafter = None
    if not (args.classify_only or args.retrieval_only or args.no_draft):
        drafter = draft_mod.ClaudeDrafter()
    judge = None
    if args.judge:
        if drafter is None:
            raise SystemExit("--judge grades PROPOSE outcomes, so it needs the full "
                             "pipeline — drop --classify-only/--retrieval-only/--no-draft.")
        judge = eval_mod.ClaudeJudge()
    rows = eval_mod.evaluate(incidents, classifier, retriever, drafter, judge=judge,
                             budget=args.budget, max_total_usd=args.max_cost,
                             scope=args.scope, limit=args.limit)
    print(eval_mod.format_report(rows, eval_mod.summarize(rows)))
    n_selected = len(eval_mod.select(incidents, scope=args.scope, limit=args.limit))
    if len(rows) < n_selected:  # §8 aggregate fail-safe fired
        cap = args.max_cost if args.max_cost is not None else args.budget * n_selected
        print(f"\n!! aggregate cost cap ${cap:.4f} tripped — stopped after "
              f"{len(rows)} of {n_selected} incidents (design §8 fail-safe)")


def cmd_demo(args: argparse.Namespace) -> None:
    from . import demo as demo_mod

    payload = demo_mod.load_examples()
    examples = payload["examples"]
    if args.incident_id:
        examples = [e for e in examples if e["incident"]["id"] == args.incident_id]
        if not examples:
            baked = ", ".join(sorted({e["incident"]["id"] for e in payload["examples"]}))
            print(f"no baked entry for {args.incident_id!r} — baked incidents: {baked}")
            sys.exit(1)
    print(f"key-free cached demo — {len(examples)} real measured transcript(s), "
          f"baked {payload['generated_at']} for ${payload['bake_cost_usd']:.4f} "
          f"(re-bake: python -m triage bake-demo)")
    for ex in examples:
        result, trace, budget = demo_mod.restore(ex)
        print()
        print(f"--- {ex['story']} ---")
        # The same renderers the live `triage` command uses — replay, not re-format.
        print(_format_triage(ex["incident"]["title"], result))
        print()
        print("\n".join(observe.format_trace(trace, budget=budget)))


def cmd_bake_demo(args: argparse.Namespace) -> None:
    from . import classify as classify_mod
    from . import demo as demo_mod
    from . import draft as draft_mod
    from . import retrieve as retrieve_mod

    classifier = classify_mod.ClaudeClassifier()
    retriever = retrieve_mod.RagRetriever()
    drafter = draft_mod.ClaudeDrafter()
    print(f"baking {len(demo_mod.SHOWCASE)} showcase runs (real pipeline) ...\n")
    payload = demo_mod.bake_examples(classifier, retriever, drafter)
    out = demo_mod.write_examples(payload)
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(payload['examples'])} entries, "
          f"bake cost ${payload['bake_cost_usd']:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inc = sub.add_parser("incidents", help="list the synthetic incident set")
    p_inc.add_argument("--scope", choices=["in", "out", "all"], default="all",
                       help="in-scope (answerable), out-of-scope (abstain), or all")
    p_inc.set_defaults(func=cmd_incidents)

    p_val = sub.add_parser("validate", help="check fixture integrity (offline)")
    p_val.set_defaults(func=cmd_validate)

    p_tri = sub.add_parser(
        "triage",
        help="run one full triage: classify -> retrieve -> draft -> decide (needs both keys + a built index)")
    p_tri.add_argument("incident_id", help="e.g. INC-0001")
    p_tri.add_argument("-k", type=int, default=5, help="top-k chunks for retrieval (default 5)")
    p_tri.add_argument("--no-rerank", action="store_true",
                       help="skip the Voyage reranker for the retrieve stage")
    p_tri.add_argument("--budget", type=float, default=observe.DEFAULT_INCIDENT_BUDGET_USD,
                       help="per-incident USD cap (design §8); crossing it skips the "
                            "remaining paid stages and abstains (default $%(default)s)")
    p_tri.set_defaults(func=cmd_triage)

    p_agent = sub.add_parser(
        "agent",
        help="run one triage through the Agent SDK shell — in-process MCP tools under "
             "the read-only policy (needs claude-agent-sdk + both keys + index)")
    p_agent.add_argument("incident_id", help="e.g. INC-0001, or the red-team INC-R001")
    p_agent.add_argument("-k", type=int, default=5, help="top-k chunks for retrieval (default 5)")
    p_agent.add_argument("--no-rerank", action="store_true",
                         help="skip the Voyage reranker for the retrieve stage")
    p_agent.add_argument("--budget", type=float, default=observe.DEFAULT_INCIDENT_BUDGET_USD,
                         help="per-incident USD cap (design §8) over stages + orchestrator "
                              "(default $%(default)s)")
    p_agent.set_defaults(func=cmd_agent)

    p_idx = sub.add_parser("index-runbooks",
                           help="build the runbook search index (reuses the tech-docs-rag retrieval package; needs VOYAGE_API_KEY)")
    p_idx.set_defaults(func=cmd_index_runbooks)

    p_eval = sub.add_parser("eval",
                            help="measure the pipeline: accuracy + recall + abstention/escalation + cost/latency")
    p_eval.add_argument("--scope", choices=["in", "out", "all"], default="all",
                        help="score in-scope, out-of-scope, or all incidents")
    p_eval.add_argument("--limit", type=int, default=0, help="cap incidents (bound live API spend)")
    p_eval.add_argument("-k", type=int, default=5, help="top-k chunks for retrieval (default 5)")
    p_eval.add_argument("--no-rerank", action="store_true",
                        help="skip the Voyage reranker (measure retrieval without rerank)")
    stage = p_eval.add_mutually_exclusive_group()
    stage.add_argument("--classify-only", action="store_true",
                       help="classify stage only — ANTHROPIC_API_KEY, no runbook index needed")
    stage.add_argument("--retrieval-only", action="store_true",
                       help="retrieve stage only — VOYAGE_API_KEY + a built index, no Anthropic spend")
    stage.add_argument("--no-draft", action="store_true",
                       help="classify + retrieve without draft/decide — no Opus spend")
    p_eval.add_argument("--budget", type=float, default=observe.DEFAULT_INCIDENT_BUDGET_USD,
                        help="per-incident USD cap (design §8; default $%(default)s)")
    p_eval.add_argument("--max-cost", type=float, default=None,
                        help="aggregate USD cap for the whole run (design §8 fail-safe; "
                             "default: budget x selected incidents)")
    p_eval.add_argument("--judge", action="store_true",
                        help="grade each PROPOSE with the claude-haiku-4-5 LLM judge — "
                             "semantic second opinion, reported as a note (design §10)")
    p_eval.set_defaults(func=cmd_eval)

    p_demo = sub.add_parser(
        "demo",
        help="replay the baked key-free demo — real measured transcripts, no key, no network")
    p_demo.add_argument("incident_id", nargs="?", default=None,
                        help="replay only this incident's entries (default: the whole showcase)")
    p_demo.set_defaults(func=cmd_demo)

    p_bake = sub.add_parser(
        "bake-demo",
        help="re-bake demo/examples.json from real pipeline runs (needs both keys + a built index)")
    p_bake.set_defaults(func=cmd_bake_demo)

    _load_dotenv()  # make `ANTHROPIC_API_KEY=...` in a local .env work for `eval`
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
