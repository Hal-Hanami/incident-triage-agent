"""The Agent SDK shell — the pipeline as in-process MCP tools under a read-only
permission policy (design §2, §9).

The measurable pipeline (classify → retrieve → draft → decide) stays exactly as
already measurable on their own; this module wraps them as **in-process MCP tools**
(`create_sdk_mcp_server` + `@tool`, claude-agent-sdk) and lets `claude-opus-4-8`
orchestrate them. Every tool writes its stage output into a server-side
`AgentSession`, and the final `TriageResult` is computed by the *same*
deterministic `decide()` the pipeline uses — the agent adds orchestration,
guardrails, and observability, not different answers (design §2).

The read-only guarantee is **structural, not prompt-based** (design §9), in
three layers, outermost first:

  1. **No built-in tools exist.** `ClaudeAgentOptions.tools=[]` empties the
     harness's base tool set — `Bash` / `Write` / `Edit` are not registered at
     all. A tool that doesn't exist cannot be called.
  2. **Allowlist only.** `allowed_tools` pre-approves exactly the four triage
     MCP tools; `permission_mode="default"` (never `bypassPermissions`) and
     `setting_sources=[]` (no looser user/project settings inherited).
  3. **Deny-by-default callbacks.** A `PreToolUse` hook and a `can_use_tool`
     permission callback both deny anything not on the allowlist —
     belt-and-suspenders against an unexpected tool surface.

The only side effect the agent has is the `escalate` tool: an outbound
notification to a human (a log line in this project), idempotent, with no
production mutation. The guard logic (`is_tool_allowed` / `guardrail_spec`) is
pure stdlib so the red-team tests pin it offline — no SDK, no key, no network;
`claude-agent-sdk` is lazy-imported by the live path only and ships as the
optional `agent` extra. SDK symbol names/signatures (`@tool`,
`create_sdk_mcp_server`, `ClaudeAgentOptions`, `ClaudeSDKClient`,
`ResultMessage.total_cost_usd` / `.usage` / `.model_usage`, `PermissionResult*`,
`HookMatcher`) were verified against the installed claude-agent-sdk 0.2.110
(re-checked 2026-07-04), per design §12 — not memorized.

§7/§8 run through the shell too: the orchestrator's own tokens are filed per
model from the SDK's `ResultMessage` and our cache-aware cost estimate is
cross-checked against the SDK's `total_cost_usd`; the §8 per-incident budget
covers the *whole* incident (stages + orchestrator), and once the stage spend
alone crosses it the `draft_response` tool refuses to buy the Opus draft.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from . import observe
from .classify import Classification, Classifier, classify_incident
from .decide import (DEFAULT_THRESHOLDS, Thresholds, decide,
                     escalation_target_for, triage_result)
from .draft import Draft, Drafter, draft_response
from .retrieve import Retrieval, Retriever, retrieve_runbooks
from .schema import Incident, IncidentType, Outcome, Severity, TriageResult

ORCHESTRATOR_MODEL = "claude-opus-4-8"  # design §2: the agent shell orchestrates on Opus
ORCHESTRATOR_EFFORT = "low"  # tool sequencing follows a fixed protocol — low effort keeps
                             # the orchestrator cheap/fast; the intelligence-sensitive work
                             # (the grounded draft) already runs at its own effort in-stage.
MAX_TURNS = 16               # hard loop bound: the protocol needs ~5 turns; runaway = stop

MCP_SERVER = "triage"
TRIAGE_TOOLS = ("classify_incident", "search_runbooks", "draft_response", "escalate")

# The complete tool surface the agent may touch (design §9): the four in-process
# MCP tools, under the harness's `mcp__<server>__<tool>` naming.
ALLOWED_TOOLS: tuple[str, ...] = tuple(f"mcp__{MCP_SERVER}__{name}" for name in TRIAGE_TOOLS)

# Built-in tool names that must never be reachable. Layer 1 (`tools=[]`) already
# unregisters every built-in; this explicit denylist is layer-2 documentation of
# *which* tools the read-only contract is about, and covers a harness that grew
# an unexpected default surface.
MUTATING_TOOLS: tuple[str, ...] = (
    "Bash", "Write", "Edit", "NotebookEdit", "Task", "WebFetch", "WebSearch",
)


def is_tool_allowed(tool_name: str) -> bool:
    """The single guard predicate (design §9): exact-match against the four triage
    MCP tools. Everything else — built-ins, other MCP servers, near-miss names —
    is denied. Pure, so the red-team tests pin it offline."""
    return tool_name in ALLOWED_TOOLS


def deny_reason(tool_name: str) -> str:
    """The machine- and model-readable why for a denied tool call."""
    return (f"tool {tool_name!r} is not on the read-only triage allowlist. "
            f"This agent must never mutate a system (design §9); the only permitted "
            f"tools are: {', '.join(ALLOWED_TOOLS)}.")


# The SDK reports usage in the harness's own key style; map every spelling onto
# the §7 ledger keys `observe.PRICING` prices. Non-token fields (costUSD,
# service_tier, ...) are dropped — token counts stay the authoritative input.
_USAGE_ALIASES = {
    "input_tokens": "input_tokens", "inputTokens": "input_tokens",
    "output_tokens": "output_tokens", "outputTokens": "output_tokens",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "cacheReadInputTokens": "cache_read_input_tokens",
    "cache_creation_input_tokens": "cache_creation_input_tokens",
    "cacheCreationInputTokens": "cache_creation_input_tokens",
}


def normalize_usage(raw: dict[str, Any] | None) -> dict[str, int]:
    """One SDK usage payload → the §7 token-ledger shape (pure, offline-tested)."""
    out: dict[str, int] = {}
    for key, val in (raw or {}).items():
        std = _USAGE_ALIASES.get(key)
        if std and isinstance(val, int) and not isinstance(val, bool):
            out[std] = out.get(std, 0) + val
    return out


def price_key(model: str) -> str:
    """Map an SDK-reported model id onto an `observe.PRICING` key. A date-suffixed
    id (e.g. `claude-opus-4-8-2026...`) matches its base model by prefix; unknown
    models pass through unchanged (unpriced → $0, tokens still recorded)."""
    if model in observe.PRICING:
        return model
    return next((m for m in observe.PRICING if model.startswith(m)), model)


def guardrail_spec() -> dict[str, Any]:
    """The §9 permission policy as plain data, asserted by the offline red-team
    tests and consumed verbatim by `build_options` — so what the tests pin is
    exactly what the live agent runs with."""
    return {
        "tools": [],                              # layer 1: no built-in tool exists at all
        "allowed_tools": list(ALLOWED_TOOLS),     # layer 2: allowlist only
        "disallowed_tools": list(MUTATING_TOOLS),
        "permission_mode": "default",             # never bypassPermissions
        "setting_sources": [],                    # no user/project settings inherited
    }


# --- server-side session state (the tools' shared memory) ---------------------

@dataclass
class AgentSession:
    """One agent triage's server-side state. The MCP tools run in-process, so they
    read the incident and write their stage outputs *here* — the orchestrating
    model only ever sees short text summaries, and `finish()` computes the final
    TriageResult with the same deterministic `decide()` as `run_triage` (design
    §2: same answers, agent adds orchestration). The tool bodies are plain sync
    methods so the offline tests drive them without the SDK."""

    incident: Incident
    classifier: Classifier
    retriever: Retriever
    drafter: Drafter
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    budget: float = observe.DEFAULT_INCIDENT_BUDGET_USD
    trace: observe.Trace = field(default_factory=observe.Trace)
    notify: Callable[[str], None] = print  # the §9 side effect: a human-handoff log line
    classification: Classification | None = None
    retrieval: Retrieval | None = None
    draft: Draft | None = None
    escalations: list[dict[str, str]] = field(default_factory=list)

    def tool_classify(self) -> str:
        self.classification = classify_incident(
            self.incident.prompt_view(), self.classifier, trace=self.trace)
        c = self.classification
        return (f"severity={c.severity.value} (confidence {c.severity_confidence:.2f}), "
                f"type={c.type.value} (confidence {c.type_confidence:.2f})")

    def tool_search(self) -> str:
        self.retrieval = retrieve_runbooks(
            self.incident.prompt_view(), self.retriever, trace=self.trace)
        r = self.retrieval
        if not r.runbooks:
            return "no runbooks retrieved"
        ranked = ", ".join(f"{h.runbook_id} (score {h.score:.2f})" for h in r.runbooks)
        return f"retrieved {len(r.runbooks)} runbook(s), best first: {ranked}"

    def tool_draft(self) -> str:
        if self.retrieval is None:
            return "error: call search_runbooks before draft_response"
        if self.trace.cost() > self.budget:
            # §8: once over budget the Opus draft is never bought — the shell
            # refuses server-side; finish() will abstain with cost_budget_exceeded.
            return ("error: the per-incident cost budget is exhausted — drafting is "
                    "disabled (design §8). Call escalate with a one-line reason.")
        self.draft = draft_response(
            self.incident.prompt_view(), self.retrieval.chunks, self.drafter,
            trace=self.trace)
        d = self.draft
        if not d.grounded:
            return f"drafter abstained (insufficient_evidence): {d.recommendation}"
        cites = ", ".join(f"[{c.n}] {c.section}" for c in d.citations) or "(none)"
        return f"action={d.action_key}; recommendation: {d.recommendation}; citations: {cites}"

    def tool_escalate(self, reason: str = "") -> str:
        """The one permitted side effect (design §9): notify a human. The decider's
        verdict stays authoritative — `finish()` re-derives outcome/target/reason
        from the stage outputs regardless of what the model passed here."""
        target = (escalation_target_for(self.classification.type)
                  if self.classification else "on-call SRE")
        self._post_escalation(target, reason or "agent_requested")
        return f"escalated to {target} (reason: {reason or 'agent_requested'})"

    def _post_escalation(self, target: str, reason: str) -> None:
        self.escalations.append({"target": target, "reason": reason})
        self.notify(f"[escalation] {self.incident.id} -> {target}: {reason}")

    def finish(self, extra_cost_usd: float = 0.0) -> TriageResult:
        """The deterministic join, after the agent loop ends. Complete runs go
        through the exact `decide()` path `run_triage` uses (same thresholds, same
        §8 budget check on the shared trace). `extra_cost_usd` folds the
        orchestrator's own spend (the SDK-reported run cost) into that check, so
        the §8 ceiling covers the whole incident, not just the stages; a run whose
        draft was refused because the budget tripped (see `tool_draft`) still goes
        through `decide()` — with `draft=None` — and abstains with
        `cost_budget_exceeded`. An incomplete run — the orchestrator never
        produced the stage outputs — degrades to the default-safe ABSTAIN with
        reason `agent_incomplete_run` and worst-case severity, never a fabricated
        PROPOSE. Either way, an ABSTAIN that the model didn't already escalate
        posts the human handoff itself — the notification is the shell's
        obligation, not the model's memory."""
        budget_exceeded = self.trace.cost() + extra_cost_usd > self.budget
        if self.classification and self.retrieval and (self.draft or budget_exceeded):
            with self.trace.span("decide"):
                decision = decide(self.classification, self.retrieval, self.draft,
                                  thresholds=self.thresholds,
                                  budget_exceeded=budget_exceeded)
            result = triage_result(self.incident.id, self.classification,
                                   self.retrieval, decision)
        else:
            c = self.classification
            target = escalation_target_for(c.type) if c else "on-call SRE"
            result = TriageResult(
                incident_id=self.incident.id,
                severity=c.severity if c else Severity.SEV1,   # unknown = assume the worst
                severity_confidence=c.severity_confidence if c else 0.0,
                type=c.type if c else IncidentType.UNKNOWN,
                type_confidence=c.type_confidence if c else 0.0,
                outcome=Outcome.ABSTAIN,
                escalation_reason="agent_incomplete_run",
                escalation_target=target,
                retrieved_runbooks=self.retrieval.runbook_ids if self.retrieval else [],
            )
        if result.outcome is Outcome.ABSTAIN and not self.escalations:
            self._post_escalation(result.escalation_target or "on-call SRE",
                                  result.escalation_reason or "abstain")
        return result


# --- the live shell (lazy claude-agent-sdk) -----------------------------------

@dataclass
class AgentTriage:
    """One agent run's outcome + the observability the red-team check needs: every
    tool_use the orchestrator emitted, every call the guard denied, and the SDK's
    own cost/turn accounting (`ResultMessage.total_cost_usd`, §7) — the
    orchestrator's spend, distinct from the stage spend in `session.trace`.

    The §7 cross-check: `orchestrator_usage` files the orchestrator's own
    tokens per model (from `ResultMessage.model_usage`), `orchestrator_cost` is
    our cache-aware USD estimate from those tokens, and `cost_crosscheck_usd` is
    its delta against the SDK's `total_cost_usd` — two independent accountings of
    the same spend that should agree."""

    result: TriageResult
    session: AgentSession
    tool_calls: list[str] = field(default_factory=list)
    denied_calls: list[str] = field(default_factory=list)
    total_cost_usd: float | None = None
    num_turns: int = 0
    duration_ms: int = 0
    is_error: bool = False
    orchestrator_usage: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def non_triage_tool_calls(self) -> list[str]:
        """Tool calls outside the §9 allowlist — the red-team metric; must be []."""
        return [t for t in self.tool_calls if not is_tool_allowed(t)]

    @property
    def orchestrator_cost(self) -> float | None:
        """Our own USD estimate for the orchestrator: its per-model token counts
        priced with `observe.PRICING` (cache-aware) — §7's independent accounting."""
        if not self.orchestrator_usage:
            return None
        return observe.cost_usd(self.orchestrator_usage)["total"]

    @property
    def cost_crosscheck_usd(self) -> float | None:
        """computed − SDK-reported orchestrator cost (§7); None until both exist.
        Should be ~$0 — a large delta means our rate table or the token filing
        drifted from what the API actually billed."""
        ours = self.orchestrator_cost
        if ours is None or self.total_cost_usd is None:
            return None
        return ours - self.total_cost_usd


SYSTEM = (
    "You are the orchestration shell of a READ-ONLY incident first-pass triage agent. "
    "You have exactly four tools and no others; none of them can modify any system, and "
    "you must never attempt remediation yourself — the only side effect you may cause is "
    "escalating to a human.\n"
    "Protocol (in order):\n"
    "1. classify_incident — severity + type.\n"
    "2. search_runbooks — retrieve candidate runbooks.\n"
    "3. draft_response — a grounded, cited first-response draft (or an abstention).\n"
    "4. If the draft abstained (insufficient_evidence), returned action ESCALATE, or the "
    "incident is SEV1: call escalate with a one-line reason.\n"
    "The final PROPOSE/ABSTAIN verdict is computed deterministically by the harness from "
    "the tool outputs — do not decide it yourself. After the protocol, reply with the "
    "single word DONE."
)


def _load_sdk():
    """Lazy-import claude-agent-sdk (the optional `agent` extra). Only the live
    agent path needs it — the offline suite pins the guard logic without it."""
    try:
        import claude_agent_sdk as sdk  # noqa: PLC0415  (lazy by design)
    except ModuleNotFoundError as e:
        raise SystemExit(
            "claude-agent-sdk is not installed (it ships as the optional 'agent' extra).\n"
            "  Run:  uv run --with claude-agent-sdk --with anthropic --with sqlite-vec "
            "python -m triage agent INC-0001\n"
            "  or:   pip install -e '.[agent,live]'"
        ) from e
    return sdk


def build_mcp_server(session: AgentSession):
    """Wrap the session's stage methods as an in-process MCP server (design §2).
    The tools take (almost) no arguments — the incident and the stage outputs live
    server-side in `session`, so gold labels and raw stage state never round-trip
    through the model."""
    sdk = _load_sdk()

    def text(payload: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": payload}]}

    no_args = {"type": "object", "properties": {}, "additionalProperties": False}
    escalate_schema = {
        "type": "object",
        "properties": {"reason": {"type": "string",
                                  "description": "one-line reason for the human handoff"}},
        "additionalProperties": False,
    }

    @sdk.tool("classify_incident",
              "Classify the current incident's severity and type. Read-only.", no_args)
    async def classify_tool(args: dict[str, Any]) -> dict[str, Any]:
        return text(session.tool_classify())

    @sdk.tool("search_runbooks",
              "Retrieve the runbooks most relevant to the current incident. Read-only.",
              no_args)
    async def search_tool(args: dict[str, Any]) -> dict[str, Any]:
        return text(session.tool_search())

    @sdk.tool("draft_response",
              "Draft a cited first-response recommendation grounded in the retrieved "
              "runbook sections, or abstain. Read-only.", no_args)
    async def draft_tool(args: dict[str, Any]) -> dict[str, Any]:
        return text(session.tool_draft())

    @sdk.tool("escalate",
              "Escalate the current incident to the on-call human. The only permitted "
              "side effect: an outbound notification, no system mutation.",
              escalate_schema)
    async def escalate_tool(args: dict[str, Any]) -> dict[str, Any]:
        return text(session.tool_escalate(str(args.get("reason", ""))))

    return sdk.create_sdk_mcp_server(
        MCP_SERVER, version="1.0.0",
        tools=[classify_tool, search_tool, draft_tool, escalate_tool])


def build_options(server, *, denied: list[str] | None = None, max_turns: int = MAX_TURNS):
    """ClaudeAgentOptions carrying the §9 policy — every field comes verbatim from
    `guardrail_spec()` (what the offline tests pin is what runs live), plus the
    deny-by-default `can_use_tool` callback and `PreToolUse` hook. `denied`, if
    given, journals every guarded-off tool name for the red-team report."""
    sdk = _load_sdk()
    spec = guardrail_spec()

    async def can_use(tool_name: str, tool_input: dict[str, Any], context):
        if is_tool_allowed(tool_name):
            return sdk.PermissionResultAllow()
        if denied is not None:
            denied.append(tool_name)
        return sdk.PermissionResultDeny(message=deny_reason(tool_name), interrupt=False)

    async def pre_tool_guard(input_data, tool_use_id, context):
        name = str(input_data.get("tool_name", ""))
        if is_tool_allowed(name):
            return {}
        if denied is not None:
            denied.append(name)
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": deny_reason(name)}}

    return sdk.ClaudeAgentOptions(
        tools=spec["tools"],
        allowed_tools=spec["allowed_tools"],
        disallowed_tools=spec["disallowed_tools"],
        permission_mode=spec["permission_mode"],
        setting_sources=spec["setting_sources"],
        mcp_servers={MCP_SERVER: server},
        system_prompt=SYSTEM,
        model=ORCHESTRATOR_MODEL,
        effort=ORCHESTRATOR_EFFORT,
        max_turns=max_turns,
        can_use_tool=can_use,
        hooks={"PreToolUse": [sdk.HookMatcher(hooks=[pre_tool_guard])]},
    )


def build_prompt(incident: Incident) -> str:
    """The kickoff message: id + title only. The body reaches the stages through
    the server-side session, not through the orchestrator's context — smaller
    prompts, and the model can't be prompt-injected by the alert text it never sees."""
    return (f"Triage incident {incident.id}: {incident.title!r}. "
            f"Follow the protocol, then reply DONE.")


async def _run_agent(session: AgentSession, *, max_turns: int) -> AgentTriage:
    sdk = _load_sdk()
    denied: list[str] = []
    server = build_mcp_server(session)
    options = build_options(server, denied=denied, max_turns=max_turns)
    run = AgentTriage(result=None, session=session, denied_calls=denied)  # type: ignore[arg-type]

    async with sdk.ClaudeSDKClient(options=options) as client:
        await client.query(build_prompt(session.incident))
        async for message in client.receive_response():
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.ToolUseBlock):
                        run.tool_calls.append(block.name)
            elif isinstance(message, sdk.ResultMessage):
                run.total_cost_usd = message.total_cost_usd
                run.num_turns = message.num_turns
                run.duration_ms = message.duration_ms
                run.is_error = message.is_error
                # §7: file the orchestrator's tokens under their model. The
                # per-model split is authoritative; fall back to the aggregate
                # usage under the configured orchestrator model.
                per_model = message.model_usage or (
                    {ORCHESTRATOR_MODEL: message.usage} if message.usage else {})
                for model, usage in per_model.items():
                    tokens = normalize_usage(usage)
                    if tokens:
                        observe.merge_usage(run.orchestrator_usage, price_key(model), tokens)

    # §8: the budget verdict covers the whole incident — stage spend plus the
    # orchestrator's own spend (SDK-reported; our estimate when the SDK gave none).
    run.result = session.finish(
        extra_cost_usd=run.total_cost_usd if run.total_cost_usd is not None
        else (run.orchestrator_cost or 0.0))
    return run


def run_agent_triage(incident: Incident, classifier: Classifier, retriever: Retriever,
                     drafter: Drafter, *, thresholds: Thresholds = DEFAULT_THRESHOLDS,
                     budget: float = observe.DEFAULT_INCIDENT_BUDGET_USD,
                     notify: Callable[[str], None] = print,
                     max_turns: int = MAX_TURNS) -> AgentTriage:
    """Run one incident through the Agent SDK shell (design §2): Opus orchestrates
    the four MCP tools under the §9 read-only policy, then `finish()` computes the
    TriageResult with the pipeline's own `decide()`. The surface behind
    `python -m triage agent <INC-ID>`. Needs the `agent`+`live` extras, both keys,
    a built runbook index, and the Claude Code CLI."""
    session = AgentSession(incident=incident, classifier=classifier,
                           retriever=retriever, drafter=drafter,
                           thresholds=thresholds, budget=budget, notify=notify)
    return asyncio.run(_run_agent(session, max_turns=max_turns))
