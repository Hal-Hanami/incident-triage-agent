"""Stage: draft — a cited first-response recommendation (design §4.3, §5).

The quality-sensitive step, and the literal continuation of tech-docs-rag's grounding
contract: the drafter sees the retrieved runbook sections **and nothing else**,
the system prompt forbids outside knowledge, and **every claim must carry a `[n]`
citation** back to a section it was given. If those sections do not actually
support a first response for *this* incident, the drafter returns the abstention
sentinel (`insufficient_evidence`) instead of guessing — exactly tech-docs-rag's
"don't answer what the sources don't support," one rung up the autonomy ladder.

Two halves behind one `Drafter` Protocol (the same seam as classify/retrieve):
`draft_response()` is the orchestration + per-stage timing, and `ClaudeDrafter`
is the real `claude-opus-4-8` structured-output call. The Protocol lets the eval
loop (design §10) run offline with a fake — no key, no network.

The action is a closed enum (the runbook action keys + `ESCALATE` + the abstention
sentinel), pinned by a structured-output `json_schema`, so it is scorable as a
plain key-match (design §10) and so a runbook that itself directs escalation
(`RB-db-failover`) surfaces as `action == ESCALATE`, which the decider (§6) turns
into an escalation regardless of the predicted severity. Citations are the `[n]`
markers in the recommendation prose, parsed back to the sections they reference —
a recommendation with no in-range `[n]` is not citation-backed and cannot be
proposed (§6.1). Model id, the request shape, and the effort/thinking knobs are
sourced from the `claude-api` skill (re-checked 2026-06-17), not memorized.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from . import observe
from .observe import Trace
from .schema import ACTION_KEYS, Citation

MODEL = "claude-opus-4-8"  # design §2: draft = Opus, the quality-sensitive grounded step
EFFORT = "medium"          # adaptive thinking + medium effort: enough to judge grounding
                           # faithfully (the abstention/citation decision) without overthinking
                           # a short, bounded drafting task. Tunable.

# The drafter's abstention sentinel (design §5/§6.2): the sources don't support a
# first response for this incident. Distinct from schema.ESCALATE (a runbook that
# directs a human handoff) and from Outcome.ABSTAIN (the decider's verdict).
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

# Every value the structured-output `action` field may take.
_ALLOWED_ACTIONS: frozenset[str] = ACTION_KEYS | {INSUFFICIENT_EVIDENCE}

_CITATION_RE = re.compile(r"\[(\d+)\]")  # the [n] markers that make a claim auditable (§5)


@dataclass
class Draft:
    """The drafter's verdict for one incident: a single action key grounded in the
    retrieved sections, the cited recommendation prose, the citations parsed from
    its `[n]` markers, and optional token usage for the cost ledger (design §7).
    `action_key == INSUFFICIENT_EVIDENCE` is the abstention sentinel (§5/§6.2).
    """

    action_key: str
    recommendation: str = ""
    citations: list[Citation] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        """True iff the drafter committed to an action (did not abstain)."""
        return self.action_key != INSUFFICIENT_EVIDENCE

    @property
    def is_citation_backed(self) -> bool:
        """A proposable draft: grounded AND carrying ≥1 in-range citation (design §5).
        The decider gates PROPOSE on this (§6.1); an uncited draft can't be proposed."""
        return self.grounded and bool(self.citations)


class Drafter(Protocol):
    """LLM seam — `draft(view, sections) -> Draft`. Faked offline in tests; the real
    `claude-opus-4-8` implementation (`ClaudeDrafter`) plugs in here."""

    def draft(self, view: dict[str, str], sections: list[dict]) -> Draft: ...


def draft_response(view: dict[str, str], sections: list[dict], drafter: Drafter,
                   *, trace: Trace | None = None) -> Draft:
    """Run the draft stage for one incident's prompt view + retrieved sections.

    Mirrors `classify.classify_incident` / `retrieve.retrieve_runbooks`: `trace`,
    if given, times the stage and files the drafter's token usage under its model
    for the per-stage latency + per-model cost story (design §7)."""
    with observe.span(trace, "draft"):
        result = drafter.draft(view, sections)
    if trace is not None and result.usage:
        trace.add_usage(getattr(drafter, "model", MODEL), result.usage)
    return result


# --- the real claude-opus-4-8 implementation ----------------------------

SYSTEM = (
    "You are an incident first-response drafter for an SRE / on-call triage agent. "
    "You are given one incident and a set of numbered runbook sections [1], [2], .... "
    "Recommend the single first-response ACTION the on-call human should take, grounded "
    "ONLY in those sections.\n"
    "Rules:\n"
    "- Use ONLY the provided runbook sections. Do NOT use outside knowledge and do NOT "
    "invent steps.\n"
    "- Choose exactly one `action` from the allowed keys, OR 'insufficient_evidence' to abstain.\n"
    "- Abstain ('insufficient_evidence') when the provided sections do not actually address "
    "THIS incident, or a section says not to act in this situation. Do NOT stretch a loosely-"
    "related runbook to fit. Abstaining is correct and expected when the incident is out of scope.\n"
    "- If a section directs handing the incident to a human (e.g. a database failover), choose "
    "action 'ESCALATE'.\n"
    "- Every claim in `recommendation` MUST carry a [n] citation pointing to the section you used "
    "(e.g. \"roll back the last deploy [2]\"). A recommendation with no [n] citation is invalid.\n"
    "- Keep `recommendation` to 1-3 sentences: the first action and why, each claim cited. When you "
    "abstain, briefly say why no provided section supports a response (no citation needed).\n"
    "Allowed action keys: " + ", ".join(sorted(ACTION_KEYS)) + ", insufficient_evidence."
)

# Structured-output schema pinning the closed action set. Per the claude-api skill,
# json_schema does not support string-length/array constraints, so `recommendation`
# is a plain string and citation validity is enforced client-side in parse_draft.
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(ACTION_KEYS) + [INSUFFICIENT_EVIDENCE]},
        "recommendation": {"type": "string"},
    },
    "required": ["action", "recommendation"],
    "additionalProperties": False,
}


def format_sections(sections: list[dict]) -> str:
    """Render the retrieved chunks as numbered, citable units [1], [2], … (design §5)."""
    blocks: list[str] = []
    for i, s in enumerate(sections, start=1):
        path = s.get("section_path") or s.get("source_url", "")
        text = (s.get("text") or "").strip()
        blocks.append(f"[{i}] {path}\n{text}")
    return "\n\n".join(blocks)


def build_draft_user(view: dict[str, str], sections: list[dict]) -> str:
    """The incident prompt view + the numbered runbook sections (and nothing else)."""
    return (f"Incident {view['id']} (source: {view['source']})\n"
            f"Title: {view['title']}\n\n{view['body']}\n\n"
            f"Runbook sections:\n{format_sections(sections)}")


def extract_citations(recommendation: str, sections: list[dict]) -> list[Citation]:
    """Parse the distinct in-range `[n]` markers from the prose into Citations.

    A claim's `[n]` maps 1:1 to the section presented as `[n]` (design §5); markers
    out of `[1, len(sections)]` are dropped (a hallucinated reference is not a valid
    citation). First occurrence wins; order preserved."""
    out: list[Citation] = []
    seen: set[int] = set()
    for m in _CITATION_RE.finditer(recommendation):
        n = int(m.group(1))
        if n in seen or not (1 <= n <= len(sections)):
            continue
        seen.add(n)
        s = sections[n - 1]
        out.append(Citation(n=n, runbook_id=s.get("source_url", ""),
                            section=s.get("section_path", ""), source=s.get("url", "")))
    return out


def parse_draft(text: str, sections: list[dict]) -> Draft:
    """Parse the model's JSON verdict into a Draft.

    Strict on the action enum (structured outputs guarantee a valid value, so a bad
    one is a real bug worth surfacing), and the citations are derived from the prose's
    `[n]` markers against the sections actually provided."""
    data = json.loads(text)
    action = data["action"]
    if action not in _ALLOWED_ACTIONS:
        raise ValueError(f"drafter returned unknown action {action!r}")
    recommendation = data.get("recommendation", "")
    citations: list[Citation] = []
    if action != INSUFFICIENT_EVIDENCE:
        citations = extract_citations(recommendation, sections)
    return Draft(action_key=action, recommendation=recommendation, citations=citations)


class ClaudeDrafter:
    """Cited first-response drafter on claude-opus-4-8 via the official SDK with
    structured outputs (design §2/§4.3/§5). Opus is the quality-sensitive path; the
    request pins the closed-enum action with an `output_config` json_schema and runs
    adaptive thinking at `medium` effort so the model can reason about grounding
    (does this runbook actually fit?) before committing — the call that drives the
    abstention contract. The lazy `anthropic` import + key check mirror
    `ClaudeClassifier`, so the offline commands and test suite need neither the SDK
    nor a key.
    """

    def __init__(self, model: str = MODEL, *, max_tokens: int = 4096, effort: str = EFFORT) -> None:
        try:
            import anthropic  # lazy: only the live draft path needs it
        except ModuleNotFoundError as e:
            raise SystemExit(
                "the `anthropic` SDK is not installed (it ships as the optional 'live' extra).\n"
                "  Run:  uv run --with anthropic --with sqlite-vec python -m triage eval\n"
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
        self.max_tokens = max_tokens   # ample headroom for adaptive thinking + a short verdict
        self.effort = effort
        self._client = anthropic.Anthropic()

    def draft(self, view: dict[str, str], sections: list[dict]) -> Draft:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort,
                           "format": {"type": "json_schema", "schema": DRAFT_SCHEMA}},
            messages=[{"role": "user", "content": build_draft_user(view, sections)}],
        )
        if msg.stop_reason == "max_tokens":
            raise SystemExit(
                f"drafter hit max_tokens ({self.max_tokens}) on {view['id']} — output truncated; "
                "raise ClaudeDrafter(max_tokens=...) or lower effort."
            )
        text = "".join(b.text for b in msg.content if b.type == "text")
        result = parse_draft(text, sections)
        result.usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        return result
