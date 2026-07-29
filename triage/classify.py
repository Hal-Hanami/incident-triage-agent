"""Stage: classify — severity + type for an inbound incident (design §4.1).

Two halves behind one `Classifier` Protocol (the tech-docs-rag `Answerer`/`Judge` seam):
`classify_incident()` is the orchestration + per-stage timing, and `ClaudeClassifier`
is the real `claude-haiku-4-5` structured-output call. The Protocol lets the eval
loop (design §10) run offline with a fake — no key, no network — while the live path
measures real accuracy.

Severity and type are closed enums (`schema.Severity` / `schema.IncidentType`) pinned
by a structured-output `json_schema`, so the output is scorable as plain accuracy. The
classifier only ever sees `Incident.prompt_view()` (id/title/body/source) — gold labels
never reach the model. Model id, the structured-output request shape, and pricing are
sourced from the `claude-api` skill (re-checked 2026-06-17), not memorized.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol

from . import observe
from .observe import Trace
from .schema import IncidentType, Severity

MODEL = "claude-haiku-4-5"  # design §2: classification = Haiku (cheap, structured, high-volume)


@dataclass
class Classification:
    """A classifier's verdict for one incident: a severity and a type, each with a
    confidence in [0,1], plus optional token usage for the cost ledger (design §7).
    """

    severity: Severity
    type: IncidentType
    severity_confidence: float = 1.0
    type_confidence: float = 1.0
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def min_confidence(self) -> float:
        """The weaker of the two confidences — the signal the decider gates on."""
        return min(self.severity_confidence, self.type_confidence)


class Classifier(Protocol):
    """LLM seam — `classify(view) -> Classification`. Faked offline in tests; the
    real `claude-haiku-4-5` implementation (`ClaudeClassifier`) plugs in here.
    """

    def classify(self, view: dict[str, str]) -> Classification: ...


def classify_incident(view: dict[str, str], classifier: Classifier,
                      *, trace: Trace | None = None) -> Classification:
    """Run the classify stage for one incident's prompt view.

    Split out so both the eval harness and the per-incident CLI share one
    orchestration path. `trace`, if given, times the stage and files the model's
    token usage under its name for the per-stage latency + per-model cost story
    (design §7); the eval harness passes `trace=None` and times the whole item
    itself, then records usage from the returned `Classification`.
    """
    with observe.span(trace, "classify"):
        result = classifier.classify(view)
    if trace is not None and result.usage:
        trace.add_usage(getattr(classifier, "model", MODEL), result.usage)
    return result


# --- the real claude-haiku-4-5 implementation --------------------------

SYSTEM = (
    "You are an incident triage classifier for an SRE / on-call first-response system. "
    "Given an inbound alert or ticket, classify its SEVERITY and TYPE.\n"
    "Severity (smaller is worse):\n"
    "- SEV1: critical - full outage, data-loss risk, or security breach.\n"
    "- SEV2: major - significant degradation or partial outage.\n"
    "- SEV3: minor - degraded but still serving, limited blast radius.\n"
    "- SEV4: low - cosmetic / informational, no user impact.\n"
    "Type is exactly one of:\n"
    "- app_error: elevated 5xx, exceptions, crash loops.\n"
    "- infra_capacity: cpu / memory / disk / quota pressure.\n"
    "- network: latency, packet loss, load balancer / DNS.\n"
    "- database: replica lag, connection exhaustion, failover.\n"
    "- auth_access: authn / authz, SSO, certificates, tokens.\n"
    "- deployment: rollout / canary / migration failures.\n"
    "- data_pipeline: ETL / batch / streaming job failures.\n"
    "- security_suspected: suspected compromise or anomalous activity.\n"
    "- third_party_outage: upstream provider degradation.\n"
    "- unknown: the alert does not fit any category or is too vague to type.\n"
    "Rules:\n"
    "- Choose exactly one severity and one type from the allowed values.\n"
    "- Give a calibrated confidence in [0,1] for each: high when the signal is clear, "
    "low when the alert is ambiguous or under-specified.\n"
    "- Judge only from the alert text. Do not invent details."
)

# Structured-output schema pinning the closed enums. Per the claude-api skill,
# json_schema does NOT support numeric constraints (minimum/maximum), so confidence
# is a plain number and is clamped to [0,1] client-side in parse_classification.
CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "severity": {"type": "string", "enum": [s.value for s in Severity]},
        "type": {"type": "string", "enum": [t.value for t in IncidentType]},
        "severity_confidence": {"type": "number"},
        "type_confidence": {"type": "number"},
    },
    "required": ["severity", "type", "severity_confidence", "type_confidence"],
    "additionalProperties": False,
}


def build_classify_user(view: dict[str, str]) -> str:
    """Render the prompt view (and nothing else) as the user message."""
    return (f"Incident {view['id']} (source: {view['source']})\n"
            f"Title: {view['title']}\n\n{view['body']}")


def _clamp01(x: object) -> float:
    return max(0.0, min(1.0, float(x)))  # type: ignore[arg-type]


def parse_classification(text: str) -> Classification:
    """Parse the model's JSON verdict into a Classification.

    Strict on the enums (structured outputs guarantee valid JSON + allowed enum
    values, so a failure is a real bug worth surfacing), lenient on confidence —
    the schema can't bound it, so it's clamped to [0,1] here.
    """
    data = json.loads(text)
    return Classification(
        severity=Severity(data["severity"]),
        type=IncidentType(data["type"]),
        severity_confidence=_clamp01(data.get("severity_confidence", 1.0)),
        type_confidence=_clamp01(data.get("type_confidence", 1.0)),
    )


class ClaudeClassifier:
    """Severity + type classifier on claude-haiku-4-5 via the official SDK with
    structured outputs (design §2/§4.1). Haiku is the cheap, high-volume path; the
    `effort` parameter is unsupported on Haiku so it is omitted, and an
    `output_config` json_schema pins the closed-enum verdict. The lazy `anthropic`
    import + key check mirror tech-docs-rag's `ClaudeAnswerer`, so the offline commands
    and the offline test suite need neither the SDK nor a key.
    """

    def __init__(self, model: str = MODEL, max_tokens: int = 512) -> None:
        try:
            import anthropic  # lazy: only the live classify path needs it
        except ModuleNotFoundError as e:
            raise SystemExit(
                "the `anthropic` SDK is not installed (it ships as the optional 'live' extra).\n"
                "  Run:  uv run --with anthropic python -m triage eval\n"
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
        self.max_tokens = max_tokens  # the verdict is tiny; 512 is ample headroom
        self._client = anthropic.Anthropic()

    def classify(self, view: dict[str, str]) -> Classification:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_classify_user(view)}],
            output_config={"format": {"type": "json_schema", "schema": CLASSIFY_SCHEMA}},
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        result = parse_classification(text)
        result.usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        return result
