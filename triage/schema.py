"""The incident / triage data model — the contract every stage reads and writes.

Pure stdlib, no LLM, no I/O beyond loading the JSONL fixtures. This is the spine
of the eval (`Incident.gold_*` are the labels we score against) and of the
guardrail story (`Decision` has exactly two outcomes — `PROPOSE` a cited
recommendation, or `ABSTAIN` and escalate to a human; there is no "execute").

Severity / IncidentType are closed enums so classification is scorable as plain
accuracy. `in_scope=False` incidents (no runbook exists) are the abstention
tests — the agent must decline rather than fabricate a response, exactly like
tech-docs-rag's out-of-corpus questions (design §4, §6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator


class Severity(str, Enum):
    """SEVn, smaller = worse. SEV1 is always escalated to a human (design §6.3)."""

    SEV1 = "SEV1"  # critical — full outage / data loss risk / security breach
    SEV2 = "SEV2"  # major — significant degradation, partial outage
    SEV3 = "SEV3"  # minor — degraded but serving, limited blast radius
    SEV4 = "SEV4"  # low — cosmetic / informational


class IncidentType(str, Enum):
    """Closed set of incident categories the classifier maps an alert onto."""

    APP_ERROR = "app_error"                # elevated 5xx / exceptions / crash loops
    INFRA_CAPACITY = "infra_capacity"      # CPU / memory / disk / quota pressure
    NETWORK = "network"                    # latency, packet loss, LB / DNS
    DATABASE = "database"                  # replica lag, connection exhaustion, failover
    AUTH_ACCESS = "auth_access"            # authn/authz, SSO, certs, tokens
    DEPLOYMENT = "deployment"              # rollout / canary / migration failures
    DATA_PIPELINE = "data_pipeline"        # ETL / batch / streaming job failures
    SECURITY_SUSPECTED = "security_suspected"  # suspected compromise / anomalous activity
    THIRD_PARTY_OUTAGE = "third_party_outage"  # upstream provider degradation
    UNKNOWN = "unknown"                    # classifier is not confident enough to commit


ESCALATE = "ESCALATE"  # sentinel gold_action: the correct first response is "hand to a human"

# The runbook each in-scope, answerable incident maps to, and that runbook's single
# first-response action key (design §3, §6). RB-db-failover's action is ESCALATE by
# policy — failover is high-blast-radius and human-only (design §6.3). ESCALATE is the
# cross-cutting "hand to a human" outcome (SEV1 rule §6.3 / out-of-scope §6.2) and is
# not tied to a runbook. Kept here so the fixture validator can confirm a gold_action
# is a real key and lines up with the runbook it expects — trustworthy eval ground
# truth (design §10).
RUNBOOK_ACTIONS: dict[str, str] = {
    "RB-app-5xx": "roll_back_last_deploy",
    "RB-disk-pressure": "relieve_disk_pressure",
    "RB-latency": "investigate_upstream_shed_load",
    "RB-deploy-canary": "halt_rollout_pin_previous",
    "RB-thirdparty": "enable_degraded_mode_monitor",
    "RB-auth-sso": "verify_sso_cert_clock",
    "RB-db-failover": ESCALATE,
}

# Every valid gold_action: the per-runbook action keys plus the ESCALATE sentinel.
ACTION_KEYS: frozenset[str] = frozenset(RUNBOOK_ACTIONS.values()) | {ESCALATE}


@dataclass(frozen=True)
class Incident:
    """One synthetic incident — the input plus the gold labels we score against.

    Only `id/title/body/source` are shown to the agent. The `gold_*`, `in_scope`,
    `expected_runbook`, and `notes` fields are eval ground truth and author notes;
    they are never part of the prompt.
    """

    id: str
    title: str
    body: str
    source: str                              # e.g. "pagerduty", "datadog", "ticket"
    gold_severity: Severity
    gold_type: IncidentType
    gold_action: str                         # a runbook action key, or ESCALATE
    in_scope: bool                           # True iff a runbook covers this -> answerable
    expected_runbook: str | None = None      # runbook id retrieval should surface (recall target)
    tags: tuple[str, ...] = ()               # e.g. "hard", "ambiguous", "out_of_scope"
    notes: str = ""                          # author rationale — NOT shown to the agent

    @property
    def must_abstain(self) -> bool:
        """Ground truth for the abstention metric: out-of-scope OR explicitly ESCALATE."""
        return (not self.in_scope) or self.gold_action == ESCALATE

    def prompt_view(self) -> dict[str, str]:
        """Exactly the fields the agent is allowed to see — guards against label leakage."""
        return {"id": self.id, "title": self.title, "body": self.body, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "Incident":
        return cls(
            id=d["id"],
            title=d["title"],
            body=d["body"],
            source=d["source"],
            gold_severity=Severity(d["gold_severity"]),
            gold_type=IncidentType(d["gold_type"]),
            gold_action=d["gold_action"],
            in_scope=bool(d["in_scope"]),
            expected_runbook=d.get("expected_runbook"),
            tags=tuple(d.get("tags", [])),
            notes=d.get("notes", ""),
        )


class Outcome(str, Enum):
    PROPOSE = "PROPOSE"    # a cited first-response recommendation for a human to apply
    ABSTAIN = "ABSTAIN"    # confidence too low / out of scope -> escalate, do not act


@dataclass
class Citation:
    """A [n] marker tying one claim back to a runbook section (design §5)."""

    n: int
    runbook_id: str
    section: str
    source: str            # path or URL of the runbook section


@dataclass
class TriageResult:
    """The agent's output for one incident — what gets scored and what a human reads."""

    incident_id: str
    severity: Severity
    severity_confidence: float
    type: IncidentType
    type_confidence: float
    outcome: Outcome
    proposed_action: str | None = None       # set iff outcome == PROPOSE
    citations: list[Citation] = field(default_factory=list)
    escalation_reason: str | None = None     # set iff outcome == ABSTAIN
    escalation_target: str | None = None     # e.g. "on-call SRE", "security on-call"
    retrieved_runbooks: list[str] = field(default_factory=list)  # ids, in rank order


def load_incidents(path: str | Path) -> list[Incident]:
    """Load the synthetic incident set from a JSONL file (one incident per line)."""
    return list(iter_incidents(path))


def iter_incidents(path: str | Path) -> Iterator[Incident]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield Incident.from_dict(json.loads(line))


def validate(incidents: Iterable[Incident]) -> list[str]:
    """Return a list of fixture-integrity problems (empty list == valid).

    Invariants that keep the eval honest: unique ids; in-scope incidents name an
    `expected_runbook`; out-of-scope incidents do not; ESCALATE actions are flagged
    so the abstention metric and the SEV1 rule line up; every `gold_action` is a real
    key and an in-scope answerable action matches the runbook it expects.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for inc in incidents:
        if inc.id in seen:
            problems.append(f"{inc.id}: duplicate id")
        seen.add(inc.id)
        if inc.in_scope and not inc.expected_runbook:
            problems.append(f"{inc.id}: in_scope incident has no expected_runbook")
        if not inc.in_scope and inc.expected_runbook:
            problems.append(f"{inc.id}: out-of-scope incident names a runbook ({inc.expected_runbook})")
        if not inc.in_scope and inc.gold_action != ESCALATE:
            problems.append(f"{inc.id}: out-of-scope incident must have gold_action=ESCALATE")
        if inc.gold_action not in ACTION_KEYS:
            problems.append(f"{inc.id}: unknown gold_action {inc.gold_action!r} (not a runbook action or ESCALATE)")
        # An in-scope incident whose action is a real fix (not ESCALATE) must point at
        # the runbook that documents that fix — otherwise retrieval recall and action
        # correctness would be scored against a mismatched target.
        if inc.in_scope and inc.expected_runbook and inc.gold_action != ESCALATE:
            expected_action = RUNBOOK_ACTIONS.get(inc.expected_runbook)
            if expected_action is None:
                problems.append(f"{inc.id}: expected_runbook {inc.expected_runbook!r} is not a known runbook")
            elif inc.gold_action != expected_action:
                problems.append(
                    f"{inc.id}: gold_action {inc.gold_action!r} does not match "
                    f"{inc.expected_runbook} (expects {expected_action!r})")
    return problems
