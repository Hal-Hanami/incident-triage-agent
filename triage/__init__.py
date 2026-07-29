"""Incident first-pass triage agent — read-only, measured, cost-capped, observable.

A four-stage pipeline behind a Claude Agent SDK + MCP shell (design `docs/design.md`):

    classify  →  retrieve  →  draft  →  decide
    (sev+type)   (runbook)    (cited)    (propose | ABSTAIN→escalate)

The retrieve stage reuses the `rag` package from the sibling tech-docs-rag
project over a synthetic runbook corpus; the draft stage grounds every claim in a
`[n]` citation or abstains. The abstention contract from tech-docs-rag ("don't
answer what the sources don't support") is inherited here as an *agent* contract:
"if confidence is low, don't act — escalate to a human." The agent's entire tool
surface is read-only.

`schema` (the incident / triage data model) and `observe` (per-model cost +
per-stage latency) are stdlib-only, so importing this package costs nothing and
needs no key; every model client is lazy-imported by the stage that uses it.
"""
