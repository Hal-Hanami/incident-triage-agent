# RB-latency — Elevated latency without elevated errors

**Applies to:** a rise in p95/p99 latency while error rate stays roughly normal,
often traceable to a slow upstream dependency. Type: `network`.

## First response (read-only)
1. Confirm latency is up but errors are flat (rules out a hard failure).
2. Walk the trace: which hop added the time — this service, an upstream, the DB?
3. Check upstream dependencies' own latency panels for a correlated rise.

## Recommended first action
**`investigate_upstream_shed_load`** — if the latency originates upstream, focus
there; if this service is saturated, recommend load shedding / concurrency limits
to protect the rest of the system. The agent proposes the option with a citation; a
human applies it.

## Escalate if
- Latency is high enough to breach SLO and trip a SEV1.
- The slow hop is a managed third party you cannot tune.
