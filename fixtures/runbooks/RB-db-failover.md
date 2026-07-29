# RB-db-failover — Primary database unreachable

**Applies to:** the primary database is unreachable or refusing connections and
services are failing. Type: `database`.

> ⚠️ Incidents that match this runbook are almost always **SEV1**. Per policy
> (design §6.3) the triage agent does **not** propose a failover — it **escalates
> to a human (on-call DBA / SRE) immediately**. This runbook documents what that
> human will check; it is here so retrieval surfaces the right context, not so the
> agent acts on it.

## First response (read-only, for the on-call human)
1. Confirm reachability of primary vs replica; distinguish "DB down" from "network
   partition" from "connection-pool exhaustion".
2. Check replication lag on the standby before any failover decision.
3. Check for an in-progress maintenance / migration that could explain it.

## Recommended first action
**`ESCALATE`** — hand to the database on-call. Failover is a high-blast-radius,
hard-to-reverse action and is never automated by this agent.
