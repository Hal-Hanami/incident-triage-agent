# RB-app-5xx — Elevated 5xx / error rate on a service

**Applies to:** a sustained jump in HTTP 5xx or application error rate on a single
service, especially shortly after a deploy. Type: `app_error`.

## First response (read-only)
1. Confirm the signal: error-rate panel for the service, last 30 min. Is it one
   endpoint or all? One pod or all?
2. Check the deploy timeline. Did the error rate rise within ~10 min of a release?
   Note the previous known-good version.
3. Check the error budget / SLO burn rate to size urgency.

## Recommended first action
**`roll_back_last_deploy`** — if the spike correlates with a recent deploy, the
fastest safe mitigation is to roll back to the previous known-good version. This
is a recommendation for the on-call engineer to execute; the agent proposes it and
cites this section. If there is **no** correlated deploy, do not recommend a
rollback — treat as ambiguous and lower confidence.

## Escalate if
- The rollback is not obviously safe (in-flight schema migration, stateful change).
- Error rate is ≥ SEV1 (full outage) — escalate to a human immediately regardless.
