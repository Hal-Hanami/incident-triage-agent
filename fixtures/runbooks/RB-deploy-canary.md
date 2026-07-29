# RB-deploy-canary — Canary / rollout failing health checks

**Applies to:** a progressive rollout whose canary is failing readiness/health
checks or burning error budget, while stable pods stay healthy. Type: `deployment`.

## First response (read-only)
1. Confirm the failure is isolated to canary pods (stable fleet healthy).
2. Read canary pod events/logs for the failure mode (crash loop, failed readiness,
   bad config).
3. Note the previous stable revision.

## Recommended first action
**`halt_rollout_pin_previous`** — pause the rollout and pin traffic to the previous
stable revision so the blast radius stays contained while the canary is debugged.
The agent recommends this and cites this section; a human applies it.

## Escalate if
- Stable pods are also degrading (the rollout is not the only problem).
- The change includes an irreversible data migration.
