# RB-disk-pressure — Disk / volume capacity pressure

**Applies to:** a volume approaching full (typically > 80%) before it causes an
outage. Type: `infra_capacity`.

## First response (read-only)
1. Confirm which volume/host and the fill rate (e.g. %/hour) to estimate time-to-full.
2. Identify the top consumers (largest directories, log growth, runaway writers).
3. Check whether anything user-facing is already affected.

## Recommended first action
**`relieve_disk_pressure`** — recommend the lowest-risk relief first: rotate or ship
logs, clear reclaimable caches, or expand the volume. Prefer expansion over deletion
when data could be load-bearing. The agent proposes the option and cites this
section; a human applies it.

## Escalate if
- The volume is a database or stateful store, or deletion risks data loss.
- Time-to-full is under ~30 min (capacity emergency).
