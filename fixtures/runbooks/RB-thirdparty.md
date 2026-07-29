# RB-thirdparty — Third-party provider degradation

**Applies to:** an upstream provider (payments, email, SMS, CDN) reporting degraded
performance while your own services are still serving. Type: `third_party_outage`.

## First response (read-only)
1. Confirm the provider's own status page / webhook and the scope of degradation.
2. Check your retry/timeout behavior against that provider — are retries amplifying
   load or masking the issue?
3. Confirm customer impact (degraded vs failing).

## Recommended first action
**`enable_degraded_mode_monitor`** — enable any graceful-degradation path (queue
and retry, fall back to a secondary provider, surface a soft error) and monitor.
Avoid aggressive retries that worsen the upstream. The agent recommends this and
cites this section; a human applies it.

## Escalate if
- There is no degraded path and the provider is critical (e.g. all payments failing).
- The outage crosses into SEV1 for your users.
