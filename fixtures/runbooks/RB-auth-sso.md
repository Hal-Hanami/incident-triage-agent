# RB-auth-sso — 401 spike after an SSO / certificate change

**Applies to:** a spike in 401/authentication failures correlated with an identity
provider change — signing-cert rotation, clock skew, clientID/secret update. Type:
`auth_access`.

## First response (read-only)
1. Confirm the 401 spike start time lines up with an IdP / cert change.
2. Verify the new signing certificate chain is what the relying party expects
   (thumbprint / kid match).
3. Check for clock skew between IdP and services (token `nbf`/`exp` rejections).

## Recommended first action
**`verify_sso_cert_clock`** — verify the certificate chain and clock alignment;
the usual fix is to refresh the trusted signing key or correct skew. The agent
recommends the verification steps and cites this section; a human applies the fix.

## Escalate if
- The failure looks like credential compromise rather than a benign rotation —
  treat as a security incident and escalate to the security on-call.
