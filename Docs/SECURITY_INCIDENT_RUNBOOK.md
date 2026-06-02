# Security Incident Runbook (Vercel, GitHub, npm)

This runbook is for emergency response when deployment providers or credential stores may be compromised.

---

## Severity and Trigger

Use this runbook when any of the following occurs:
- Provider confirms a breach affecting credentials or internal systems.
- You detect unknown deploy actors, unknown tokens, or suspicious release activity.
- Private keys or signing keys may have been exposed.

Default severity: **SEV-1** until disproven.

---

## 0-30 Minutes: Containment

1. Freeze production deploys.
   - Pause auto-deploy in Vercel.
   - Keep preview deploys only if needed for debugging.
2. Restrict merge/release paths.
   - Require green CI and branch protection on `production`.
3. Snapshot evidence.
   - Export active Vercel tokens and integration list.
   - Record recent deployment IDs and actors.
   - Record npm/GitHub token inventories.

---

## 30-90 Minutes: Credential Rotation

1. Rotate Vercel credentials.
   - Revoke old tokens (personal and team).
   - Create new least-privilege tokens.
2. Rotate GitHub credentials.
   - Revoke PATs/deploy keys used in CI/CD.
   - Re-issue only required tokens with narrow scopes.
3. Rotate npm publish credentials.
   - Revoke automation tokens.
   - Re-issue scoped automation tokens.
   - Enforce 2FA for publish-capable maintainers.
4. If private keys may be exposed.
   - Generate new keys.
   - Migrate funds/workloads/accounts immediately.

---

## 90-180 Minutes: Integrity and Secret Audit

1. Audit Vercel environment variables.
   - Remove unknown or stale entries.
   - Re-add only rotated secrets.
2. Run local repository checks.

```bash
pytest tests/ -v
pip-audit --strict
ruff check .
python scripts/security/scan_secrets.py --root .
python scripts/security/deploy_freeze_checklist.py --strict
```

3. Review recent dependency and lockfile changes.
   - Look for unexpected install/build scripts.
   - Validate lockfile integrity changes are expected.

---

## Recovery: Controlled Re-Enable

1. Perform one controlled production redeploy.
2. Verify backend endpoints:

```bash
curl https://YOUR_API/health
curl https://YOUR_API/ready
curl https://YOUR_API/version
curl https://YOUR_API/metrics
```

3. Monitor alerts for at least 30 minutes.
   - Error-rate spikes.
   - API availability drops.
4. Re-enable auto-deploy only after checks pass.

---

## Hardening After Recovery

1. Keep environment-based URLs and secrets only in env vars.
2. Avoid hardcoded production hostnames in app metadata or config.
3. Periodically rotate tokens and remove stale service accounts.
4. Keep incident notes: timeline, actions, impacted assets, lessons learned.

---

## Evidence Checklist

- [ ] Deploy freeze timestamp and owner recorded
- [ ] Token inventories captured before revocation
- [ ] All high-risk tokens rotated
- [ ] Key migration completed (if applicable)
- [ ] Security and test checks passed
- [ ] Controlled redeploy validated
- [ ] Monitoring window completed
- [ ] Post-incident actions tracked
