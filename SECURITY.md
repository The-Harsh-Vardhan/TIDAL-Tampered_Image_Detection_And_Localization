# Security Policy
## Reporting a Vulnerability
1. Do NOT open a public issue.
2. Email with subject "TIDAL Security Report"
3. Include: description, reproduction steps, potential impact.

## Security Measures
- Rate limiting (30 req/min per IP)
- File type whitelist (JPEG, PNG, WebP)
- File size limit (20 MB), pixel limit (16MP)
- Non-root Docker user, multi-stage build
- pip-audit + ruff bandit rules in CI
- SHA-256 model checkpoint integrity
- No PII collected, no uploaded images persisted

## Incident Response
- Use `docs/SECURITY_INCIDENT_RUNBOOK.md` for active incidents affecting deployment providers, credentials, or supply-chain integrity.
- Treat suspected credential exposure as compromised until rotation and validation are complete.
