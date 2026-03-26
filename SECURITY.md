# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public issue.
2. Email: [your-email@example.com] with the subject "TIDAL Security Report"
3. Include: description, reproduction steps, and potential impact.
4. We will respond within **48 hours** and aim to fix critical issues within **7 days**.

## Security Measures

### API
- Rate limiting (30 requests/minute per IP)
- File type whitelist (JPEG, PNG, WebP only)
- File size limit (20 MB)
- Image pixel count limit (16 megapixels)
- No stack traces leaked in production responses
- CORS origin whitelist

### Docker
- Non-root user (`tidal`)
- Multi-stage build (no build tools in runtime)
- Read-only model volume mounts
- Health check for orchestrator integration

### Dependencies
- `pip-audit` for vulnerability scanning
- Pinned versions in `requirements-prod.txt`
- Ruff with bandit security rules (`S` rules)

### Data
- No PII collected or stored
- Model checkpoint integrity verified via SHA-256
- No user-uploaded images are persisted
