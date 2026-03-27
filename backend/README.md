# TIDAL Backend API

FastAPI-based inference server for Tampered Image Detection & Localization.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe — always 200 if alive |
| `GET` | `/ready` | Readiness probe — 200 only when model loaded |
| `POST` | `/infer` | Upload image → tamper detection result + mask |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/version` | API and model version info |

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run dev server
DEVICE=cpu python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## POST /infer

**Request:** `multipart/form-data` with `file` field (JPEG, PNG, or WebP).

**Response:**
```json
{
  "is_tampered": true,
  "confidence": 0.8742,
  "tampered_ratio": 0.1523,
  "inference_time_ms": 234.5,
  "mask_shape": [384, 384],
  "mask_base64": "iVBORw0KGgoAAAANS..."
}
```

## Security

- Rate limiting (30 req/min per IP)
- File type whitelist (JPEG/PNG/WebP)
- Max file size: 20 MB
- Max image pixels: 16M (4000×4000)
- No stack traces in production
- CORS configurable via `CORS_ORIGINS` env var

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `MODEL_DIR` | `models` | Path to model checkpoints |
| `MODEL_FILENAME` | `best_model.pt` | Checkpoint filename |
| `MODEL_VERSION` | `vR.P.19_U` | Model version label |
| `MAX_CONCURRENT` | `4` | Max concurrent inferences |
| `RATE_LIMIT_PER_MINUTE` | `30` | Rate limit per IP |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated) |
