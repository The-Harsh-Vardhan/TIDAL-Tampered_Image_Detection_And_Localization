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

**Request:** `multipart/form-data` with:
- required `file`
- optional `pixel_threshold`
- optional `mask_area_threshold`
- optional `min_prediction_area_pixels`
- optional `review_confidence_threshold`
- optional `threshold_sensitivity_preset`

**Response:**
```json
{
  "is_tampered": true,
  "confidence": 0.8742,
  "confidence_mean_prob": 0.1031,
  "tampered_ratio": 0.1523,
  "raw_tampered_pixel_count": 2814,
  "tampered_pixel_count": 2814,
  "area_filter_triggered": false,
  "needs_review": false,
  "model_version": "vR.P.30.1",
  "applied_settings": {
    "pixel_threshold": 0.7,
    "mask_area_threshold": 400,
    "min_prediction_area_pixels": 0,
    "review_confidence_threshold": 0.65,
    "threshold_sensitivity_preset": "balanced",
    "threshold_sensitivity_levels": [0.3, 0.5, 0.7]
  },
  "threshold_sensitivity": [
    {
      "threshold": 0.3,
      "raw_pixels": 6310,
      "final_pixels": 6310,
      "ratio": 0.042793,
      "area_filtered": false
    }
  ],
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
| `MODEL_DIR` | `models/inference/vR.P.30.1` | Path to model checkpoint bundle |
| `MODEL_FILENAME` | `best_model.pt` | Checkpoint filename |
| `MODEL_VERSION` | `vR.P.30.1` | Model version label |
| `MAX_CONCURRENT` | `4` | Max concurrent inferences |
| `RATE_LIMIT_PER_MINUTE` | `30` | Rate limit per IP |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated) |
