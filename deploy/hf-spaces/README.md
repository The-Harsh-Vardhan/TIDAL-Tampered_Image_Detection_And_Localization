---
title: TIDAL — Tampered Image Detection & Localization
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# TIDAL API

Tampered Image Detection & Localization — FastAPI inference server.

Upload an image to detect and localize tampered regions using Multi-Quality RGB ELA + UNet segmentation.

**Pixel F1 = 0.7965** on CASIA 2.0

## Endpoints

- `POST /infer` — Upload image → tamper mask + verdict
- `GET /health` — Liveness check
- `GET /ready` — Model readiness
- `GET /version` — Version info
- `GET /metrics` — Prometheus metrics
