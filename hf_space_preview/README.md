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

Upload an image to detect and localize tampered regions using the `vR.P.30.1` notebook pipeline:
- grayscale multi-quality ELA (`Q=75/85/95`)
- UNet + `resnet34`
- CBAM decoder attention
- analyst-facing threshold controls on `/infer`

This deployment keeps the legacy response fields for compatibility and adds notebook-style diagnostics such as:
- `raw_tampered_pixel_count`
- `tampered_pixel_count`
- `needs_review`
- `threshold_sensitivity`
- `applied_settings`
- `overlay_base64`

## Endpoints

- `POST /infer` — Upload image → tamper mask, overlay, and verdict
- `GET /health` — Liveness check
- `GET /ready` — Model readiness
- `GET /version` — Version info
- `GET /metrics` — Prometheus metrics
