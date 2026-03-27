# TIDAL Architecture
## System: Docker Compose Stack
- tidal-api (FastAPI :8000) - /health, /ready, /infer, /metrics, /version
- tidal-frontend (Nginx :3000) - Static glassmorphism UI
- prometheus (:9090) - Metrics scraping
- grafana (:3001) - Dashboards

## Inference Pipeline
Input -> Resize 384x384 -> Multi-Q RGB ELA (9ch) -> UNet+ResNet-34 -> Sigmoid -> Binary Mask + Classify

## DVC Pipeline: preprocess -> train -> evaluate / visualize
