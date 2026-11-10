# TIDAL Architecture
## System: Docker Compose Stack
- tidal-api (FastAPI :8000) - /health, /ready, /infer, /metrics, /version
- tidal-frontend (Nginx :3000) - Static glassmorphism UI
- prometheus (:9090) - Metrics scraping
- grafana (:3001) - Dashboards

## Inference Pipeline
Input -> Resize 384x384 -> Multi-Q grayscale ELA (Q=75/85/95, 3ch) -> UNet+ResNet-34+CBAM -> Sigmoid -> Pixel threshold -> Minimum area filter -> Image decision threshold + review flag

## DVC Pipeline: preprocess -> train -> evaluate / visualize
