# TIDAL Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                       │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │  tidal-frontend   │    │    tidal-api      │               │
│  │  (Nginx :3000)    │───▶│  (FastAPI :8000)  │               │
│  │                   │    │                   │               │
│  │  • index.html     │    │  • /health        │               │
│  │  • app.js         │    │  • /ready         │               │
│  │  • styles.css     │    │  • /infer         │               │
│  └──────────────────┘    │  • /metrics       │               │
│                          │  • /version       │               │
│                          └────────┬──────────┘               │
│                                   │                          │
│                          ┌────────▼──────────┐               │
│                          │ TIDALInferenceEngine│              │
│                          │  ┌───────────────┐ │               │
│                          │  │ Preprocessing │ │               │
│                          │  │ Multi-Q ELA   │ │               │
│                          │  │ Q=75,85,95    │ │               │
│                          │  │ → 9 channels  │ │               │
│                          │  └───────┬───────┘ │               │
│                          │  ┌───────▼───────┐ │               │
│                          │  │ UNet+ResNet34  │ │               │
│                          │  │ (best_model.pt)│ │               │
│                          │  └───────┬───────┘ │               │
│                          │  ┌───────▼───────┐ │               │
│                          │  │ Binary Mask   │ │               │
│                          │  │ + Classify    │ │               │
│                          │  └───────────────┘ │               │
│                          └───────────────────┘               │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │   Prometheus      │◀───│   /metrics       │               │
│  │   (:9090)         │    │   (scrape 15s)   │               │
│  └────────┬─────────┘    └──────────────────┘               │
│           │                                                  │
│  ┌────────▼─────────┐                                       │
│  │    Grafana        │                                       │
│  │    (:3001)        │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

## Inference Pipeline

```
Input (any size) → Resize 384×384 → Multi-Q RGB ELA (9ch) → UNet → Sigmoid → Binary Mask + Classify
```

1. **Input**: JPEG/PNG/WebP image uploaded via `/infer`
2. **ELA**: Recompressed at Q=75, 85, 95 → pixel-level difference → 9 channels
3. **Normalization**: Standardized with precomputed channel mean/std from training set
4. **Model**: UNet + ResNet-34 encoder (body frozen, BN unfrozen, conv1 unfrozen for 9ch)
5. **Output**: Probability map → binary mask (threshold=0.5) + image-level classification

## DVC Training Pipeline

```
preprocess → train → evaluate
                   ↘ visualize
```

All hyperparameters tracked in `dvc_pipeline/params.yaml`. Reproducible with `dvc repro`.

## Security Architecture

| Layer | Protection |
|-------|-----------|
| Network | CORS whitelist, rate limiting |
| Input | File type + size + resolution validation |
| Docker | Non-root user, multi-stage build |
| CI/CD | pip-audit, ruff bandit rules |
| Model | SHA-256 checkpoint integrity |
