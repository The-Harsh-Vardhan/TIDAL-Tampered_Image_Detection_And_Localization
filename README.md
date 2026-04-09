# Tampered Image Detection & Localization

<p align="center">
  <img src="figures/logos/TIDAL Logo.png" alt="TIDAL Logo" width="420"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Pixel_F1-0.7965-brightgreen" alt="Pixel F1"/>
  <img src="https://img.shields.io/badge/Experiments-60+-blueviolet" alt="Experiments"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/W%26B-Tracked-yellow?logo=weightsandbiases&logoColor=black" alt="W&B"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

<p align="center">
  <b>
    <a href="https://wandb.ai/tampered-image-detection-and-localization/Tampered%20Image%20Detection%20%26%20Localization/reports/Tampered-Image-Detection-Localization--VmlldzoxNjIyMjMxNg?accessToken=35b8v807ums5jnxtg6z8wieul1ylpetxrv2x4n7k9tr39mwf79ngtqs8w6d6tuaa">W&B Dashboard</a>
    · <a href="submission/submission_report.md">Submission Report</a>
    · <a href="submission/final_notebook.ipynb">Final Notebook</a>
    · <a href="docs/ARCHITECTURE.md">Architecture</a>
  </b>
</p>

---

A production-ready deep learning system for detecting and localizing tampered regions in images, achieving **Pixel F1 = 0.7965** on the CASIA 2.0 dataset.

The current deployed inference API uses the notebook-derived **vR.P.30.1** forensic pipeline with analyst controls for thresholding and review triage. The offline experiment leaderboard still shows **vR.P.19** as the strongest benchmarked run.

Through 60+ controlled ablation experiments, one finding dominated: **input representation matters most**. Switching from raw RGB to Multi-Quality RGB ELA produced a **+34.19pp improvement in Pixel F1** — more than all architectural changes, attention mechanisms, and training strategies combined.

> **Keywords:** image forensics · ELA · UNet · ResNet-34 · CASIA 2.0 · ablation study · segmentation · FastAPI · DVC

---

## ✨ Try It

### 🔬 Run the Research Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization/blob/main/submission/final_notebook.ipynb)

### 🚀 Run the Production API

```bash
git clone https://github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization.git
cd TIDAL-Tampered_Image_Detection_And_Localization
git checkout production
pip install -e ".[dev]"

# Start the backend (CPU mode)
DEVICE=cpu uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Start the frontend (new terminal)
cd frontend && python -m http.server 3000
```

Open **http://localhost:3000** → drag & drop any image to scan for tampering.
Use the **Forensic Controls** panel to tighten or relax pixel-level and image-level decision thresholds.

---

## 📊 Overview

| | |
| --- | --- |
| **Task** | Pixel-level tamper localization + image-level detection |
| **Dataset** | CASIA 2.0 (12,614 images — 7,491 authentic + 5,123 tampered) |
| **Architecture** | UNet + ResNet-34 (ImageNet pretrained) via SMP |
| **Best Input** | Multi-Quality RGB ELA 9-channel (Q=75/85/95 × RGB) |
| **Best Result** | Pixel F1 = 0.7965, IoU = 0.6615, AUC = 0.9665 |
| **Live API Pipeline** | vR.P.30.1 — grayscale Multi-Q ELA + CBAM + forensic controls |
| **Experiments** | 60+ controlled ablation experiments across 4 research tracks |
| **Framework** | PyTorch + Segmentation Models PyTorch (SMP) |
| **Tracking** | Weights & Biases |

---

## 🔬 Pipeline

```text
Input Image (any size)
        │
        ▼
   Resize to 384×384 RGB
        │
        ▼
   Production API:
   Multi-Q grayscale ELA
   Q=75, Q=85, Q=95 = 3 channels
        │
        ▼
   UNet + ResNet-34 Encoder
   + CBAM decoder attention
        │
        ▼
   Sigmoid → Binary Mask (384×384)
   + pixel threshold
   + minimum-area filter
   + image classification / review triage
```

**Error Level Analysis (ELA)** re-saves an image as JPEG at a given quality level and measures the difference from the original. Tampered regions show inconsistent compression artifacts. Using three quality levels (75, 85, 95) captures different compression frequency bands. The live API uses grayscale multi-quality ELA to match the `vR.P.30.1` notebook checkpoint.

---

## 📈 Key Results

### Best Run: vR.P.19

| Metric | Value |
| --- | --- |
| Pixel F1 | **0.7965** |
| IoU (Jaccard) | 0.6615 |
| Pixel AUC | 0.9665 |

### Top 5 Runs

| Rank | Version | Pixel F1 | Key Configuration |
| --- | --- | --- | --- |
| 1 | **vR.P.19** | **0.7965** | Multi-Q RGB ELA 9ch, 25 epochs |
| 2 | vR.P.30.1 | 0.7762 | Multi-Q ELA + CBAM, 50 epochs |
| 3 | vR.P.30.4 | 0.7745 | Multi-Q ELA + CBAM + augmentation |
| 4 | vR.P.30.2 | 0.7721 | Multi-Q ELA + CBAM + unfreeze |
| 5 | vR.P.30 | 0.7714 | Multi-Q ELA + CBAM, 25 epochs |

### Key Finding

> **Input preprocessing is the single most impactful variable.** Switching from raw RGB to Multi-Quality RGB ELA improved Pixel F1 by **+34.19 percentage points** — more than any architectural change.

---

## 📸 Visual Results

### Multi-Metric Comparison

<p align="center">
  <img src="figures/plots/radar_comparison.png" alt="Radar chart comparing baseline vs Multi-Q RGB ELA across 5 metrics" width="520"/>
</p>

### Ablation Progression

<p align="center">
  <img src="figures/plots/ablation_progression.png" alt="Pixel F1 and IoU across all experiments"/>
</p>

### Feature Set Comparison

<p align="center">
  <img src="figures/plots/feature_set_comparison.png" alt="Pixel F1 and IoU by input representation"/>
</p>

---

## 🏗️ Architecture

```text
┌────────────────┐      ┌──────────────────┐
│  Frontend      │──────│  FastAPI Backend  │
│  (Nginx :3000) │ POST │  (Uvicorn :8000)  │
│  index.html    │/infer│                   │
└────────────────┘      │  /health /ready   │
                        │  /infer /metrics  │
                        │  /version         │
                        └────────┬──────────┘
                                 │
                        ┌────────▼──────────┐
                        │ TIDALInference    │
                        │ Engine            │
                        │  ┌─────────────┐  │
                        │  │ Multi-Q ELA │  │
                        │  │ 9 channels  │  │
                        │  └──────┬──────┘  │
                        │  ┌──────▼──────┐  │
                        │  │ UNet        │  │
                        │  │ ResNet-34   │  │
                        │  └──────┬──────┘  │
                        │  ┌──────▼──────┐  │
                        │  │ Mask +      │  │
                        │  │ Classify    │  │
                        │  └─────────────┘  │
                        └───────────────────┘
```

### Security

| Layer | Protection |
| --- | --- |
| Network | CORS whitelist, rate limiting (30 req/min) |
| Input | File type + size (20MB) + resolution (16MP) validation |
| Docker | Non-root user, multi-stage build, read-only mounts |
| CI/CD | pip-audit, ruff bandit rules, local axios IOC triage script |
| Model | SHA-256 checkpoint integrity |

For local supply-chain verification, run `python scripts/security/check_axios_compromise.py`. The
runbook is in `Docs/SECURITY_AXIOS_IOC_RESPONSE.md`.

---

## 🚢 Deploy

### Hugging Face Spaces (Backend API)

```bash
# Copy deploy config to create a new HF Space (SDK: Docker)
cp deploy/hf-spaces/* .
# Push to your HF Space repo
```

Use **CPU Basic** (free) or **T4 GPU** (recommended for real-time inference).

### Vercel (Frontend)

```bash
# Option 1: Vercel CLI
cd frontend && vercel --prod

# Option 2: Connect GitHub repo
# Set root directory to "frontend/" in Vercel dashboard
```

Update `API_BASE` in `frontend/app.js` to point to your HF Spaces URL.

### Render (Backend API)

```bash
# Connect GitHub repo to Render
# Point to deploy/render/render.yaml as Blueprint
```

### Docker Compose (Full Stack — Local)

```bash
cd docker && docker compose up -d
```

| Service | URL |
| --- | --- |
| API | http://localhost:8000 |
| Frontend | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3001 |

---

## 🧪 Development

### Tests

```bash
pytest tests/ -v         # All tests (17)
ruff check backend/      # Lint
ruff format --check .    # Format check
```

### DVC Training Pipeline

```bash
cd dvc_pipeline
dvc repro   # preprocess → train → evaluate → visualize
```

All hyperparameters in `dvc_pipeline/params.yaml`. Reproducible with seed=42.

### CI/CD

GitHub Actions on every push/PR to `main` / `production`:
- **code_quality.yml** — Ruff lint + pytest + pip-audit
- **docker_build.yml** — Docker build with BuildKit caching

---

## 📁 Repository Structure

```text
TIDAL/
├── backend/                    # FastAPI inference server
│   ├── app.py                  # 5 endpoints + CORS + Prometheus
│   ├── security.py             # Rate limiting + input validation
│   └── inference/              # Engine, model loader, ELA preprocessing
├── dvc_pipeline/               # Reproducible training pipeline
│   ├── dvc.yaml                # 4-stage DAG
│   ├── params.yaml             # All hyperparameters
│   └── src/                    # Dataset, model, train, evaluate, visualize
├── frontend/                   # Dark glassmorphism web UI
│   ├── index.html              # Hero + pipeline + demo + results
│   ├── app.js                  # API client + drag-drop + health polling
│   └── styles.css              # Dark theme + animations
├── docker/                     # Production containerization
│   ├── Dockerfile              # Multi-stage CUDA 12.1 image
│   ├── Dockerfile.dev          # Hot-reload dev image
│   └── docker-compose.yml      # Full stack (4 services)
├── deploy/                     # Deployment configs
│   ├── hf-spaces/              # HF Spaces Docker config
│   ├── vercel/                 # Vercel static site config
│   └── render/                 # Render Blueprint
├── observability/              # Prometheus + Grafana + alerting
├── tests/                      # pytest suite (17 tests)
├── .github/workflows/          # CI/CD (lint + test + Docker)
├── figures/                    # Plots + logos for README
├── Notebooks/                  # Research notebooks (60+ experiments)
├── submission/                 # Assignment deliverables
├── docs/                       # Architecture + production readiness
├── pyproject.toml              # Project config + ruff rules
├── requirements-prod.txt       # Pinned production dependencies
├── LICENSE                     # MIT
├── CONTRIBUTING.md             # How to contribute
└── SECURITY.md                 # Vulnerability reporting
```

---

## 🔬 Research Methodology

- **Architecture:** UNet encoder-decoder via [SMP](https://github.com/qubvel/segmentation_models.pytorch), with ResNet-34 encoder (ImageNet pretrained)
- **Training:** BCE + Dice hybrid loss, AdamW optimizer, AMP, early stopping (patience=7)
- **Evaluation:** 70/15/15 train-val-test split with full held-out test metrics
- **Single-variable ablation:** Each experiment changes exactly one variable from its parent
- **W&B logging:** All metrics, configs, and checkpoints tracked

---

## Tech Stack

| Category | Technologies |
| --- | --- |
| **ML** | PyTorch 2.x · SMP · Albumentations · OpenCV |
| **Backend** | FastAPI · Uvicorn · Prometheus · Pydantic |
| **Frontend** | Vanilla HTML/CSS/JS · Glassmorphism design |
| **Infra** | Docker · Docker Compose · DVC · GitHub Actions |
| **Monitoring** | Prometheus · Grafana · Custom alerting rules |
| **Tracking** | Weights & Biases |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🔗 Links

- [W&B Dashboard](https://wandb.ai/tampered-image-detection-and-localization/Tampered%20Image%20Detection%20%26%20Localization/reports/Tampered-Image-Detection-Localization--VmlldzoxNjIyMjMxNg?accessToken=35b8v807ums5jnxtg6z8wieul1ylpetxrv2x4n7k9tr39mwf79ngtqs8w6d6tuaa)
- [Submission Report](submission/submission_report.md)
- [Final Notebook](submission/final_notebook.ipynb)
- [Architecture Guide](docs/ARCHITECTURE.md)
- [Production Readiness](docs/PRODUCTION_READINESS.md)
- [Project Lifecycle Tracker](Project_Lifecycle_Tracker.md)

---

<p align="center">
  Built by <a href="https://github.com/The-Harsh-Vardhan">Harsh Vardhan</a> · IIIT Nagpur
</p>
