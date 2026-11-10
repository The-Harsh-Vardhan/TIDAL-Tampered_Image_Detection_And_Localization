# Setup

All commands come from the repository README and `backend/README.md`.

## Backend API

```bash
git clone https://github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization.git
cd TIDAL-Tampered_Image_Detection_And_Localization
git checkout production
pip install -e ".[dev]"

DEVICE=cpu uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

`POST /infer` takes `multipart/form-data` with a required `file` and optional `pixel_threshold` and `mask_area_threshold`.

## Frontend

```bash
docker run --rm -it \
  -v "${PWD}:/workspace" \
  -w /workspace/frontend \
  node:22-bookworm \
  bash -lc "corepack enable && pnpm install"

cd frontend && pnpm dev
```

Open http://localhost:3000. Set `NEXT_PUBLIC_API_BASE` in `frontend/.env.local` to point at a different backend.

## Tests, lint, DVC

```bash
pytest tests/ -v
ruff check backend/
ruff format --check .

cd dvc_pipeline
dvc repro      # preprocess -> train -> evaluate -> visualize
```

Hyperparameters are in `dvc_pipeline/params.yaml`.

## Docker Compose and deployment

```bash
cd docker && docker compose up -d
```

API on :8000, frontend on :3000, Prometheus on :9090, Grafana on :3001. For Hugging Face Spaces, Vercel and Render see the [deployment guide](https://github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization/blob/main/docs/DEPLOYMENT.md).

## Research notebook

The final notebook is `submission/vr-p-19-u-image-tampering-detection-and-localization.ipynb`; individual experiment notebooks are under `experiments/runs/` and `Notebooks/`.
