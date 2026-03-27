# TIDAL Deployment Guide

Step-by-step deployment for **Hugging Face Spaces** (backend), **Vercel** (frontend), and **Render** (alternative backend), with CI/CD.

---

## Prerequisites

- GitHub repo pushed to `production` branch ✅
- A [Hugging Face](https://huggingface.co) account (free)
- A [Vercel](https://vercel.com) account (free)
- A [Render](https://render.com) account (free)
- Your trained `best_model.pt` weights file (for live inference)

---

## 1. Hugging Face Spaces — Backend API

HF Spaces hosts your FastAPI backend with Docker, with optional GPU.

### Step 1.1: Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Owner:** your HF username
   - **Space name:** `tidal-api`
   - **License:** MIT
   - **SDK:** Select **Docker**
   - **Hardware:** `CPU Basic` (free) or `T4 GPU` ($0.60/hr, recommended for fast inference)
3. Click **Create Space**

### Step 1.2: Clone the Space repo locally

```bash
# Install git-lfs (needed for model weights)
git lfs install

# Clone your new Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/tidal-api
cd tidal-api
```

### Step 1.3: Copy files into the Space

```bash
# From your TIDAL project root:
cp deploy/hf-spaces/Dockerfile   ../tidal-api/Dockerfile
cp deploy/hf-spaces/README.md    ../tidal-api/README.md
cp requirements-prod.txt         ../tidal-api/requirements-prod.txt
cp -r backend/                   ../tidal-api/backend/

# Copy model weights (required for inference)
mkdir -p ../tidal-api/models
cp models/best_model.pt          ../tidal-api/models/best_model.pt
```

### Step 1.4: Push to HF

```bash
cd ../tidal-api
git add -A
git commit -m "feat: deploy TIDAL API"
git push
```

### Step 1.5: Verify

- Go to `https://huggingface.co/spaces/YOUR_USERNAME/tidal-api`
- Wait for the build (2-5 min)
- Test: `https://YOUR_USERNAME-tidal-api.hf.space/health` should return `{"status": "alive"}`
- Swagger docs: `https://YOUR_USERNAME-tidal-api.hf.space/docs`

> **Note your API URL:** `https://YOUR_USERNAME-tidal-api.hf.space` — you'll need this for the frontend.

---

## 2. Vercel — Frontend

Vercel hosts your static frontend (HTML/CSS/JS).

### Step 2.1: Update API URL in frontend

Before deploying, point the frontend to your live HF Spaces API:

Edit `frontend/app.js` — find this line near the top:

```javascript
const API = location.hostname === "localhost" || location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "";
```

Change it to:

```javascript
const API = location.hostname === "localhost" || location.hostname === "127.0.0.1"
    ? "http://localhost:8000"
    : "https://YOUR_USERNAME-tidal-api.hf.space";
```

Commit this change:

```bash
git add frontend/app.js
git commit -m "feat: point frontend to live HF Spaces API"
git push origin production
```

### Step 2.2: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy (from project root)
cd frontend
vercel

# Follow prompts:
#   - Set up and deploy? → Y
#   - Which scope? → your account
#   - Link to existing project? → N
#   - Project name? → tidal-frontend
#   - Directory containing code? → ./
#   - Override settings? → N

# Deploy to production
vercel --prod
```

### Step 2.3: Alternative — Deploy via GitHub (auto-deploy on push)

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **Import** → select your GitHub repo
3. Configure:
   - **Framework Preset:** `Other`
   - **Root Directory:** `frontend`
   - **Build Command:** (leave empty — it's static HTML)
   - **Output Directory:** `.`
4. Click **Deploy**

> Every push to `production` will auto-deploy.

### Step 2.4: Verify

- Open your Vercel URL (e.g., `tidal-frontend.vercel.app`)
- Check the status indicator shows "API connected" (green dot)
- Upload an image to test end-to-end

---

## 3. Render — Backend API (Alternative to HF Spaces)

Render is a simpler alternative if you don't need GPU.

### Step 3.1: Connect GitHub

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New** → **Blueprint**
3. Connect your GitHub repo
4. Point to: `deploy/render/render.yaml`
5. Click **Apply**

### Step 3.2: Alternative — Manual Web Service

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Configure:
   - **Name:** `tidal-api`
   - **Branch:** `production`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements-prod.txt`
   - **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `DEVICE` = `cpu`
   - `MODEL_DIR` = `/opt/render/project/src/models`
   - `CORS_ORIGINS` = `*`
   - `PYTHON_VERSION` = `3.11.0`
6. Click **Create Web Service**

### Step 3.3: Upload model weights

Render doesn't have persistent storage on free tier, so you need to either:
- **Option A:** Include `models/best_model.pt` in the repo (use Git LFS)
- **Option B:** Download from HF Hub on startup (add a download script)
- **Option C:** Run in degraded mode (API works but `/infer` returns 503)

### Step 3.4: Verify

- API URL: `https://tidal-api.onrender.com`
- Test: `https://tidal-api.onrender.com/health`
- Note: Free tier spins down after 15min of inactivity (50s cold start)

---

## 4. CI/CD — GitHub Actions

Your CI/CD is already configured. Here's how it works:

### What's already set up

```text
.github/workflows/
├── code_quality.yml    # Runs on every push/PR to main or production
│   ├── ruff check      # Linting
│   ├── ruff format     # Format check
│   ├── pytest          # 17 tests
│   └── pip-audit       # Security vulnerability scan
│
└── docker_build.yml    # Runs when backend/ or docker/ files change
    ├── Docker build     # Validates the image builds
    └── Compose config   # Validates docker-compose.yml
```

### How to trigger

CI runs automatically on:
- **Push** to `main` or `production`
- **Pull request** targeting `main` or `production`

### Adding auto-deploy to Vercel

Vercel auto-deploys when connected via GitHub (Step 2.3). No extra config needed.

### Adding auto-deploy to HF Spaces

Add this workflow to auto-sync your backend to HF Spaces on push:

```yaml
# Save as: .github/workflows/deploy_hf_spaces.yml

name: Deploy to HF Spaces

on:
  push:
    branches: [production]
    paths: [backend/**, requirements-prod.txt, deploy/hf-spaces/**]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Push to HF Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          # Build the HF Space directory
          mkdir -p hf_space/backend hf_space/models
          cp deploy/hf-spaces/Dockerfile hf_space/
          cp deploy/hf-spaces/README.md hf_space/
          cp requirements-prod.txt hf_space/
          cp -r backend/* hf_space/backend/

          # Push to HF
          cd hf_space
          git init
          git remote add space https://YOUR_USERNAME:$HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/tidal-api
          git add -A
          git commit -m "auto-deploy from GitHub Actions"
          git push space main --force
```

**Setup required:**
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a **Write** token
3. Go to your GitHub repo → Settings → Secrets → Actions
4. Add secret: `HF_TOKEN` = your HF token
5. Replace `YOUR_USERNAME` in the workflow file

---

## 5. Post-Deployment Checklist

| Step | Status |
| --- | --- |
| HF Spaces backend deployed and `/health` returns 200 | ☐ |
| Vercel frontend deployed and loads in browser | ☐ |
| Frontend → Backend connection works (green status indicator) | ☐ |
| Upload test image → get tamper verdict + mask | ☐ |
| GitHub Actions CI passes on push | ☐ |
| HF auto-deploy workflow added (optional) | ☐ |

---

## Quick Reference

| Platform | What | Free Tier | URL Pattern |
| --- | --- | --- | --- |
| **HF Spaces** | Backend API | CPU Basic (free), T4 ($0.60/hr) | `username-tidal-api.hf.space` |
| **Vercel** | Frontend | 100GB bandwidth/month | `tidal-frontend.vercel.app` |
| **Render** | Backend (alt) | 750 hrs/month, sleeps after 15min | `tidal-api.onrender.com` |
| **GitHub Actions** | CI/CD | 2000 min/month | Runs on push/PR |
