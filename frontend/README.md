# TIDAL Frontend

Next.js App Router frontend for the TIDAL inference API.

## Quick Start

Install dependencies inside Docker to preserve the repo's package-age gate:

```bash
docker run --rm -it \
  -v "${PWD}:/workspace" \
  -w /workspace/frontend \
  node:22-bookworm \
  bash -lc "corepack enable && pnpm install"
```

Run the development server:

```bash
cd frontend
pnpm dev
```

Open `http://localhost:3000`.

## Usage

1. Start the backend first: `uvicorn backend.app:app --port 8000`
2. Open `http://localhost:3000`
3. Drag and drop or browse for a JPEG, PNG, or WebP image
4. Adjust the forensic controls to rerun the notebook-style thresholds
5. Review the comparison gallery, heatmap, summary meters, and diagnostics

## Features

- Next.js App Router single-page experience that preserves the original TIDAL layout
- Direct browser-to-HF inference flow with local fallback to `http://localhost:8000`
- Vercel Analytics pageviews plus custom interaction telemetry
- Drag-and-drop upload, notebook-style threshold controls, comparison views, heatmap toggle, and forensic summary cards
- Responsive glassmorphism UI with self-hosted `next/font` typography

## Files

| Path | Purpose |
|------|---------|
| `app/layout.jsx` | Metadata, fonts, global CSS, Vercel Analytics |
| `app/page.jsx` | App Router entrypoint |
| `app/globals.css` | Global styles migrated from the original static site |
| `components/` | UI sections and reusable presentation components |
| `hooks/use-tidal-forensics.js` | Upload, inference, analytics, and derived-image state |
| `lib/analytics.js` | Safe custom event wrapper for Vercel Analytics |
