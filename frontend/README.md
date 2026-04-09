# TIDAL Frontend

Static dark glassmorphism web UI for the TIDAL inference API.

## Quick Start

```bash
# Serve locally
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

## Usage

1. **Start the backend** first (`uvicorn backend.app:app --port 8000`)
2. Open `http://localhost:3000`
3. Drag & drop or browse for a JPEG/PNG/WebP image
4. Results appear automatically — verdict, confidence, tampered area %, heatmap mask, and notebook-style diagnostics

## Features

- Dark glassmorphism design with Inter typography
- Drag-and-drop or click-to-browse image upload
- Automatic API health polling with live status indicator
- Exposes forensic controls for pixel threshold, image area threshold, minimum prediction area, review confidence, and threshold sensitivity preset
- Displays tamper verdict, confidence score, tampered area %, binary heatmap, and diagnostic sensitivity data
- Responsive layout (mobile-friendly)
- Auto-submits on file select, and re-runs when a forensic control changes

## API Integration

Points to `http://localhost:8000` when running locally. For production, update `API_BASE` in `app.js` or serve via Nginx reverse proxy (see `docker/docker-compose.yml`).

## Files

| File | Purpose |
|------|---------|
| `index.html` | Page structure — hero, pipeline, upload, results |
| `styles.css` | Dark theme, glassmorphism cards, animations |
| `app.js` | API client, drag-drop, health polling, result rendering |
