"""
backend/app.py
===============
Production FastAPI application for TIDAL tampered image detection.

Endpoints:
  GET  /health    — Liveness probe (Docker/K8s)
  GET  /ready     — Readiness probe (model loaded?)
  POST /infer     — Upload image → tamper detection result + mask
  GET  /metrics   — Prometheus metrics
  GET  /version   — API and model version info
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from .inference.engine import TIDALInferenceEngine
from .inference.model_loader import ModelLoader
from .security import RateLimiter, validate_file_size, validate_file_type, validate_image_dimensions

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("tidal.api")

# ── Configuration ──────────────────────────────────────────────────────────
API_VERSION = "1.0.0"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "vR.P.19")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "4"))

# ── Prometheus Metrics ─────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "tidal_requests_total",
    "Total inference requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "tidal_request_duration_seconds",
    "Inference request latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
ACTIVE_REQUESTS = Counter(
    "tidal_active_requests",
    "Currently processing requests",
)

# ── Globals ────────────────────────────────────────────────────────────────
engine: TIDALInferenceEngine | None = None
rate_limiter = RateLimiter()
_semaphore: asyncio.Semaphore | None = None
_start_time: float = 0.0


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model + warm up. Shutdown: release GPU memory."""
    global engine, _semaphore, _start_time

    _start_time = time.time()
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    logger.info("Starting TIDAL API v%s", API_VERSION)

    try:
        engine = TIDALInferenceEngine()
        engine.warm_up()
        logger.info("Model loaded and engine warmed up")
    except FileNotFoundError:
        logger.warning(
            "Model checkpoint not found — API starts in degraded mode. "
            "Upload a model to models/best_model.pt and restart."
        )
        engine = None

    yield

    # Shutdown
    logger.info("Shutting down — releasing resources")
    loader = ModelLoader.get_instance()
    loader.unload()


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TIDAL API",
    description="Tampered Image Detection And Localization",
    version=API_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


# ── Error Handlers ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: never leak stack traces in production."""
    logger.error("Unhandled error: %s\n%s", exc, traceback.format_exc())
    REQUEST_COUNT.labels(status="error").inc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "alive", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


@app.get("/ready")
async def ready():
    """Readiness probe — returns 200 only when the model is loaded."""
    if engine is None or not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Model not loaded"},
        )
    return {"status": "ready", "model": MODEL_VERSION}


@app.post("/infer")
async def infer(request: Request, file: UploadFile = File(...)):
    """Run tamper detection on an uploaded image.

    Returns JSON with:
      - is_tampered: bool
      - confidence: float (0-1)
      - tampered_ratio: float (fraction of tampered pixels)
      - mask_base64: base64-encoded PNG of the binary mask
      - inference_time_ms: float
    """
    # Security checks
    rate_limiter.check(request)

    if engine is None:
        raise HTTPException(503, "Model not loaded — API in degraded mode")

    # Validate file
    validate_file_type(file.content_type, file.filename)

    # Read file bytes
    file_bytes = await file.read()
    validate_file_size(len(file_bytes))

    # Open and validate image
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file — could not decode")

    validate_image_dimensions(image.width, image.height)

    # Concurrency control
    assert _semaphore is not None
    async with _semaphore:
        t0 = time.perf_counter()

        # Run inference in thread pool (GPU ops are blocking)
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, engine.predict, image)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.error("GPU OOM during inference")
            REQUEST_COUNT.labels(status="oom").inc()
            raise HTTPException(503, "GPU out of memory — try a smaller image")
        except Exception as exc:
            logger.error("Inference failed: %s", exc)
            REQUEST_COUNT.labels(status="error").inc()
            raise HTTPException(500, "Inference failed")

        elapsed = time.perf_counter() - t0
        REQUEST_LATENCY.observe(elapsed)
        REQUEST_COUNT.labels(status="success").inc()

    # Encode mask as base64 PNG
    mask_img = Image.fromarray((result.mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        **result.to_dict(),
        "mask_base64": mask_b64,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/version")
async def version():
    """API and model version info."""
    loader = ModelLoader.get_instance()
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "model_loaded": loader.is_loaded,
        "checkpoint_hash": loader.checkpoint_hash,
        "device": str(loader.device),
        "cuda_available": torch.cuda.is_available(),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }
