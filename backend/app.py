"""FastAPI application for TIDAL tampered image detection."""

from __future__ import annotations

# Load .env before any other imports so that HF_TOKEN and other secrets
# are available via os.environ throughout the application.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)  # override=False: real env vars always win
except ImportError:
    pass  # python-dotenv is optional; env vars can be set externally

import asyncio
import base64
import hashlib
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response as StarletteResponse

from .cache import InferenceCache
from .inference.engine import (
    DEFAULT_MASK_AREA_THRESHOLD,
    DEFAULT_MIN_PREDICTION_AREA_PIXELS,
    DEFAULT_PIXEL_THRESHOLD,
    DEFAULT_REVIEW_CONFIDENCE_THRESHOLD,
    DEFAULT_THRESHOLD_SENSITIVITY_PRESET,
    MAX_AREA_PIXELS,
    MAX_PIXEL_THRESHOLD,
    MAX_REVIEW_CONFIDENCE_THRESHOLD,
    MIN_PIXEL_THRESHOLD,
    MIN_REVIEW_CONFIDENCE_THRESHOLD,
    InferenceSettings,
    TIDALInferenceEngine,
)
from .inference.model_loader import ModelLoader
from .report import build_report
from .security import RateLimiter, validate_file_size, validate_file_type, validate_image_dimensions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("tidal.api")
API_VERSION = "1.0.0"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "vR.P.30.1")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "4"))
REQUEST_COUNT = Counter("tidal_requests_total", "Total requests", ["status"])
REQUEST_LATENCY = Histogram(
    "tidal_request_duration_seconds", "Latency", buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30]
)
engine = None
rate_limiter = RateLimiter()
_semaphore = None
_start_time = 0.0
_cache = InferenceCache(max_size=int(os.environ.get("CACHE_MAX_SIZE", "128")))
BATCH_MAX_FILES = int(os.environ.get("BATCH_MAX_FILES", "8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, _semaphore, _start_time
    _start_time = time.time()
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    try:
        engine = TIDALInferenceEngine()
        engine.warm_up()
    except FileNotFoundError:
        logger.warning("Model not found - API in degraded mode")
        engine = None
    yield
    ModelLoader.get_instance().unload()


app = FastAPI(
    title="TIDAL API",
    description="Tampered Image Detection And Localization",
    version=API_VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Error: %s", exc)
    REQUEST_COUNT.labels(status="error").inc()
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "alive", "timestamp": datetime.now(tz=timezone.utc).isoformat()}


@app.get("/ready")
async def ready():
    if engine is None or not engine.is_ready:
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return {"status": "ready", "model": ModelLoader.get_instance().manifest.get("model_version", MODEL_VERSION)}


@app.post("/infer")
async def infer(
    request: Request,
    file: UploadFile = File(...),  # noqa: B008
    pixel_threshold: float | None = Form(
        default=None, ge=MIN_PIXEL_THRESHOLD, le=MAX_PIXEL_THRESHOLD
    ),
    mask_area_threshold: int | None = Form(default=None, ge=0, le=MAX_AREA_PIXELS),
    min_prediction_area_pixels: int | None = Form(default=None, ge=0, le=MAX_AREA_PIXELS),
    review_confidence_threshold: float | None = Form(
        default=None, ge=MIN_REVIEW_CONFIDENCE_THRESHOLD, le=MAX_REVIEW_CONFIDENCE_THRESHOLD
    ),
    threshold_sensitivity_preset: Literal["lenient", "balanced", "strict"] | None = Form(
        default=None
    ),
):
    rate_limiter.check(request)
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    validate_file_type(file.content_type, file.filename)
    file_bytes = await file.read()
    validate_file_size(len(file_bytes))
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, "Invalid image file") from exc
    validate_image_dimensions(image.width, image.height)
    cache_key = hashlib.sha256(file_bytes).hexdigest()
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.info("Cache HIT for %s", cache_key[:16])
        return {**cached, "cache_hit": True}

    async with _semaphore:
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        settings = InferenceSettings(
            pixel_threshold=(
                pixel_threshold if pixel_threshold is not None else DEFAULT_PIXEL_THRESHOLD
            ),
            mask_area_threshold=(
                mask_area_threshold
                if mask_area_threshold is not None
                else DEFAULT_MASK_AREA_THRESHOLD
            ),
            min_prediction_area_pixels=(
                min_prediction_area_pixels
                if min_prediction_area_pixels is not None
                else DEFAULT_MIN_PREDICTION_AREA_PIXELS
            ),
            review_confidence_threshold=(
                review_confidence_threshold
                if review_confidence_threshold is not None
                else DEFAULT_REVIEW_CONFIDENCE_THRESHOLD
            ),
            threshold_sensitivity_preset=(
                threshold_sensitivity_preset
                if threshold_sensitivity_preset is not None
                else DEFAULT_THRESHOLD_SENSITIVITY_PRESET
            ),
        )
        try:
            result = await loop.run_in_executor(
                None, engine.predict, image, settings, file_bytes
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            REQUEST_COUNT.labels(status="oom").inc()
            raise HTTPException(503, "GPU out of memory") from exc
        except Exception as exc:
            REQUEST_COUNT.labels(status="error").inc()
            raise HTTPException(500, "Inference failed") from exc
        REQUEST_LATENCY.observe(time.perf_counter() - t0)
        REQUEST_COUNT.labels(status="success").inc()
    mask_img = Image.fromarray((result.mask * 255).astype(np.uint8), mode="L")
    mask_buf = io.BytesIO()
    mask_img.save(mask_buf, format="PNG")

    overlay_base64 = ""
    if result.overlay is not None:
        overlay_img = Image.fromarray(result.overlay, mode="RGB")
        overlay_buf = io.BytesIO()
        overlay_img.save(overlay_buf, format="PNG")
        overlay_base64 = base64.b64encode(overlay_buf.getvalue()).decode("ascii")

    # Feature 2: probability heatmap (jet colormap, blue→green→red)
    heatmap_rgba = plt.cm.jet(result.prob_map)  # float32 RGBA in [0,1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb, mode="RGB")
    heatmap_buf = io.BytesIO()
    heatmap_img.save(heatmap_buf, format="PNG")
    heatmap_base64 = base64.b64encode(heatmap_buf.getvalue()).decode("ascii")

    response_data = {
        **result.to_dict(),
        "mask_base64": base64.b64encode(mask_buf.getvalue()).decode("ascii"),
        "overlay_base64": overlay_base64,
        "heatmap_base64": heatmap_base64,
        "cache_key": cache_key,
        "cache_hit": False,
    }
    _cache.set(cache_key, response_data)
    return response_data


@app.get("/metrics")
async def metrics():
    return StarletteResponse(
        content=generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.get("/version")
async def version():
    loader = ModelLoader.get_instance()
    return {
        "api_version": API_VERSION,
        "model_version": loader.manifest.get("model_version", MODEL_VERSION),
        "model_loaded": loader.is_loaded,
        "checkpoint_hash": loader.checkpoint_hash,
        "model_dir": str(loader.model_dir),
        "model_filename": loader.model_filename,
        "model_bundle_path": str(loader.model_dir / loader.model_filename),
        "device": str(loader.device),
        "cuda_available": torch.cuda.is_available(),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


# ── Feature 4: Cache stats ─────────────────────────────────────────────────

@app.get("/cache/stats")
async def cache_stats():
    """Return current inference result cache statistics."""
    return _cache.stats()


# ── Feature 5: PDF Report ──────────────────────────────────────────────────

@app.get("/report/{cache_key}")
async def report(cache_key: str):
    """Download a forensic PDF report for a previously inferred image.

    The image must have been submitted to /infer first (its cache_key is
    returned in the /infer response).  Returns 404 if the key has been
    evicted or was never inferred.
    """
    cached = _cache.get(cache_key)
    if cached is None:
        raise HTTPException(
            404,
            "Result not found in cache. Submit the image to /infer first, "
            "then use the returned cache_key here.",
        )
    try:
        pdf_bytes = build_report(cache_key, cached)
    except Exception as exc:
        logger.error("Report generation failed: %s", exc)
        raise HTTPException(500, "Failed to generate report") from exc
    return StarletteResponse(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="tidal_report_{cache_key[:12]}.pdf"'
        },
    )


# ── Feature 6: Batch inference ─────────────────────────────────────────────

@app.post("/infer/batch")
async def infer_batch(
    request: Request,
    files: list[UploadFile] = File(...),  # noqa: B008
):
    """Run inference on multiple images in a single request.

    Accepts up to BATCH_MAX_FILES images.  Each image is validated and
    inferred independently.  Per-image failures are reported inline rather
    than aborting the whole batch.

    Returns::

        {
          "results": [ ... ],   # one entry per file
          "total": N,
          "succeeded": M,
          "failed": K,
        }
    """
    rate_limiter.check(request)
    if engine is None:
        raise HTTPException(503, "Model not loaded")
    if not files:
        raise HTTPException(422, "No files provided")
    if len(files) > BATCH_MAX_FILES:
        raise HTTPException(
            422, f"Too many files. Maximum is {BATCH_MAX_FILES} per batch request."
        )

    results = []
    succeeded = 0
    failed = 0

    for upload in files:
        entry: dict = {"filename": upload.filename}
        try:
            validate_file_type(upload.content_type, upload.filename)
            file_bytes = await upload.read()
            validate_file_size(len(file_bytes))
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            validate_image_dimensions(image.width, image.height)

            # Cache lookup
            cache_key = hashlib.sha256(file_bytes).hexdigest()
            cached = _cache.get(cache_key)
            if cached is not None:
                entry.update({**cached, "cache_hit": True})
                succeeded += 1
                results.append(entry)
                continue

            async with _semaphore:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, engine.predict, image, None, file_bytes
                )

            mask_img = Image.fromarray((result.mask * 255).astype(np.uint8), mode="L")
            mask_buf = io.BytesIO()
            mask_img.save(mask_buf, format="PNG")

            overlay_base64 = ""
            if result.overlay is not None:
                overlay_img = Image.fromarray(result.overlay, mode="RGB")
                overlay_buf = io.BytesIO()
                overlay_img.save(overlay_buf, format="PNG")
                overlay_base64 = base64.b64encode(overlay_buf.getvalue()).decode("ascii")

            heatmap_rgba = plt.cm.jet(result.prob_map)
            heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
            heatmap_buf = io.BytesIO()
            Image.fromarray(heatmap_rgb, mode="RGB").save(heatmap_buf, format="PNG")
            heatmap_base64 = base64.b64encode(heatmap_buf.getvalue()).decode("ascii")

            result_data = {
                **result.to_dict(),
                "mask_base64": base64.b64encode(mask_buf.getvalue()).decode("ascii"),
                "overlay_base64": overlay_base64,
                "heatmap_base64": heatmap_base64,
                "cache_key": cache_key,
                "cache_hit": False,
            }
            _cache.set(cache_key, result_data)
            entry.update(result_data)
            succeeded += 1
            REQUEST_COUNT.labels(status="success").inc()
        except HTTPException as exc:
            entry["error"] = exc.detail
            failed += 1
            REQUEST_COUNT.labels(status="error").inc()
        except Exception as exc:
            entry["error"] = str(exc)
            failed += 1
            REQUEST_COUNT.labels(status="error").inc()
        results.append(entry)

    return {"results": results, "total": len(files), "succeeded": succeeded, "failed": failed}
