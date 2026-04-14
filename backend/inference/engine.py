"""TIDAL inference engine for the vR.P.30.1 notebook pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .model_loader import ModelLoader
from .preprocessing import DEFAULT_IMAGE_SIZE, preprocess_image

logger = logging.getLogger(__name__)
MIN_PIXEL_THRESHOLD = 0.001
MAX_PIXEL_THRESHOLD = 0.95
MIN_REVIEW_CONFIDENCE_THRESHOLD = 0.05
MAX_REVIEW_CONFIDENCE_THRESHOLD = 0.95
DEFAULT_PIXEL_THRESHOLD = 0.70
DEFAULT_MASK_AREA_THRESHOLD = 400
DEFAULT_MIN_PREDICTION_AREA_PIXELS = 0
DEFAULT_REVIEW_CONFIDENCE_THRESHOLD = 0.65
MAX_AREA_PIXELS = DEFAULT_IMAGE_SIZE * DEFAULT_IMAGE_SIZE
THRESHOLD_SENSITIVITY_PRESETS = {
    "lenient": [0.20, 0.35, 0.50],
    "balanced": [0.30, 0.50, 0.70],
    "strict": [0.50, 0.70, 0.85],
}
DEFAULT_THRESHOLD_SENSITIVITY_PRESET = "balanced"


@dataclass(slots=True)
class InferenceSettings:
    pixel_threshold: float = DEFAULT_PIXEL_THRESHOLD
    mask_area_threshold: int = DEFAULT_MASK_AREA_THRESHOLD
    min_prediction_area_pixels: int = DEFAULT_MIN_PREDICTION_AREA_PIXELS
    review_confidence_threshold: float = DEFAULT_REVIEW_CONFIDENCE_THRESHOLD
    threshold_sensitivity_preset: str = DEFAULT_THRESHOLD_SENSITIVITY_PRESET

    @property
    def threshold_sensitivity_levels(self):
        return THRESHOLD_SENSITIVITY_PRESETS[self.threshold_sensitivity_preset]

    def to_dict(self):
        return {
            "pixel_threshold": round(self.pixel_threshold, 4),
            "mask_area_threshold": self.mask_area_threshold,
            "min_prediction_area_pixels": self.min_prediction_area_pixels,
            "review_confidence_threshold": round(self.review_confidence_threshold, 4),
            "threshold_sensitivity_preset": self.threshold_sensitivity_preset,
            "threshold_sensitivity_levels": self.threshold_sensitivity_levels,
        }


def apply_prediction_area_filter(pred_bin: np.ndarray, min_area_pixels: int = 0):
    raw_pixels = int(pred_bin.sum())
    if min_area_pixels > 0 and raw_pixels < min_area_pixels:
        return np.zeros_like(pred_bin, dtype=np.uint8), True
    return pred_bin, False


def compute_threshold_sensitivity(prob_map: np.ndarray, thresholds, min_area_pixels: int = 0):
    rows = []
    for threshold in thresholds:
        raw_bin = (prob_map > threshold).astype(np.uint8)
        raw_pixels = int(raw_bin.sum())
        final_bin, area_filtered = apply_prediction_area_filter(raw_bin, min_area_pixels)
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "raw_pixels": raw_pixels,
                "final_pixels": int(final_bin.sum()),
                "ratio": round(float(final_bin.mean()), 6),
                "area_filtered": bool(area_filtered),
            }
        )
    return rows


def is_tampered_prediction(tampered_pixel_count: int, mask_area_threshold: int) -> bool:
    if mask_area_threshold <= 0:
        return tampered_pixel_count > 0
    return tampered_pixel_count >= mask_area_threshold


def build_overlay_image(
    image: Image.Image,
    mask: np.ndarray,
    overlay_alpha: float = 0.42,
) -> np.ndarray | None:
    if int(mask.sum()) == 0:
        return None

    height, width = mask.shape
    resized_image = image.resize((width, height), Image.Resampling.BILINEAR)
    base_pixels = np.asarray(resized_image, dtype=np.float32)
    overlay_color = np.array([255.0, 59.0, 48.0], dtype=np.float32)
    alpha_mask = (mask.astype(np.float32) * overlay_alpha)[..., None]
    blended = (base_pixels * (1.0 - alpha_mask)) + (overlay_color * alpha_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


@dataclass(slots=True)
class InferenceResult:
    mask: np.ndarray
    overlay: np.ndarray | None
    is_tampered: bool
    confidence: float
    confidence_mean_prob: float
    tampered_ratio: float
    raw_tampered_pixel_count: int
    tampered_pixel_count: int
    area_filter_triggered: bool
    needs_review: bool
    threshold_sensitivity: list[dict]
    applied_settings: dict
    model_version: str
    inference_time_ms: float

    def to_dict(self):
        return {
            "is_tampered": self.is_tampered,
            "confidence": round(self.confidence, 4),
            "confidence_mean_prob": round(self.confidence_mean_prob, 4),
            "tampered_ratio": round(self.tampered_ratio, 4),
            "raw_tampered_pixel_count": self.raw_tampered_pixel_count,
            "tampered_pixel_count": self.tampered_pixel_count,
            "area_filter_triggered": self.area_filter_triggered,
            "needs_review": self.needs_review,
            "threshold_sensitivity": self.threshold_sensitivity,
            "applied_settings": self.applied_settings,
            "model_version": self.model_version,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "mask_shape": list(self.mask.shape),
        }


class TIDALInferenceEngine:
    def __init__(self, image_size=DEFAULT_IMAGE_SIZE):
        self.image_size = image_size
        self._loader = ModelLoader.get_instance()

    @property
    def is_ready(self):
        return self._loader.is_loaded

    def warm_up(self):
        if not self._loader.is_loaded:
            self._loader.load()
        dummy = torch.randn(1, 3, self.image_size, self.image_size).to(self._loader.device)
        with torch.no_grad():
            self._loader.model(dummy)
        logger.info("Engine warmed up")

    @torch.no_grad()
    def predict(self, image: Image.Image, settings: InferenceSettings | None = None) -> InferenceResult:
        if settings is None:
            settings = InferenceSettings()
        t0 = time.perf_counter()
        model, device = self._loader.model, self._loader.device
        tensor = preprocess_image(image, size=self.image_size).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits.float()).squeeze(0).squeeze(0)
        prob_map = probs.cpu().numpy()
        raw_mask = (prob_map > settings.pixel_threshold).astype(np.uint8)
        raw_tampered_pixel_count = int(raw_mask.sum())
        mask, area_filter_triggered = apply_prediction_area_filter(
            raw_mask, settings.min_prediction_area_pixels
        )
        tampered_pixel_count = int(mask.sum())
        tampered_ratio = float(mask.mean())
        is_tampered = is_tampered_prediction(tampered_pixel_count, settings.mask_area_threshold)
        confidence = float(prob_map.max())
        confidence_mean_prob = float(prob_map.mean())
        near_threshold = settings.mask_area_threshold > 0 and (
            0 < tampered_pixel_count < int(settings.mask_area_threshold * 1.25)
        )
        needs_review = bool(
            confidence < settings.review_confidence_threshold
            or near_threshold
            or area_filter_triggered
        )
        threshold_sensitivity = compute_threshold_sensitivity(
            prob_map,
            thresholds=settings.threshold_sensitivity_levels,
            min_area_pixels=settings.min_prediction_area_pixels,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return InferenceResult(
            mask=mask,
            overlay=build_overlay_image(image, mask),
            is_tampered=is_tampered,
            confidence=confidence,
            confidence_mean_prob=confidence_mean_prob,
            tampered_ratio=tampered_ratio,
            raw_tampered_pixel_count=raw_tampered_pixel_count,
            tampered_pixel_count=tampered_pixel_count,
            area_filter_triggered=area_filter_triggered,
            needs_review=needs_review,
            threshold_sensitivity=threshold_sensitivity,
            applied_settings=settings.to_dict(),
            model_version=self._loader.manifest.get("model_version", "vR.P.30.1"),
            inference_time_ms=elapsed_ms,
        )
