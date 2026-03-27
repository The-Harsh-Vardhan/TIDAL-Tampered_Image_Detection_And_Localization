"""TIDAL Inference Engine - ELA to UNet pipeline."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from PIL import Image

from .model_loader import ModelLoader
from .preprocessing import DEFAULT_IMAGE_SIZE, preprocess_image

logger = logging.getLogger(__name__)
PIXEL_THRESHOLD = 0.5
CLASSIFICATION_AREA_THRESHOLD = 0.005


class InferenceResult:
    __slots__ = ("confidence", "inference_time_ms", "is_tampered", "mask", "tampered_ratio")

    def __init__(self, mask, is_tampered, confidence, tampered_ratio, inference_time_ms):
        self.mask = mask
        self.is_tampered = is_tampered
        self.confidence = confidence
        self.tampered_ratio = tampered_ratio
        self.inference_time_ms = inference_time_ms

    def to_dict(self):
        return {
            "is_tampered": self.is_tampered,
            "confidence": round(self.confidence, 4),
            "tampered_ratio": round(self.tampered_ratio, 4),
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
        dummy = torch.randn(1, 9, self.image_size, self.image_size).to(self._loader.device)
        with torch.no_grad():
            self._loader.model(dummy)
        logger.info("Engine warmed up")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> InferenceResult:
        t0 = time.perf_counter()
        model, device = self._loader.model, self._loader.device
        tensor = preprocess_image(image, size=self.image_size).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits.float()).squeeze(0).squeeze(0)
        mask = (probs > PIXEL_THRESHOLD).cpu().numpy().astype(np.uint8)
        prob_map = probs.cpu().numpy()
        tampered_ratio = float(mask.sum()) / mask.size
        is_tampered = tampered_ratio > CLASSIFICATION_AREA_THRESHOLD
        confidence = (
            float(prob_map[mask == 1].mean())
            if is_tampered and mask.sum() > 0
            else 1.0 - float(prob_map.mean())
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return InferenceResult(
            mask=mask,
            is_tampered=is_tampered,
            confidence=confidence,
            tampered_ratio=tampered_ratio,
            inference_time_ms=elapsed_ms,
        )
