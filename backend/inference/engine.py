"""
backend/inference/engine.py
============================
TIDAL Inference Engine — runs the full ELA → UNet pipeline.

Takes a PIL Image, computes Multi-Quality RGB ELA (9 channels),
feeds through the UNet+ResNet-34 model, and returns:
  - Binary tamper mask (H × W)
  - Image-level classification (authentic/tampered)
  - Confidence score
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from PIL import Image

from .model_loader import ModelLoader
from .preprocessing import DEFAULT_IMAGE_SIZE, preprocess_image

logger = logging.getLogger(__name__)

# Classification threshold: if >THRESHOLD fraction of pixels
# are predicted as tampered, classify the image as tampered.
PIXEL_THRESHOLD = 0.5
CLASSIFICATION_AREA_THRESHOLD = 0.005  # 0.5% of pixels


class InferenceResult:
    """Container for inference output."""

    __slots__ = (
        "mask",
        "is_tampered",
        "confidence",
        "tampered_ratio",
        "inference_time_ms",
    )

    def __init__(
        self,
        mask: np.ndarray,
        is_tampered: bool,
        confidence: float,
        tampered_ratio: float,
        inference_time_ms: float,
    ) -> None:
        self.mask = mask
        self.is_tampered = is_tampered
        self.confidence = confidence
        self.tampered_ratio = tampered_ratio
        self.inference_time_ms = inference_time_ms

    def to_dict(self) -> dict:
        return {
            "is_tampered": self.is_tampered,
            "confidence": round(self.confidence, 4),
            "tampered_ratio": round(self.tampered_ratio, 4),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "mask_shape": list(self.mask.shape),
        }


class TIDALInferenceEngine:
    """Production inference engine for tampered image detection.

    Usage:
        engine = TIDALInferenceEngine()
        result = engine.predict(pil_image)
        print(result.is_tampered, result.confidence)
    """

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE) -> None:
        self.image_size = image_size
        self._loader = ModelLoader.get_instance()

    @property
    def is_ready(self) -> bool:
        return self._loader.is_loaded

    def warm_up(self) -> None:
        """Force model load and run a dummy inference to warm GPU caches."""
        if not self._loader.is_loaded:
            self._loader.load()
        # Dummy forward pass
        dummy = torch.randn(1, 9, self.image_size, self.image_size).to(
            self._loader.device
        )
        with torch.no_grad():
            self._loader.model(dummy)
        logger.info("Engine warmed up")

    @torch.no_grad()
    def predict(self, image: Image.Image) -> InferenceResult:
        """Run tamper detection on a single image.

        Args:
            image: Input PIL Image (any size, will be resized).

        Returns:
            InferenceResult with mask, classification, and timing.
        """
        t0 = time.perf_counter()

        model = self._loader.model
        device = self._loader.device

        # Preprocess: PIL → 9-channel ELA tensor
        tensor = preprocess_image(image, size=self.image_size)
        tensor = tensor.to(device)

        # Forward pass
        logits = model(tensor)  # (1, 1, H, W)
        probs = torch.sigmoid(logits.float()).squeeze(0).squeeze(0)  # (H, W)

        # Binary mask
        mask = (probs > PIXEL_THRESHOLD).cpu().numpy().astype(np.uint8)
        prob_map = probs.cpu().numpy()

        # Image-level classification
        tampered_ratio = float(mask.sum()) / mask.size
        is_tampered = tampered_ratio > CLASSIFICATION_AREA_THRESHOLD

        # Confidence: mean probability in tampered regions, or
        # (1 - mean_prob) for authentic images
        if is_tampered:
            tampered_probs = prob_map[mask == 1]
            confidence = float(tampered_probs.mean()) if len(tampered_probs) > 0 else 0.5
        else:
            confidence = 1.0 - float(prob_map.mean())

        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Inference: tampered=%s confidence=%.3f ratio=%.4f time=%.1fms",
            is_tampered,
            confidence,
            tampered_ratio,
            elapsed_ms,
        )

        return InferenceResult(
            mask=mask,
            is_tampered=is_tampered,
            confidence=confidence,
            tampered_ratio=tampered_ratio,
            inference_time_ms=elapsed_ms,
        )
