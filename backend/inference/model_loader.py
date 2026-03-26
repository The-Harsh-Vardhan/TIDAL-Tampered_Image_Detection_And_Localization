"""
backend/inference/model_loader.py
==================================
Singleton model loader with checkpoint validation.

Loads the UNet + ResNet-34 (9-channel input) model from a .pt checkpoint
and caches it for reuse across requests.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_model.pt")
DEVICE = os.environ.get("DEVICE", "auto")

# Model architecture constants (must match training config)
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None  # weights loaded from checkpoint, not ImageNet
IN_CHANNELS = 9
NUM_CLASSES = 1


def _resolve_device() -> torch.device:
    """Resolve device string to torch.device."""
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def _build_model() -> smp.Unet:
    """Build the UNet + ResNet-34 model structure (no weights)."""
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for integrity checking."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


class ModelLoader:
    """Thread-safe singleton model loader.

    Usage:
        loader = ModelLoader.get_instance()
        model = loader.model  # ready for inference
        device = loader.device
    """

    _instance: ModelLoader | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model: smp.Unet | None = None
        self._device: torch.device | None = None
        self._checkpoint_hash: str | None = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> ModelLoader:
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model(self) -> smp.Unet:
        if not self._loaded:
            self.load()
        assert self._model is not None
        return self._model

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = _resolve_device()
        return self._device

    @property
    def checkpoint_hash(self) -> str | None:
        return self._checkpoint_hash

    def load(self) -> None:
        """Load model checkpoint from disk."""
        checkpoint_path = Path(MODEL_DIR) / MODEL_FILENAME

        if not checkpoint_path.exists():
            msg = (
                f"Model checkpoint not found at {checkpoint_path}. "
                f"Set MODEL_DIR and MODEL_FILENAME env vars, or place "
                f"the checkpoint at models/best_model.pt"
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Loading model from %s", checkpoint_path)

        # Integrity check
        self._checkpoint_hash = _compute_sha256(checkpoint_path)
        logger.info("Checkpoint SHA-256: %s", self._checkpoint_hash[:16])

        # Build model and load weights
        self._device = _resolve_device()
        self._model = _build_model()

        checkpoint = torch.load(
            checkpoint_path, map_location=self._device, weights_only=False
        )

        # Handle both raw state_dict and wrapped checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self._model.load_state_dict(state_dict, strict=False)
        self._model.to(self._device)
        self._model.eval()
        self._loaded = True

        total_params = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "Model loaded: %s params on %s (hash: %s)",
            f"{total_params:,}",
            self._device,
            self._checkpoint_hash[:16],
        )

    def unload(self) -> None:
        """Release model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")
