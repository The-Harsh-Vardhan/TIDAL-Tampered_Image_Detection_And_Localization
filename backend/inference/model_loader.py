"""Singleton model loader with checkpoint validation."""
from __future__ import annotations
import hashlib, logging, os, threading
from pathlib import Path
import segmentation_models_pytorch as smp
import torch

logger = logging.getLogger(__name__)
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_model.pt")
DEVICE = os.environ.get("DEVICE", "auto")

def _resolve_device():
    if DEVICE == "auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)

def _build_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=9, classes=1)

def _compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): sha.update(chunk)
    return sha.hexdigest()

class ModelLoader:
    _instance = None
    _lock = threading.Lock()
    def __init__(self):
        self._model = None; self._device = None; self._checkpoint_hash = None; self._loaded = False
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None: cls._instance = cls()
        return cls._instance
    @property
    def is_loaded(self): return self._loaded
    @property
    def model(self):
        if not self._loaded: self.load()
        return self._model
    @property
    def device(self):
        if self._device is None: self._device = _resolve_device()
        return self._device
    @property
    def checkpoint_hash(self): return self._checkpoint_hash
    def load(self):
        path = Path(MODEL_DIR) / MODEL_FILENAME
        if not path.exists(): raise FileNotFoundError(f"Model not found at {path}")
        logger.info("Loading model from %s", path)
        self._checkpoint_hash = _compute_sha256(path)
        self._device = _resolve_device()
        self._model = _build_model()
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        self._model.load_state_dict(sd, strict=False)
        self._model.to(self._device).eval()
        self._loaded = True
        logger.info("Model loaded on %s (hash: %s)", self._device, self._checkpoint_hash[:16])
    def unload(self):
        if self._model: del self._model; self._model = None
        self._loaded = False
        if torch.cuda.is_available(): torch.cuda.empty_cache()
