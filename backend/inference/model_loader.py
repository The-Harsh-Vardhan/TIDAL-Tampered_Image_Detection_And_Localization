"""Singleton model loader with checkpoint validation for vR.P.30.1."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path

import torch

from .model_architecture import build_vrp301_model

logger = logging.getLogger(__name__)
DEFAULT_MODEL_DIR = Path("models") / "inference" / "vR.P.30.1"
MODEL_DIR = os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "best_model.pt")
DEVICE = os.environ.get("DEVICE", "auto")
MANIFEST_FILENAME = "manifest.json"

# Hugging Face Hub settings
HF_REPO_ID = os.environ.get("HF_REPO_ID", "the-harsh-vardhan/TIDAL")
HF_MODEL_FILE = os.environ.get("HF_MODEL_FILE", "checkpoints/vR.P.30.1/best_model.pt")
HF_MANIFEST_FILE = os.environ.get("HF_MANIFEST_FILE", "checkpoints/vR.P.30.1/manifest.json")
HF_TOKEN = os.environ.get("HF_TOKEN")


def _resolve_device():
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def _compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _load_manifest(model_dir: Path) -> dict:
    manifest_path = model_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class ModelLoader:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._device = None
        self._checkpoint_hash = None
        self._manifest = {}
        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def model(self):
        if not self._loaded:
            self.load()
        return self._model

    @property
    def device(self):
        if self._device is None:
            self._device = _resolve_device()
        return self._device

    @property
    def checkpoint_hash(self):
        return self._checkpoint_hash

    @property
    def manifest(self):
        return self._manifest

    @property
    def model_dir(self):
        return Path(MODEL_DIR)

    @property
    def model_filename(self):
        return MODEL_FILENAME

    def _try_download_from_hub(self, path: Path) -> None:
        """Attempt to download the model checkpoint from Hugging Face Hub."""
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import]
        except ImportError:
            logger.warning("huggingface_hub not installed — cannot auto-download model.")
            return

        logger.info(
            "Model not found locally. Attempting HF Hub download from %s ...", HF_REPO_ID
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MODEL_FILE,
                local_dir=str(path.parent),
                local_dir_use_symlinks=False,
                token=HF_TOKEN or None,
                force_download=False,
            )
            dl_path = Path(downloaded)
            if dl_path.resolve() != path.resolve() and dl_path.exists():
                dl_path.rename(path)
            logger.info("Model downloaded to %s", path)
        except Exception as exc:
            logger.error("HF Hub download failed: %s", exc)
            return

        # Also pull the manifest (best-effort, non-fatal)
        manifest_path = path.parent / MANIFEST_FILENAME
        if not manifest_path.exists():
            try:
                mf = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=HF_MANIFEST_FILE,
                    local_dir=str(path.parent),
                    local_dir_use_symlinks=False,
                    token=HF_TOKEN or None,
                    force_download=False,
                )
                mf_path = Path(mf)
                if mf_path.resolve() != manifest_path.resolve() and mf_path.exists():
                    mf_path.rename(manifest_path)
            except Exception:
                pass  # manifest is optional

    def load(self):
        path = self.model_dir / MODEL_FILENAME
        if not path.exists():
            self._try_download_from_hub(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. "
                "Set HF_TOKEN env var to enable auto-download from Hugging Face Hub."
            )
        logger.info("Loading model from %s", path)
        self._manifest = _load_manifest(self.model_dir)
        self._checkpoint_hash = _compute_sha256(path)
        self._device = _resolve_device()
        self._model = build_vrp301_model()
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        sd = (
            ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            if isinstance(ckpt, dict)
            else ckpt
        )
        self._model.load_state_dict(sd, strict=True)
        self._model.to(self._device).eval()
        self._loaded = True
        logger.info("Model loaded on %s (hash: %s)", self._device, self._checkpoint_hash[:16])

    def unload(self):
        if self._model:
            del self._model
            self._model = None
        self._manifest = {}
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
