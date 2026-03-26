"""
dvc_pipeline/src/dataset.py
============================
CASIA 2.0 Segmentation Dataset for DVC pipeline.

Clean extraction from the research notebook (vrp19.py) with configurable
paths loaded from params.yaml.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance
from torch.utils.data import Dataset


class CASIASegmentationDataset(Dataset):
    """PyTorch Dataset for CASIA 2.0 tampered image segmentation.

    Produces 9-channel Multi-Quality RGB ELA inputs and binary masks.
    """

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[str | None],
        labels: list[int],
        ela_mean: torch.Tensor,
        ela_std: torch.Tensor,
        img_size: int = 384,
        qualities: list[int] | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.ela_mean = ela_mean
        self.ela_std = ela_std
        self.img_size = img_size
        self.qualities = qualities or [75, 85, 95]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Multi-quality RGB ELA (9 channels)
        try:
            mqela = self._compute_multi_q_ela(self.image_paths[idx])
        except Exception:
            mqela = np.zeros((self.img_size, self.img_size, 9), dtype=np.uint8)

        # Normalize
        tensor = torch.from_numpy(mqela.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1)  # (9, H, W)
        for c in range(9):
            tensor[c] = (tensor[c] - self.ela_mean[c]) / self.ela_std[c]

        # Mask
        mask_path = self.mask_paths[idx]
        if mask_path and os.path.exists(mask_path):
            mask = (
                Image.open(mask_path)
                .convert("L")
                .resize((self.img_size, self.img_size), Image.NEAREST)
            )
            mask_arr = np.array(mask).astype(np.float32) / 255.0
            mask_arr = (mask_arr > 0.5).astype(np.float32)
        else:
            label = self.labels[idx]
            mask_arr = (
                np.ones((self.img_size, self.img_size), dtype=np.float32)
                if label == 1
                else np.zeros((self.img_size, self.img_size), dtype=np.float32)
            )

        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)
        return tensor, mask_tensor, self.labels[idx]

    def _compute_multi_q_ela(self, image_path: str) -> np.ndarray:
        """Compute 9-channel Multi-Q RGB ELA for a single image."""
        image = Image.open(image_path).convert("RGB")
        channels = []
        for q in self.qualities:
            channels.append(self._compute_ela_rgb(image, q))
        return np.concatenate(channels, axis=-1)

    def _compute_ela_rgb(self, image: Image.Image, quality: int) -> np.ndarray:
        """Compute ELA at given quality, return (H, W, 3) uint8."""
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf)
        ela = ImageChops.difference(image, resaved)
        extrema = ela.getextrema()
        max_diff = max(v[1] for v in extrema)
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        ela = ImageEnhance.Brightness(ela).enhance(scale)
        ela = ela.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.array(ela)


def collect_image_paths(directory: str, extensions: set[str] | None = None) -> list[str]:
    """Collect sorted image file paths from a directory."""
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
    paths = []
    for f in sorted(os.listdir(directory)):
        if Path(f).suffix.lower() in extensions:
            paths.append(os.path.join(directory, f))
    return paths
