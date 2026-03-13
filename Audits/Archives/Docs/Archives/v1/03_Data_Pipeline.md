# 03 — Data Pipeline

## Purpose

This document specifies the PyTorch dataset class, augmentation policy, and DataLoader configuration.

## PyTorch Dataset Class

```python
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class TamperingDataset(Dataset):
    def __init__(self, pairs, transform=None, image_size=512):
        """
        Args:
            pairs: List of dicts with keys: image_path, mask_path, label, forgery_type.
            transform: albumentations Compose pipeline.
            image_size: Target spatial resolution.
        """
        self.pairs = pairs
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]

        # Load image (BGR -> RGB)
        image = cv2.imread(entry['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load or generate mask
        if entry['mask_path'] is not None:
            mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask = (mask > 128).astype(np.uint8)
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']       # Tensor (3, H, W)
            mask = augmented['mask']         # Tensor (H, W)

        # Ensure mask has channel dimension
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0).float()

        label = torch.tensor(entry['label'], dtype=torch.float32)

        return image, mask, label
```

## Augmentation Policy

All augmentations use the `albumentations` library, which automatically synchronizes spatial transforms between image and mask.

### Training Transforms

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3,
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=20,
        p=0.2,
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])
```

### Validation and Test Transforms

```python
val_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])
```

No augmentation is applied during validation or testing, only resize and normalization.

### Transforms Excluded by Design

| Transform | Reason for exclusion |
|---|---|
| `RandomCrop` / `CenterCrop` | Can crop out the tampered region, producing incorrect labels |
| `ElasticTransform` | Destroys CFA/demosaicing patterns that are forensic signals |
| Heavy `GaussianBlur` | Destroys noise residuals used for tampering detection |
| `CoarseDropout` / `Cutout` | Creates false mask-like patterns |
| `GridDistortion` | Adds unrealistic distortion |
| Extreme `ColorJitter` | Color inconsistency is itself a forensic signal |

### Stage 2 Additions (Performance Optimization)

These augmentations are added only after the MVP baseline is stable:

```python
A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
```

These improve robustness to noise and JPEG compression but are not required for the initial working model.

## DataLoader Configuration

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | 4 | Fits T4 16 GB VRAM with AMP enabled |
| `num_workers` | 2 | Colab free tier provides 2 CPU cores |
| `pin_memory` | True | Faster CPU-to-GPU transfer |
| `drop_last` | True (train) | Avoids small final batch causing BatchNorm instability |
| `shuffle` | True (train) | Randomize training order each epoch |
| `persistent_workers` | True | Keep workers alive between epochs to reduce overhead |

Images are loaded on-the-fly (not pre-cached) because the full CASIA v2.0 dataset (~2.6 GB) can exceed Colab RAM limits.

## Data Pipeline Order

```
Load image (RGB) → Load mask (grayscale) → Binarize mask → Resize to 512×512 → 
Apply augmentations → Normalize image → Convert to tensor → Return (image, mask, label)
```

## Related Documents

- [02_Dataset_and_Preprocessing.md](02_Dataset_and_Preprocessing.md) — Dataset selection and cleaning
- [04_Model_Architecture.md](04_Model_Architecture.md) — Model that consumes pipeline output
- [05_Training_Pipeline.md](05_Training_Pipeline.md) — Training loop
