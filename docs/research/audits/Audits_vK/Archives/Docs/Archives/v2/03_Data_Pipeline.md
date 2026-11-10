# 03 — Data Pipeline

## Purpose

Specify the PyTorch dataset class, augmentation policy, and DataLoader configuration.

## PyTorch Dataset Class

```python
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TamperingDataset(Dataset):
    def __init__(self, pairs, transform=None, image_size=512):
        self.pairs = pairs
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]

        # Load image
        image = cv2.imread(entry['image_path'])
        if image is None:
            raise IOError(f"Failed to load image: {entry['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load or generate mask
        if entry['mask_path'] is not None:
            mask = cv2.imread(entry['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise IOError(f"Failed to load mask: {entry['mask_path']}")
            mask = (mask > 128).astype(np.uint8)
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        # Apply transforms (or fallback to basic resize + normalize + tensor)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size),
                              interpolation=cv2.INTER_NEAREST)
            image = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).float()

        # Ensure mask has channel dimension
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(0).float()

        label = torch.tensor(entry['label'], dtype=torch.float32)
        return image, mask, label
```

Key design decisions:
- `transform=None` path still returns normalized tensors with channelized mask (avoids raw NumPy leaking into DataLoader).
- Explicit read-error checks on `cv2.imread` results.
- Mask uses nearest-neighbor interpolation to preserve binary values.

## Augmentation Policy

### MVP Transforms (Phase 1)

Spatial augmentations only. Photometric transforms are deferred to Phase 2.

```python
train_transform_mvp = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])
```

### Phase 2 Transforms (Optimization)

Add photometric augmentations after the MVP baseline is stable:

```python
train_transform_v2 = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.3,
    ),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.2,
    ),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])
```

Note: Verify that `GaussNoise` and `ImageCompression` parameter names match the installed `albumentations` version before running.

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

### Transforms Excluded by Design

| Transform | Reason |
|---|---|
| `RandomCrop` / `CenterCrop` | Can crop out the tampered region, producing incorrect labels |
| `ElasticTransform` | Destroys CFA/demosaicing patterns (forensic signals) |
| Heavy `GaussianBlur` | Destroys noise residuals used for tampering detection |
| `CoarseDropout` / `Cutout` | Creates false mask-like patterns |
| `GridDistortion` | Adds unrealistic distortion |
| Extreme `ColorJitter` | Color inconsistency is itself a forensic signal |

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
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)
```

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size` | 4 | Fits T4 VRAM with AMP |
| `num_workers` | 2 | Colab typically provides 2 CPU cores |
| `pin_memory` | True | Faster CPU-to-GPU transfer |
| `drop_last` | True (train only) | Avoids small final batch causing BatchNorm instability |

Note: `persistent_workers=True` may work on some Colab runtimes but is not guaranteed. Test before assuming it works.

## Related Documents

- [02_Dataset_and_Preprocessing.md](02_Dataset_and_Preprocessing.md) — Dataset selection and cleaning
- [04_Model_Architecture.md](04_Model_Architecture.md) — Model that consumes pipeline output
- [05_Training_Strategy.md](05_Training_Strategy.md) — Training loop
