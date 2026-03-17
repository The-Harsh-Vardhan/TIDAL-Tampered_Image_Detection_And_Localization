# Implementation Plan — vR.P.12: ELA + Data Augmentation

## 1. Cell-by-Cell Changes from vR.P.3

| Cell | Type | Change |
|------|------|--------|
| 0 | Markdown | Title -> "vR.P.12", metadata (augmentation), pipeline (add augmentation step, Focal+Dice) |
| 1 | Markdown | Changelog + diff table (P.3 -> P.12) |
| 2 | Code | VERSION='vR.P.12', add albumentations install+import, FOCAL_ALPHA/GAMMA, EPOCHS=50, PATIENCE=10, NUM_WORKERS=4, PREFETCH_FACTOR=2, cudnn.benchmark=True |
| 7 | Markdown | Data preparation (add augmentation strategy description) |
| 8 | Code | Dataset class accepts optional `transform` parameter, applies Albumentations to both image and mask |
| 9 | Code | Define augmentation pipeline, pass to train_dataset only, DataLoaders with prefetch_factor |
| 11 | Markdown | Architecture: update loss line to Focal+Dice |
| 13 | Markdown | Training config (Focal+Dice, epochs=50, patience=10, augmentation mention) |
| 14 | Code | Replace SoftBCEWithLogitsLoss with FocalLoss, keep train/validate unchanged |
| 25 | Code | Results table with augmentation column |
| 26 | Markdown | Discussion: augmentation hypothesis, forensic signal preservation |
| 27 | Code | Model filename -> vR.P.12, config dict updates (augmentation info) |

All other cells (3-6, 10, 12, 15-24) remain unchanged from vR.P.3.

---

## 2. Key Implementation Details

### Constants (Cell 2)

```python
VERSION = 'vR.P.12'
CHANGE = 'ELA + data augmentation (Albumentations) + Focal+Dice loss'
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
```

### Augmentation Pipeline (Cell 9)

```python
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3
    ),
    A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    A.RandomBrightnessContrast(
        brightness_limit=0.1, contrast_limit=0.1, p=0.2
    ),
])
```

### Dataset Modification (Cell 8)

```python
# Key change: accept transform parameter
class CASIASegmentationDataset(Dataset):
    def __init__(self, ..., transform=None):
        self.transform = transform

    def __getitem__(self, idx):
        # ... compute ELA, resize, load mask ...
        if self.transform is not None:
            ela_np = np.array(ela)  # PIL -> numpy for Albumentations
            augmented = self.transform(image=ela_np, mask=mask)
            ela = augmented['image']  # numpy
            mask = augmented['mask']  # numpy
        ela_tensor = self.to_tensor(ela)  # Works for PIL and numpy
        # ... normalize, return ...
```

### Loss Function (Cell 14)

```python
# Before (P.3):
bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
# After (P.12):
focal_loss_fn = smp.losses.FocalLoss(mode='binary', alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
```

---

## 3. Verification Checklist

- [ ] Cell count = 28
- [ ] VERSION = 'vR.P.12'
- [ ] IMAGE_SIZE = 384 (unchanged)
- [ ] BATCH_SIZE = 16 (unchanged)
- [ ] FOCAL_ALPHA = 0.25
- [ ] FOCAL_GAMMA = 2.0
- [ ] EPOCHS = 50
- [ ] PATIENCE = 10
- [ ] FocalLoss used (not SoftBCEWithLogitsLoss in code cells)
- [ ] albumentations imported
- [ ] HorizontalFlip in code
- [ ] transform parameter in Dataset class
- [ ] Model filename uses VERSION variable

---

## 4. What Does NOT Change

The model architecture, data loading logic (except augmentation), evaluation, and visualization remain identical to vR.P.3 (cells 3-6, 10, 12, 15-24). Cell 15 (training loop) uses EPOCHS and PATIENCE constants from cell 2, so it automatically adopts the new values.

Augmentation is applied **only to training data**. Validation and test datasets receive `transform=None` to ensure clean, reproducible evaluation.
