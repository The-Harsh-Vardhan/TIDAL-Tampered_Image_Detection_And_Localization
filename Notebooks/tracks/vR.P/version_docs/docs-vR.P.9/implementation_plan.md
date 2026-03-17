# Implementation Plan — vR.P.9: Focal + Dice Loss

## 1. Cell-by-Cell Changes from vR.P.3

| Cell | Type | Change |
|------|------|--------|
| 0 | Markdown | Title → "vR.P.9 — Focal + Dice Loss", metadata, pipeline (loss line) |
| 1 | Markdown | Changelog + diff table (P.3 → P.9) |
| 2 | Code | VERSION='vR.P.9', CHANGE, NUM_WORKERS=4, PREFETCH_FACTOR=2, add FOCAL_ALPHA/GAMMA |
| 9 | Code | DataLoaders: NUM_WORKERS=4, prefetch_factor=2 |
| 13 | Markdown | Training config table (Focal+Dice loss description) |
| 14 | Code | Replace SoftBCEWithLogitsLoss with FocalLoss(alpha=0.25, gamma=2.0) |
| 25 | Code | Results table version info |
| 26 | Markdown | Discussion: Focal vs BCE analysis, next steps |
| 27 | Code | Model filename → vR.P.9_unet_resnet34_model.pth |

All other cells (3–8, 10–12, 15–24) remain unchanged from vR.P.3.

---

## 2. Key Implementation Detail

### Loss Function (Cell 14)

```python
# Before (vR.P.3):
bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def criterion(pred, target):
    return bce_loss_fn(pred, target) + dice_loss_fn(pred, target)

# After (vR.P.9):
focal_loss_fn = smp.losses.FocalLoss(mode='binary', alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def criterion(pred, target):
    return focal_loss_fn(pred, target) + dice_loss_fn(pred, target)
```

Both `FocalLoss` and `DiceLoss` from SMP operate on raw logits. No sigmoid pre-processing is needed.

### Constants (Cell 2)

```python
VERSION = 'vR.P.9'
CHANGE = 'Focal + Dice loss (replace BCE with focal, alpha=0.25, gamma=2.0)'
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
```

---

## 3. Verification Checklist

- [ ] VERSION = 'vR.P.9'
- [ ] FOCAL_ALPHA = 0.25
- [ ] FOCAL_GAMMA = 2.0
- [ ] FocalLoss used (not SoftBCEWithLogitsLoss)
- [ ] DiceLoss preserved (from_logits=True)
- [ ] NUM_WORKERS = 4
- [ ] prefetch_factor = 2
- [ ] ELA_QUALITY = 90 (unchanged from P.3)
- [ ] EPOCHS = 25 (unchanged from P.3)
- [ ] Encoder frozen + BN unfrozen (unchanged from P.3)
- [ ] Model filename contains 'vR.P.9'
- [ ] Cell count = 28

---

## 4. What Does NOT Change

The entire training pipeline, model architecture, data loading, evaluation, and visualization remain identical to vR.P.3. This experiment isolates the effect of the loss function alone.

Cells 3–8 (dataset), 10 (visualization), 12 (model build), 15 (training loop), 16–24 (evaluation + visualization) are copied verbatim.
