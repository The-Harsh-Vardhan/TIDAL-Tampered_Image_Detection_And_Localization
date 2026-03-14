# Implementation Plan -- vR.P.5: ResNet-50 Encoder

## 1. Cell-by-Cell Changes from vR.P.1.5

**Source notebook:** `vR.P.1.5 Image Detection and Localisation.ipynb` (28 cells)
**Target notebook:** `vR.P.5 Image Detection and Localisation.ipynb` (28 cells)

| Cell | Type | Change Description |
|------|------|--------------------|
| 0 | markdown | Title -> vR.P.5, change description -> ResNet-50 encoder, updated pipeline diagram |
| 1 | markdown | Add vR.P.5 row to changelog, ResNet-34 vs 50 comparison table |
| 2 | code | `VERSION='vR.P.5'`, `CHANGE='ResNet-50 encoder (test deeper features)'`, `ENCODER='resnet50'` |
| 3 | markdown | UNCHANGED (Dataset section header) |
| 4 | code | UNCHANGED (Dataset path discovery) |
| 5 | code | UNCHANGED (Image path collection + GT mask matching) |
| 6 | code | UNCHANGED (ELA pseudo-mask fallback) |
| 7 | markdown | UNCHANGED (Data preparation header) |
| 8 | code | UNCHANGED (Dataset class + ImageNet transforms) |
| 9 | code | UNCHANGED (Data splitting + DataLoader creation) |
| 10 | code | UNCHANGED (Sample visualization) |
| 11 | markdown | Updated architecture description for ResNet-50 bottleneck blocks and channel sizes |
| 12 | code | No code logic changes (uses `ENCODER` variable). Updated print statement. |
| 13 | markdown | Updated training config: note about VRAM with ResNet-50 |
| 14 | code | UNCHANGED (Loss, optimizer, training/validation functions with AMP) |
| 15 | code | UNCHANGED (Training loop with checkpoint save/resume + scaler) |
| 16 | markdown | UNCHANGED (Evaluation header) |
| 17 | code | UNCHANGED (Pixel-level test evaluation) |
| 18 | code | UNCHANGED (Image-level test evaluation) |
| 19 | code | UNCHANGED (Confusion matrix + ROC curve) |
| 20 | code | UNCHANGED (Training curves) |
| 21 | markdown | UNCHANGED (Visualization header) |
| 22 | code | UNCHANGED (Prediction visualization grid) |
| 23 | code | UNCHANGED (Per-image metric distribution) |
| 24 | markdown | UNCHANGED (Results summary header) |
| 25 | code | Updated results tracking table references for vR.P.5 |
| 26 | markdown | Updated discussion: ResNet-50 rationale, next steps (vR.P.6 EfficientNet-B0) |
| 27 | code | Updated save model config dict with encoder info |

**Summary:** 10 cells modified, 18 cells unchanged.

---

## 2. Key Implementation Details

### 2.1 Cell 2 -- Config Changes (the ONE functional change)

```python
VERSION = 'vR.P.5'
CHANGE = 'ResNet-50 encoder (test deeper features)'
ENCODER = 'resnet50'  # was 'resnet34' in vR.P.1.5
```

All other config values remain identical:
- `ENCODER_WEIGHTS = 'imagenet'`
- `IN_CHANNELS = 3`
- `BATCH_SIZE = 16`
- `LEARNING_RATE = 1e-3`
- `NUM_WORKERS = 2`
- AMP + TF32 enabled

### 2.2 Cell 12 -- Model Build (no code change needed)

The model build code already uses the `ENCODER` variable:

```python
model = smp.Unet(
    encoder_name=ENCODER,       # <-- reads 'resnet50' from config
    encoder_weights=ENCODER_WEIGHTS,
    in_channels=IN_CHANNELS,
    classes=NUM_CLASSES,
    activation=None
)
```

SMP automatically handles:
- Loading ResNet-50 pretrained weights
- Adapting UNet decoder to ResNet-50's channel sizes (256, 512, 1024, 2048)
- Correct parameter counting

### 2.3 Expected Parameter Counts

| Component | ResNet-34 (vR.P.1.5) | ResNet-50 (vR.P.5) |
|-----------|----------------------|---------------------|
| Encoder (frozen) | ~21.3M | ~23.5M |
| Decoder (trainable) | ~3.1M | ~8.7M |
| Seg head (trainable) | ~65 | ~17 |
| **Total** | **~24.4M** | **~32.2M** |
| **Trainable** | **~3.1M** | **~8.7M** |

---

## 3. Verification Checklist

1. [ ] `VERSION = 'vR.P.5'` in cell 2
2. [ ] `ENCODER = 'resnet50'` in cell 2
3. [ ] `ENCODER_WEIGHTS = 'imagenet'` (unchanged)
4. [ ] String `resnet34` does NOT appear in cell 2 config section
5. [ ] `BATCH_SIZE = 16` (unchanged)
6. [ ] `LEARNING_RATE = 1e-3` (unchanged)
7. [ ] `NUM_WORKERS = 2` (unchanged)
8. [ ] AMP imports: `from torch.amp import autocast, GradScaler`
9. [ ] TF32 enabled: `torch.backends.cuda.matmul.allow_tf32 = True`
10. [ ] `non_blocking=True` in training function
11. [ ] `set_to_none=True` in `optimizer.zero_grad()`
12. [ ] `GradScaler('cuda')` in training loop
13. [ ] Total cells = 28
14. [ ] BCEDiceLoss present
15. [ ] Notebook is valid JSON

---

## 4. Runtime Estimate

| Phase | Time |
|-------|------|
| Setup + data loading | ~3 minutes |
| Per epoch (ResNet-50, AMP, batch=16) | ~6-8 minutes (vs ~4-5 for ResNet-34) |
| Total (25 epochs max, likely ~15 with early stopping) | ~90-120 minutes |
| Evaluation + visualization | ~5 minutes |
| **Total estimated** | **~100-130 minutes** |

ResNet-50 forward passes are ~30-40% slower than ResNet-34 due to wider activations, partially offset by AMP.
