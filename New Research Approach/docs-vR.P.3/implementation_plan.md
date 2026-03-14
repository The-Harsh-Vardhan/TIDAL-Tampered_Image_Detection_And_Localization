# Implementation Plan — vR.P.3: ELA as Input

## 1. Cell-by-Cell Changes from vR.P.2

| Cell | Type | Change |
|------|------|--------|
| 0 | Markdown | Title → "vR.P.3 — ELA as Input", pipeline diagram updated |
| 1 | Markdown | Changelog → ELA input change, encoder back to frozen+BN |
| 2 | Code | VERSION='vR.P.3', CHANGE updated, remove ENCODER_LR, add ELA_QUALITY=90 |
| 7 | Markdown | Data prep → explain ELA input instead of RGB, normalization strategy |
| 8 | Code | Dataset class → compute ELA in `__getitem__`, compute ELA mean/std from sample |
| 10 | Code | Sample viz → show ELA input (what the model sees) instead of RGB |
| 11 | Markdown | Architecture → frozen encoder with BN unfrozen, ELA input |
| 12 | Code | Model build → freeze all encoder, then unfreeze BN layers only |
| 13 | Markdown | Training config → single LR, no differential LR |
| 14 | Code | Optimizer → single param group (decoder + encoder BN), remove ENCODER_LR |
| 20 | Code | Fix LR history key bug from vR.P.2 |
| 25 | Code | Tracking table → add vR.P.3 live row |
| 26 | Markdown | Discussion → ELA input rationale |
| 27 | Code | Model save → vR.P.3 filename |

All other cells (dataset discovery, ELA pseudo-mask, data splitting, training loop, evaluation, visualization) remain unchanged.

---

## 2. Key Implementation Details

### 2.1 ELA in the Dataset `__getitem__`

```python
def compute_ela_tensor(image_path, quality=90, size=384):
    """Load image, compute ELA, resize, return as PIL Image."""
    original = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    ela = ImageChops.difference(original, resaved)
    extrema = ela.getextrema()
    max_diff = max(val[1] for val in extrema)
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return ela
```

The ELA computation happens per-image in `__getitem__`. The transform pipeline:
1. Compute ELA from raw image path
2. Resize to 384×384
3. ToTensor (→ [0, 1] range)
4. Normalize with ELA-specific mean/std

### 2.2 ELA Normalization Strategy

ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]) are wrong for ELA. We compute ELA-specific statistics from a sample of the training set:

```python
# Sample 500 training images, compute ELA, measure channel mean/std
ela_means, ela_stds = compute_ela_statistics(train_paths, n_samples=500)
```

This gives the encoder normalized inputs — critical for BN layer adaptation.

### 2.3 Encoder Freeze with BN Unfreeze

```python
# Freeze ALL encoder parameters
for param in model.encoder.parameters():
    param.requires_grad = False

# Unfreeze ONLY BatchNorm layers in encoder
for module in model.encoder.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        for param in module.parameters():
            param.requires_grad = True
        module.track_running_stats = True
```

This allows BN running mean/variance to adapt to ELA distribution while keeping convolutional weights frozen.

---

## 3. Verification Checklist

- [ ] VERSION = 'vR.P.3'
- [ ] ELA_QUALITY = 90
- [ ] No ENCODER_LR (single LR)
- [ ] Dataset computes ELA in __getitem__
- [ ] ELA mean/std computed from training set (not ImageNet)
- [ ] Encoder frozen except BN layers
- [ ] Single optimizer param group (or 2 groups at same LR)
- [ ] All frozen params preserved (loss, scheduler, split, batch size, epochs)
- [ ] Checkpoint save/resume works
- [ ] AMP + TF32 enabled
- [ ] Tracking table includes vR.P.2 hardcoded row

---

## 4. Runtime Estimate

| Component | Time |
|-----------|------|
| ELA computation per image | ~50ms (JPEG re-save + diff) |
| ELA stats computation (500 images) | ~30s |
| Training per epoch (8,829 images) | ~3-4 min (with AMP) |
| Total (25 epochs max) | ~75-100 min |

Note: ELA computation adds ~50ms per image overhead vs direct RGB loading. With batch_size=16 and 552 train batches, this adds ~27s per epoch.
