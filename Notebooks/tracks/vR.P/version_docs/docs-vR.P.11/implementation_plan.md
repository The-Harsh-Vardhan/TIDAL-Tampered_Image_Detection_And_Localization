# Implementation Plan — vR.P.11: Higher Resolution (512x512)

## 1. Cell-by-Cell Changes from vR.P.3

| Cell | Type | Change |
|------|------|--------|
| 0 | Markdown | Title -> "vR.P.11", metadata (512x512), pipeline (512, Focal+Dice) |
| 1 | Markdown | Changelog + diff table (P.3 -> P.11) |
| 2 | Code | VERSION='vR.P.11', IMAGE_SIZE=512, BATCH_SIZE=8, GRAD_ACCUM_STEPS=2, FOCAL_ALPHA/GAMMA, EPOCHS=50, PATIENCE=10, NUM_WORKERS=4, PREFETCH_FACTOR=2, cudnn.benchmark=True |
| 7 | Markdown | Resolution table: 384->512 |
| 8 | Code | Update hardcoded defaults: size=512 in compute_ela_statistics(), mask_size=512 in Dataset |
| 9 | Code | DataLoaders: prefetch_factor=PREFETCH_FACTOR |
| 11 | Markdown | Architecture: 384->512 in all dimension annotations |
| 13 | Markdown | Training config table (Focal+Dice, batch=8, accum=2, epochs=50, patience=10) |
| 14 | Code | Replace SoftBCEWithLogitsLoss with FocalLoss, add gradient accumulation to train_one_epoch |
| 25 | Code | Results table: "ELA 512^2", resolution column |
| 26 | Markdown | Discussion: resolution hypothesis, VRAM considerations, next steps |
| 27 | Code | Model filename -> vR.P.11, config dict updates |

All other cells (3-6, 10, 12, 15-24) remain unchanged from vR.P.3.

---

## 2. Key Implementation Details

### Constants (Cell 2)

```python
VERSION = 'vR.P.11'
CHANGE = 'Higher resolution (512x512) + Focal+Dice loss + gradient accumulation'
IMAGE_SIZE = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2  # Effective batch = 8 * 2 = 16
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
EPOCHS = 50
PATIENCE = 10
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
```

### Loss + Gradient Accumulation (Cell 14)

```python
# Before (vR.P.3):
bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
def criterion(pred, target):
    return bce_loss_fn(pred, target) + dice_loss_fn(pred, target)

# After (vR.P.11):
focal_loss_fn = smp.losses.FocalLoss(mode='binary', alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
def criterion(pred, target):
    return focal_loss_fn(pred, target) + dice_loss_fn(pred, target)

# train_one_epoch with gradient accumulation:
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for step, (images, masks, labels) in enumerate(tqdm(loader, desc='Train', leave=False)):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with autocast('cuda'):
            predictions = model(images)
            loss = criterion(predictions, masks) / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * GRAD_ACCUM_STEPS
        num_batches += 1
    return total_loss / num_batches
```

### Hardcoded 384 References (Cell 8)

```python
# compute_ela_statistics: size=384 -> size=512
# CASIASegmentationDataset: mask_size=384 -> mask_size=512
```

---

## 3. Verification Checklist

- [ ] Cell count = 28
- [ ] VERSION = 'vR.P.11'
- [ ] IMAGE_SIZE = 512
- [ ] BATCH_SIZE = 8
- [ ] GRAD_ACCUM_STEPS = 2
- [ ] FOCAL_ALPHA = 0.25
- [ ] FOCAL_GAMMA = 2.0
- [ ] EPOCHS = 50
- [ ] PATIENCE = 10
- [ ] FocalLoss used (not SoftBCEWithLogitsLoss in code cells)
- [ ] No hardcoded '384' remaining in any cell
- [ ] Model filename contains 'vR.P.11'

---

## 4. What Does NOT Change

The model architecture, data loading logic, evaluation, and visualization remain identical to vR.P.3 (cells 3-6, 10, 12, 15-24). These cells use the IMAGE_SIZE variable or model output dimensions, so they automatically adapt to 512x512.

Cell 15 (training loop) does NOT change because:
- It calls train_one_epoch() which handles gradient accumulation internally
- Epoch count and patience come from constants defined in cell 2
- Checkpoint format is identical
