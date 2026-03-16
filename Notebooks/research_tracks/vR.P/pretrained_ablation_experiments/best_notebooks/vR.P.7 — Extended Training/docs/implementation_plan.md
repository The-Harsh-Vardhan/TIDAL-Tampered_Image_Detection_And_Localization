# Implementation Plan — vR.P.7: ELA + Extended Training

## 1. Cell-by-Cell Changes from vR.P.3

| Cell | Type | Change |
|------|------|--------|
| 0 | Markdown | Title → "vR.P.7 — ELA + Extended Training", update pipeline diagram |
| 1 | Markdown | Changelog → extended training, patience increase, P.3 was still improving |
| 2 | Code | `VERSION='vR.P.7'`, `CHANGE` updated, `EPOCHS=50`, `PATIENCE=10`, `NUM_WORKERS=4`, add `PREFETCH_FACTOR=2` |
| 22-27 | Code | **FIX P.3's NameError bug:** rename `denormalize` → `denormalize_ela` in `visualize_predictions` |
| 25 | Code | Tracking table → add vR.P.7 live row, include P.3 hardcoded row |
| 26 | Markdown | Discussion → extended training rationale, compare with P.3 |
| 27 | Code | Model save → `vR.P.7` filename |

All other cells remain IDENTICAL to vR.P.3: ELA dataset class, ELA normalization computation, freeze strategy, model creation, loss function, training loop, evaluation, confusion matrix, ROC curves.

---

## 2. Key Implementation Details

### 2.1 Configuration Changes (Cell 2)

```python
# --- CHANGED from P.3 ---
VERSION = 'vR.P.7'
CHANGE = 'Extended training (50 epochs, patience 10) — P.3 was still improving at epoch 25'
EPOCHS = 50          # was 25 in P.3
PATIENCE = 10        # was 7 in P.3
NUM_WORKERS = 4      # was 2 in P.3
PREFETCH_FACTOR = 2  # explicit (was default in P.3)

# --- UNCHANGED from P.3 ---
SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 3
NUM_CLASSES = 1
LEARNING_RATE = 1e-3
ELA_QUALITY = 90
CHECKPOINT_PATH = f'{VERSION}_checkpoint.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 2.2 DataLoader Changes (Cell 9)

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=NUM_WORKERS > 0,
                          prefetch_factor=PREFETCH_FACTOR, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=NUM_WORKERS > 0,
                        prefetch_factor=PREFETCH_FACTOR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         persistent_workers=NUM_WORKERS > 0,
                         prefetch_factor=PREFETCH_FACTOR)
```

### 2.3 P.3 Bug Fix (Cell 22-27)

P.3 crashed with `NameError: name 'denormalize' is not defined` in the prediction visualization cell. The function was defined as `denormalize_ela` but called as `denormalize`. Fix:

```python
# In visualize_predictions function:
# BEFORE (P.3 — broken):
img_display = denormalize(img_tensor).permute(1, 2, 0).numpy()

# AFTER (P.7 — fixed):
img_display = denormalize_ela(img_tensor).permute(1, 2, 0).numpy()
```

This is a cosmetic bug fix, not a variable change. It does not affect training or evaluation results.

---

## 3. ELA Pipeline (Unchanged from P.3)

The complete ELA preprocessing pipeline is carried forward without modification:

```
For each image:
  1. Open as RGB (PIL)
  2. Re-save as JPEG at quality=90 (BytesIO buffer)
  3. Compute pixel-wise absolute difference (original - resaved)
  4. Find maximum channel difference across image
  5. Scale brightness by 255/max_diff (normalize to full range)
  6. Resize to 384×384
  7. Convert to tensor [0, 1] via ToTensor()
  8. Normalize with ELA-specific mean/std (computed from 500 training samples)
```

ELA normalization values (computed at runtime, expected to be identical to P.3):
```
ELA_MEAN ≈ [0.0497, 0.0418, 0.0590]
ELA_STD  ≈ [0.0663, 0.0570, 0.0756]
```

---

## 4. Freeze Strategy (Unchanged from P.3)

```python
# Step 1: Freeze ALL encoder parameters
for param in model.encoder.parameters():
    param.requires_grad = False

# Step 2: Unfreeze ONLY BatchNorm layers in encoder
for module in model.encoder.modules():
    if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        for param in module.parameters():
            param.requires_grad = True
        module.track_running_stats = True
```

Expected trainable parameter breakdown:
- Encoder BN params: ~17,024
- Decoder: ~3,151,552
- Segmentation head: ~145
- **Total trainable: ~3,168,721 (13.0%)**
- Frozen: ~21,267,648 (87.0%)

---

## 5. Speed Optimizations (from P.1.5, unchanged)

| Optimization | Code Pattern |
|-------------|-------------|
| AMP autocast | `with autocast('cuda'):` in train/val |
| GradScaler | `scaler = GradScaler('cuda')` + scale/step/update |
| TF32 matmul | `torch.backends.cuda.matmul.allow_tf32 = True` |
| TF32 cuDNN | `torch.backends.cudnn.allow_tf32 = True` |
| Pin memory | `pin_memory=True` on all DataLoaders |
| Persistent workers | `persistent_workers=True` on all DataLoaders |
| Non-blocking transfer | `.to(device, non_blocking=True)` |
| Fast grad zero | `optimizer.zero_grad(set_to_none=True)` |
| Drop last (train) | `drop_last=True` on train_loader |

---

## 6. Verification Checklist

- [ ] VERSION = 'vR.P.7'
- [ ] EPOCHS = 50
- [ ] PATIENCE = 10
- [ ] NUM_WORKERS = 4
- [ ] PREFETCH_FACTOR = 2
- [ ] ELA_QUALITY = 90
- [ ] LEARNING_RATE = 1e-3 (unchanged from P.3)
- [ ] Dataset computes ELA in __getitem__ (unchanged)
- [ ] ELA mean/std computed from training set (unchanged)
- [ ] Encoder frozen except BN layers (unchanged)
- [ ] `denormalize_ela` bug fixed in visualization cells
- [ ] Checkpoint save/resume works
- [ ] AMP + TF32 enabled
- [ ] Model saved as `vR.P.7_unet_resnet34_model.pth`
- [ ] Tracking table includes P.3 hardcoded row + P.7 live row

---

## 7. Runtime Estimate

| Component | Time |
|-----------|------|
| ELA computation per image | ~50ms (JPEG re-save + diff) |
| ELA stats computation (500 images) | ~30s |
| Training per epoch (8,829 images) | ~3-4 min (with AMP, 4 workers) |
| Best case (early stop at ~35) | ~105-140 min |
| Worst case (full 50 epochs) | ~150-200 min |
| Evaluation | ~5 min |
| **Total session** | **~115-210 min** |

Slightly faster per-epoch than P.3 due to 4 workers (vs 2), but total time is longer due to more epochs. Well within Kaggle T4/P100 session limits.

---

## 8. Expected LR Schedule

Based on P.3's LR trajectory and extending to 50 epochs:

| Epoch Range | Expected LR | Trigger |
|-------------|-------------|---------|
| 1-8 | 1e-3 | Initial LR |
| 9-19 | 5e-4 | ReduceLROnPlateau (patience=3) |
| 20-28 | 2.5e-4 | Second reduction |
| 29-35* | 1.25e-4 | Third reduction (new in P.7) |
| 36-42* | 6.25e-5 | Fourth reduction (new in P.7) |
| 43-50* | 3.125e-5 | Fifth reduction (if reached) |

*Projected — actual schedule depends on val_loss behaviour.

The key insight is that each LR reduction opens a new learning region. P.3 only experienced 2 reductions. P.7 may experience 3-5 reductions, each potentially unlocking small but cumulative improvements.
