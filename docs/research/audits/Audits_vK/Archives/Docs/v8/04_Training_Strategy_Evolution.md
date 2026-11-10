# 04 — Training Strategy Evolution

## Purpose

Trace the evolution of loss function, optimizer, scheduler, augmentation, and regularization from Docs7 → Audit → Run01, and define the training strategy for v8.

---

## Phase 1: Docs7 Design

| Component | Specification |
|---|---|
| Loss | `BCEDiceLoss` = BCEWithLogitsLoss + DiceLoss (smooth=1.0) |
| Optimizer | AdamW (encoder LR=1e-4, decoder LR=1e-3, weight_decay=1e-4) |
| Scheduler | **None** |
| Gradient accumulation | 4 steps (effective batch=16) |
| Mixed precision | AMP (autocast + GradScaler) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=10, monitor=val_pixel_f1 |
| Max epochs | 50 |
| Augmentation | HFlip, VFlip, RandomRotate90 |
| BatchNorm | From pretrained ImageNet statistics |
| BCE pos_weight | **Not set** |
| Dice computation | **Batch-level** (aggregated across batch) |

## Phase 2: Audit Critique

### Critical Gaps Identified

| Finding | Severity | Impact |
|---|---|---|
| No `pos_weight` for BCE despite <5% tampered pixels | CRITICAL | Model learns "predict background" → low threshold |
| No LR scheduler | CRITICAL | Model overshoots after convergence → overfitting |
| Batch-level Dice | MEDIUM | Large masks dominate loss; small masks underrepresented |
| No LR warmup | MEDIUM | Pretrained encoder weights may be disrupted early |
| Batch-size-4 with BatchNorm | MEDIUM | Unstable BN statistics, especially with DataParallel (2/GPU) |
| Augmentation too minimal | HIGH | Only geometric transforms; no photometric regularization |
| No multi-seed validation | MEDIUM | Single-run results may be seed-dependent |
| No encoder freezing phase | LOW | Unnecessary for fine-tuning but could help stability |

### The Central Prediction

Audit6 Pro predicted that missing `pos_weight` and no scheduler would cause:
1. Suppressed positive predictions → very low optimal threshold
2. Overfitting after convergence → validation loss divergence

Run01 confirmed both predictions exactly.

## Phase 3: Run01 Evidence

### Training Dynamics (25 epochs, early stop at 15+10)

| Epoch | Train Loss | Val Loss | Val F1 | Observation |
|---|---|---|---|---|
| 1 | 0.9534 | 0.8981 | 0.5028 | Normal startup |
| 5 | 0.7276 | 0.8236 | 0.6600 | Rapid improvement |
| 11 | 0.6599 | 0.7739 | 0.7237 | Near-convergence |
| 15 | 0.6338 | 0.8141 | **0.7289** | **Best epoch** — val loss already rising |
| 20 | 0.5706 | 1.0095 | 0.7028 | Clear overfitting |
| 25 | 0.5149 | 1.2010 | 0.6766 | Severe overfitting; training terminated |

**Key observations:**
- Train loss decreased monotonically (0.95 → 0.51) — model capacity is sufficient
- Val loss hit minimum at epoch ~11 (0.77) but val F1 peak was epoch 15 (0.73) — loss and metric diverged
- Val loss nearly doubled from minimum to termination (0.77 → 1.20)
- 10 epochs of patience were wasted on overfitting

### Threshold Analysis

- **Best threshold: 0.1327** (from sweep over 0.1–0.9)
- Normal calibrated models produce thresholds near 0.5
- 0.1327 means the model assigns low probabilities even to tampered pixels
- **Root cause:** No `pos_weight` → BCE gradient dominated by background → model outputs conservative predictions

### Overfitting Analysis

The overfitting pattern is textbook: constant high LR + small dataset + minimal augmentation.

```
Epoch 11: val_loss=0.7739, val_f1=0.7237  ← convergence zone
Epoch 15: val_loss=0.8141, val_f1=0.7289  ← F1 still improving, loss diverging
Epoch 25: val_loss=1.2010, val_f1=0.6766  ← model memorizing training set
```

A scheduler reducing LR at epoch ~13 (3 patience after best loss at 11) would have allowed fine-tuning without overshoot.

---

## v8 Training Strategy

### P0: Critical Changes

**1. Add Learning Rate Scheduler**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
)
# After each validation epoch:
scheduler.step(val_pixel_f1)
```
Expected impact: Extends useful training from 15 to 30+ epochs by reducing LR when progress stalls.

Alternative: `CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)` for periodic exploration.

**2. Add BCE pos_weight**
```python
# Compute from training set
total_fg = sum(mask.sum() for mask in train_masks if mask is not None)
total_bg = sum(mask.numel() - mask.sum() for mask in train_masks if mask is not None)
pos_weight = torch.tensor([total_bg / max(total_fg, 1)]).to(device)

# In loss:
self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```
Expected impact: Shifts optimal threshold from ~0.13 to ~0.3–0.5, improving tampered-pixel recall.

**3. Expand Training Augmentation**
```python
train_transform = A.Compose([
    A.Resize(384, 384),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # NEW: photometric augmentation
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    # Keep normalize + tensor last
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```
Expected impact: Reduces overfitting, improves JPEG robustness, forces model to learn structural features.

### P1: Important Changes

**4. Per-Sample Dice Loss**
```python
def dice_loss(pred, target, smooth=1.0):
    # Compute per-sample, then average
    pred = torch.sigmoid(pred)
    losses = []
    for i in range(pred.shape[0]):
        p, t = pred[i].view(-1), target[i].view(-1)
        intersection = (p * t).sum()
        dice = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
        losses.append(1 - dice)
    return torch.stack(losses).mean()
```
Prevents large masks from dominating the Dice gradient.

**5. LR Warmup (First 2 Epochs)**
```python
# Linear warmup for first 2 epochs
warmup_epochs = 2
if epoch < warmup_epochs:
    warmup_factor = (epoch + 1) / warmup_epochs
    for pg in optimizer.param_groups:
        pg['lr'] = pg['initial_lr'] * warmup_factor
```
Protects pretrained encoder weights from large initial gradients.

**6. Fix cudnn.benchmark Contradiction**

Run01 has `set_seed()` setting `benchmark=False` immediately overridden by `setup_device()` setting `benchmark=True`.

```python
# In setup_device(), remove or defer:
# torch.backends.cudnn.benchmark = True  # REMOVE — conflicts with set_seed()
```

### P2: Moderate Changes

**7. Gradient norm logging**
```python
# After backward, before optimizer step:
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
wandb.log({'grad_norm': total_norm})
```

**8. Experiment with Focal Loss variant**
```python
# Replace BCE with Focal Loss for hard example mining:
# FocalLoss(alpha=pos_weight, gamma=2.0) + DiceLoss
```

---

## Expected Impact

| Change | Run01 Baseline | Expected v8 |
|---|---|---|
| Optimal threshold | 0.1327 | 0.30–0.50 |
| Overfitting onset | Epoch 15 | Epoch 30+ |
| Useful training epochs | 15/50 | 30–40/50 |
| Tampered-only F1 | 0.4101 | 0.50–0.60 |
| Robustness Δ (JPEG) | −0.13 | −0.05 to −0.08 |
