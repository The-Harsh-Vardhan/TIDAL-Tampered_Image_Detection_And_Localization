# 08 — Notebook V8 Implementation Plan

## Purpose

Concrete, ordered checklist of changes to implement in Notebook v8. Every item traces back to a specific Run01 finding or audit recommendation. No vague improvements — each change has a reason, a code sketch, and an expected impact.

---

## Implementation Order

Changes are ordered by dependency and priority. Complete each phase before moving to the next.

---

## Phase 1: Loss & Optimizer Fixes (P0)

These changes address the two most critical Run01 failures: overfitting and probability suppression.

### 1.1 Add BCE pos_weight

**Why:** Run01 threshold=0.1327. Background pixel dominance suppresses tampered-region predictions.

**Source:** Audit6 Pro §02 Finding 5, Audit 6.5 §04, §08 Fix 2

**Change:**
```python
# In dataset preparation (before training loop):
total_fg, total_bg = 0, 0
for pair in train_pairs:
    if pair['mask_path']:
        mask = cv2.imread(pair['mask_path'], cv2.IMREAD_GRAYSCALE)
        fg = (mask > 0).sum()
        total_fg += fg
        total_bg += mask.size - fg
pos_weight = torch.tensor([total_bg / max(total_fg, 1)]).to(device)
print(f"pos_weight: {pos_weight.item():.2f}")  # Expected: ~10-30

# In BCEDiceLoss.__init__:
self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Expected impact:** Threshold shifts from ~0.13 to ~0.30–0.50. Tampered-only F1 improves by 0.05–0.15.

**Validation:** After training, check that optimal threshold is in 0.25–0.55 range.

### 1.2 Add Learning Rate Scheduler

**Why:** Run01 overfits at epoch 15. Constant LR=1e-3 is too aggressive after convergence.

**Source:** Audit 6.5 §01, §08 Fix 1

**Change:**
```python
# After optimizer definition:
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
)

# At end of each validation epoch:
scheduler.step(val_pixel_f1)
current_lr = optimizer.param_groups[0]['lr']
print(f"  LR: encoder={optimizer.param_groups[0]['lr']:.2e}, "
      f"decoder={optimizer.param_groups[1]['lr']:.2e}")
```

**Expected impact:** Extends useful training from ~15 to ~30+ epochs. Reduces val loss divergence.

**Validation:** Val loss should not diverge >20% above minimum during training.

### 1.3 Fix cudnn.benchmark Contradiction

**Why:** `set_seed()` sets `benchmark=False`, `setup_device()` overrides to `True`.

**Source:** Audit 6.5 §07

**Change:**
```python
# In setup_device(), REMOVE or comment out:
# torch.backends.cudnn.benchmark = True
# set_seed() already handles this
```

---

## Phase 2: Augmentation Expansion (P1)

### 2.1 Add Photometric & Compression Augmentation

**Why:** Run01 overfits at epoch 15. 13% robustness drop under JPEG compression.

**Source:** Audit6 Pro §02 Finding 7, Audit 6.5 §06, §08 Fix 4

**Change:**
```python
train_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # NEW augmentations:
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    # Standard:
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

**Expected impact:** Reduces JPEG robustness Δ from −0.13 to −0.05. Delays overfitting onset.

**Validation:** JPEG QF50 F1 should be within 0.05 of clean F1.

---

## Phase 3: Evaluation Improvements (P0/P1)

### 3.1 Report Tampered-Only Metrics as Primary

**Why:** Mixed-set F1=0.72 is inflated by authentic images scoring 1.0. True performance is tampered-only F1=0.41.

**Source:** Audit6 Pro §03 Finding 2, Audit 6.5 §02, §08 Fix 3

**Change:** Reorder evaluation output to lead with tampered-only metrics. See [05_Evaluation_Methodology_Evolution.md](05_Evaluation_Methodology_Evolution.md) for the reporting template.

### 3.2 Add Per-Forgery-Type Reporting to Main Output

**Why:** Copy-move F1=0.31 hidden in secondary output in Run01.

**Change:** Include forgery-type breakdown in the main evaluation cell, not just supplementary analysis.

### 3.3 Expand Threshold Sweep

**Why:** Run01 sweeps 0.1–0.9 in 0.1 steps (9 candidates). Best was 0.1327 — at the edge of the range.

**Change:**
```python
thresholds = np.arange(0.05, 0.80, 0.05)  # 15 candidates, starts at 0.05
```

### 3.4 Add Mask-Size Stratification

**Why:** 6/10 worst Run01 failures had mask area <2%.

**Change:**
```python
# After per-image evaluation:
for img_result in results:
    mask_ratio = img_result['gt_mask'].sum() / img_result['gt_mask'].numel()
    if mask_ratio < 0.02:
        img_result['size_bucket'] = 'tiny'
    elif mask_ratio < 0.05:
        img_result['size_bucket'] = 'small'
    elif mask_ratio < 0.15:
        img_result['size_bucket'] = 'medium'
    else:
        img_result['size_bucket'] = 'large'
# Report F1 per bucket
```

---

## Phase 4: Per-Sample Dice & Logging (P1)

### 4.1 Per-Sample Dice Computation

**Why:** Batch-level Dice lets large masks dominate gradients.

**Source:** Audit6 Pro §02 Finding 5

**Change:**
```python
def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    batch_size = pred.shape[0]
    losses = []
    for i in range(batch_size):
        p = pred[i].reshape(-1)
        t = target[i].reshape(-1)
        intersection = (p * t).sum()
        dice = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
        losses.append(1 - dice)
    return torch.stack(losses).mean()
```

### 4.2 Add Gradient Norm Logging

**Why:** Monitor training stability, especially with new scheduler + pos_weight.

**Change:**
```python
# After loss.backward():
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
if CONFIG.get('use_wandb'):
    wandb.log({'grad_norm': total_norm.item()})
```

### 4.3 Log LR per Epoch to W&B

```python
if CONFIG.get('use_wandb'):
    wandb.log({
        'lr_encoder': optimizer.param_groups[0]['lr'],
        'lr_decoder': optimizer.param_groups[1]['lr'],
    })
```

---

## Phase 5: Documentation Alignment (P1)

### 5.1 Update Docs to Match v6.5 Notebook

**Why:** Audit6 Pro §04 found doc-notebook version mismatch (docs reference v5.1, notebooks are v6+).

**Change:**
- Update all doc references to point to the current notebook version
- Reconcile image-level detection description (docs say top-k mean, notebook uses max)
- Update notebook structure description to match actual cell layout

### 5.2 Add Evolution Notes to Notebook Markdown Cells

Add markdown cells documenting what changed from Run01 → v8 and why. This creates a self-contained narrative within the notebook.

---

## Change Summary

| Phase | Changes | Priority | Dependencies |
|---|---|---|---|
| 1 | pos_weight, scheduler, cudnn fix | P0 | None |
| 2 | Augmentation expansion | P1 | None (parallel with Phase 1) |
| 3 | Evaluation reporting | P0/P1 | None (parallel with Phase 1) |
| 4 | Per-sample Dice, logging | P1 | Phase 1 (need stable training) |
| 5 | Documentation | P1 | Phase 1–4 (document final state) |

---

## Pre-Flight Checklist Before Running v8

- [ ] `pos_weight` computed and passed to BCEWithLogitsLoss
- [ ] `ReduceLROnPlateau` scheduler added and stepping on val_f1
- [ ] `cudnn.benchmark` contradiction resolved
- [ ] Training augmentations include ColorJitter, ImageCompression, GaussNoise
- [ ] Validation/test transforms remain unchanged (no augmentation)
- [ ] Threshold sweep range starts at 0.05 (not 0.1)
- [ ] Tampered-only metrics reported first in evaluation output
- [ ] Per-forgery-type F1 in main evaluation output
- [ ] Per-sample Dice loss implemented
- [ ] Gradient norm logging enabled
- [ ] LR logging per epoch enabled
- [ ] W&B run name clearly identifies this as "Run02-v8"

## Post-Run Validation Checklist

- [ ] Optimal threshold is in 0.25–0.55 range (was 0.1327)
- [ ] Val loss does not diverge >30% above minimum
- [ ] Training runs for >20 useful epochs (was 15)
- [ ] Tampered-only F1 > 0.50 (was 0.41)
- [ ] Copy-move F1 > 0.35 (was 0.31)
- [ ] JPEG QF50 F1 within 0.05 of clean F1 (was 0.13 gap)
- [ ] Gradient norms are stable (no spikes >10×)
