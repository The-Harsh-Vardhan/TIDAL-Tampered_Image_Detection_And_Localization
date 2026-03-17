# Audit 6.5 — Part 8: Recommended Fixes

## Priority Classification

- **P0 (Critical):** Must fix before next training run
- **P1 (Important):** Should fix to significantly improve results
- **P2 (Moderate):** Will improve quality/reliability
- **P3 (Nice-to-have):** Polish and best practices

---

## P0 — Critical Fixes

### Fix 1: Add Learning Rate Scheduler

**Problem:** No learning rate scheduler is used. The constant decoder LR=1e-3 causes the model to overshoot optimal weights after epoch 15, leading to severe overfitting (val loss doubles from 0.77 to 1.20 while train loss continues to decrease).

**Why it matters:** This is likely the single biggest performance limiter. The model converges to F1=0.73 by epoch 11–15 but cannot fine-tune further because the learning rate is too aggressive.

**Fix:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
)
# After validation each epoch:
scheduler.step(val_f1)
```

Alternative: `CosineAnnealingWarmRestarts` for more aggressive exploration.

---

### Fix 2: Add BCE pos_weight for Class Imbalance

**Problem:** The BCE component of the loss treats all pixels equally. Since tampered pixels are typically 2–10% of each image, the gradient is dominated by background pixels, pushing the model toward conservative (low-probability) predictions.

**Why it matters:** This explains the extremely low optimal threshold (0.1327) and the weak tampered-only F1 (0.41). The model is being trained to predict "not tampered" for most pixels.

**Fix:**
```python
# Compute pos_weight from training set
total_fg, total_bg = 0, 0
for pair in train_pairs:
    if pair['mask_path']:
        mask = cv2.imread(pair['mask_path'], cv2.IMREAD_GRAYSCALE)
        fg = (mask > 0).sum()
        total_fg += fg
        total_bg += mask.size - fg
pos_weight = torch.tensor([total_bg / max(total_fg, 1)]).to(device)

# In BCEDiceLoss:
self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

### Fix 3: Report Tampered-Only Metrics Prominently

**Problem:** Mixed-set Pixel-F1 (0.72) is reported as the primary metric, but it's inflated by authentic images automatically scoring F1=1.0. The true localization quality is tampered-only F1 (0.41).

**Why it matters:** Presenting inflated metrics could mislead reviewers or stakeholders about actual model performance.

**Fix:** In the evaluation output cell, lead with tampered-only metrics:
```python
print(f'\nPRIMARY METRIC — Tampered-only ({test_results["num_tampered_images"]} images):')
print(f'  Pixel-F1:  {test_results["tampered_f1_mean"]:.4f}')
print(f'\nSecondary — Mixed-set ({test_results["num_test_images"]} images):')
print(f'  Pixel-F1:  {test_results["pixel_f1_mean"]:.4f} (includes authentic)')
```

---

## P1 — Important Fixes

### Fix 4: Strengthen Data Augmentation

**Problem:** Only geometric augmentations (flip, rotate) are used. No photometric or noise augmentations. This contributes to overfitting and compression artifact dependency.

**Why it matters:** The model overfits by epoch 15 and has 13% F1 drop under JPEG compression.

**Fix:**
```python
train_transform = A.Compose([
    A.Resize(CONFIG['image_size'], CONFIG['image_size']),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # Add these:
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

### Fix 5: Investigate Copy-Move Failure

**Problem:** Copy-move F1=0.31 (near-random) while splicing F1=0.59 is reasonable. The model fails systematically on copy-move forgeries.

**Why it matters:** Copy-move represents 64% of tampered images (3,295 of 5,123). Poor copy-move performance tanks the overall tampered-only F1.

**Fix:**
1. Visualize worst copy-move predictions to understand failure mode
2. Consider copy-move-specific data augmentation (random patch duplication)
3. Evaluate whether SRM (Spatial Rich Model) noise features help detect copy-move boundaries
4. Consider dedicated copy-move detection branches or attention modules

### Fix 6: Fix cudnn.benchmark Contradiction

**Problem:** `set_seed()` sets `cudnn.benchmark = False` for reproducibility, but `setup_device()` immediately overrides it to `True`.

**Why it matters:** Bit-level reproducibility is compromised without the user knowing.

**Fix:** Choose one behavior:
```python
# Option A: Prioritize reproducibility
def setup_device(config):
    # ... GPU detection ...
    # Don't override benchmark setting from set_seed()
    pass

# Option B: Prioritize performance (current behavior, but make explicit)
CONFIG['cudnn_benchmark'] = True  # Set in CONFIG, not two places
```

---

## P2 — Moderate Fixes

### Fix 7: Save Training History in Checkpoints

**Problem:** The `history` dict is not included in checkpoint state. If training resumes from `last_checkpoint.pt`, the history is empty.

**Fix:**
```python
state = {
    # ...existing fields...
    'history': history,
}
```

### Fix 8: Add Warmup Phase

**Problem:** Training starts at full learning rate, causing large loss oscillations in early epochs.

**Fix:**
```python
from torch.optim.lr_scheduler import LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=CONFIG['max_epochs'] - 5)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])
```

### Fix 9: Increase DataLoader Workers

**Problem:** `num_workers=2` is conservative. The T4 has sufficient CPU cores for more workers.

**Fix:** Set `num_workers=4` or `min(4, os.cpu_count())`.

---

## P3 — Nice-to-Have

### Fix 10: Add Cross-Dataset Evaluation

Test on Coverage, CoMoFoD, or Columbia datasets to assess generalization beyond CASIA.

### Fix 11: Add per-image Metric Distribution Plots

Histogram of per-image F1 scores would make the high variance (±0.41) more visible and interpretable.

### Fix 12: Consider Focal Loss

Replace BCE with Focal Loss for better handling of easy vs hard examples:
```python
# alpha-balanced focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()
```

---

## Expected Impact

| Fix | Expected F1 Improvement | Effort |
|---|---|---|
| LR Scheduler | +0.05–0.10 | Low (3 lines) |
| BCE pos_weight | +0.05–0.08 | Low (10 lines) |
| Stronger augmentation | +0.03–0.07 | Low (10 lines) |
| Copy-move investigation | +0.05–0.15 (on copy-move) | Medium |
| All combined | +0.10–0.20 (tampered-only F1) | Low-Medium |

**Estimated improved tampered-only F1: 0.50–0.60** (from current 0.41)
