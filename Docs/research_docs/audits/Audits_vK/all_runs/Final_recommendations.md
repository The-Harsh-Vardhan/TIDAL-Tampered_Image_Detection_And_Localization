# Final Recommendations

**Based on Technical Audit of All Runs (vK.1 → vK.10.6)**
**Date:** 2026-03-14
**Priority:** Ordered by expected impact on assignment score

---

## The Real Story

v6.5 already implemented the most impactful architectural decision — a pretrained ResNet34 encoder via SMP — and achieved Tam-F1=0.41 with a comprehensive evaluation suite. v8 added important methodology improvements (scheduler, augmentations, shortcut detection, per-sample Dice) but broke the model with pos_weight=30.01 and 16× batch increase. The vK.10.x series abandoned the pretrained approach, reverting to training from scratch.

**vK.10.6 is the turning point for the vK.x series:** by running 100 epochs with patience=30 (vs vK.10.5's ~10 epochs with patience=10), it achieved Tam-F1=0.22 and the best classification metrics of any vK.x run (AUC=0.91, Acc=0.84). It also added the most comprehensive evaluation suite in the project (12 features including confusion matrix, PR curves, pixel-AUC). However, segmentation still trails v6.5 by 50% — confirming that pretrained encoders remain essential.

**The recommendations below reflect the post-vK.10.6 landscape.**

---

## Priority 0 — Adopt v6.5's Architecture in vK.10.x

### R1: Port SMP Pretrained Encoder to the vK.10.x Codebase

**Impact:** Expected to improve vK.10.6 from F1=0.22 → 0.40+ (already proven in v6.5)
**Effort:** Medium — requires merging v6.5's model setup into vK.10.6's training infrastructure
**Risk:** Zero — v6.5 already demonstrated this works

The vK.10.6 codebase now has excellent engineering (CONFIG, AMP, seeding, DataParallel, `get_base_model()`) AND the most comprehensive evaluation suite. It just needs the pretrained encoder to unlock meaningful segmentation:

```python
# In vK.10.x's CONFIG:
CONFIG = {
    'model': 'smp.Unet',
    'encoder': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,
    'image_size': 384,
    # ... keep vK.10.x's training infrastructure
}
```

**What v6.5 already solved:**
- SMP model initialization with pretrained weights
- DataParallel prefix handling for SMP models
- Image-level classification from max pixel probability (no separate classifier head needed)
- Differential learning rates (encoder: 1e-4, decoder: 1e-3)

---

## Priority 1 — Fix v8's Mistakes While Keeping Its Improvements

### R2: Fix pos_weight Computation

**Impact:** Removes the primary cause of v8's regression
**Current:** pos_weight=30.01 (computed from ALL pixels including authentic images)
**Fix:** Compute on tampered images only → expected pos_weight ~3–5×

```python
# WRONG (v8's approach):
pos_weight = total_bg_pixels / total_fg_pixels  # 30.01 (includes authentic)

# CORRECT:
tampered_masks = [m for m in train_masks if m.sum() > 0]
fg = sum(m.sum() for m in tampered_masks)
bg = sum(m.numel() - m.sum() for m in tampered_masks)
pos_weight = bg / fg  # ~3-5 (tampered images only)
```

### R3: Adopt v8's Improvements Without Its Batch Size Error

**Impact:** Addresses v6.5's genuine weaknesses
**What to adopt from v8:**
- ReduceLROnPlateau(patience=3, factor=0.5) — fills v6.5's biggest training gap
- Per-sample Dice loss — fixes batch-level bias toward large masks
- Expanded augmentations: ColorJitter, ImageCompression, GaussNoise, GaussianBlur
- Encoder warmup option (freeze encoder for first 3–5 epochs)

**What NOT to adopt from v8:**
- Effective batch size 256 without LR rescaling (keep at 16–32)
- pos_weight=30.01 (see R2)

### R4: Apply LR Scaling if Increasing Batch Size

**Lesson from v8:** If increasing effective batch from 16 to N, scale LR proportionally:

```python
base_lr = 1e-4  # for effective batch 16
scale = effective_batch / 16
encoder_lr = base_lr * scale
decoder_lr = 1e-3 * scale
```

---

## Priority 2 — Complete the Evaluation Suite

### R5: Add Confusion Matrix + PR Curves

**Impact:** Assignment compliance
**Status:** **DONE in vK.10.6** — confusion matrix, ROC curve, and PR curve with AP annotation are all implemented and producing results. This recommendation is now fulfilled.

### R6: Fix the Robustness Evaluation Bug in v6.5

**Impact:** Makes robustness results trustworthy
**Issue:** v6.5's robustness evaluation produces identical F1=0.5938 for jpeg_qf50, gaussian_noise_light, and gaussian_noise_heavy — statistically impossible for different degradations
**Fix:** Debug the perturbation application pipeline. Verify that transforms are actually modifying the input tensors (add a visual sanity check).

v8's robustness results are more plausible (different F1 values for different degradations), suggesting the bug was fixed in v8.

### R7: Adopt v8's Shortcut Learning Validation

**Impact:** Validates model integrity, interview-impressive
**Status:** **DONE in vK.10.6** — mask randomization (F1=0.0658, PASS) and boundary erosion (delta=+0.0007, PASS) are implemented. Both v8 and vK.10.6 now have this.

---

## Priority 3 — Push Beyond v6.5's F1=0.41

### R8: Add ELA (Error Level Analysis) as 4th Input Channel

**Impact:** 10–20% F1 improvement on JPEG-compressed forgeries
**Effort:** ~30 lines + change `in_channels=4`
**Not tried in any run**

```python
def compute_ela(image_path, quality=90):
    img = Image.open(image_path)
    buffer = io.BytesIO()
    img.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    ela = np.abs(np.float32(img) - np.float32(Image.open(buffer)))
    return ela.mean(axis=2)  # single-channel ELA map
```

### R9: Edge-Aware Loss

**Impact:** Better boundary delineation around tampered regions
**Not tried in any run**

```python
edges = sobel(gt_mask)
boundary_weight = 1.0 + 5.0 * edges
seg_loss = (bce_loss * boundary_weight).mean()
```

### R10: Test-Time Augmentation (TTA)

**Impact:** ~2–3% F1 boost for free at inference time
**Not tried in any run**

```python
pred1 = model(image)
pred2 = torch.flip(model(torch.flip(image, [3])), [3])
final = (pred1 + pred2) / 2
```

### R11: Multi-Scale Input

Use input at 384×384 and 512×512, average predictions from both scales.

---

## What NOT to Change

1. **Keep v6.5's SMP pretrained encoder approach** — the single most impactful decision in the project
2. **Keep vK.10.6's evaluation suite** — the most comprehensive in the project: 12 features including confusion matrix, PR curves, pixel-AUC, shortcut detection
3. **Keep vK.10.x's CONFIG dict system** — well-designed centralized configuration
4. **Keep vK.10.5/vK.10.6's DataParallel + `get_base_model()`** — properly implemented multi-GPU support
5. **Keep AMP** — free speedup, present in both tracks
6. **Keep v8's per-sample Dice** — corrects v6.5's batch-level bias
7. **Keep v8's expanded augmentations** — proven JPEG robustness improvement
8. **Keep vK.10.6's 100 epochs + patience=30** — proved from-scratch model needs long training

---

## Recommended Next Run Configuration

Merge the best of v6.5, v8, and vK.10.x:

```python
CONFIG = {
    # Architecture (from v6.5)
    'model': 'smp.Unet',
    'encoder': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 3,
    'classes': 1,
    'image_size': 384,

    # Training (from vK.10.x + v8 fixes)
    'optimizer': 'AdamW',
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'scheduler': 'ReduceLROnPlateau',   # from v8
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,
    'batch_size': 4,                     # keep v6.5's batch
    'accumulation_steps': 4,             # effective 16
    'max_epochs': 50,
    'amp': True,

    # Loss (from v8, with pos_weight fix)
    'seg_loss': 'bce_dice',
    'dice_mode': 'per_sample',           # from v8
    'pos_weight': 4.0,                   # FIXED from v8's 30.01

    # Early Stopping (from vK.10.x)
    'early_stop_metric': 'val_f1_tampered',
    'early_stop_patience': 10,

    # Encoder Freezing (from v8 infrastructure)
    'freeze_encoder_epochs': 3,

    # Augmentations (from v8)
    'augmentations': ['HFlip', 'VFlip', 'Rotate90',
                      'ColorJitter', 'ImageCompression',
                      'GaussNoise', 'GaussianBlur'],
}
```

---

## Expected Impact Summary

| Recommendation | Baseline | Expected F1 | Effort | Risk |
|---|---|---|---|---|
| R1: Port SMP to vK.10.6 | vK.10.6: 0.22 | 0.40–0.50 (match/exceed v6.5) | Medium | Zero |
| R2: Fix pos_weight | v8: 0.2949 | 0.40–0.45 (exceed v6.5) | Trivial | Zero |
| R3: Adopt v8 improvements | v6.5: 0.4101 | 0.43–0.48 | Low | Low |
| R5: Confusion matrix + PR | ~~missing~~ | **DONE in vK.10.6** | — | — |
| R6: Fix robustness bug | v6.5: buggy | Trustworthy results | Low | Zero |
| R7: Shortcut detection | ~~missing~~ | **DONE in vK.10.6** | — | — |
| R8: ELA input | v6.5: 0.41 | 0.45–0.55 | Medium | Low |
| R9: Edge-aware loss | v6.5: 0.41 | 0.43–0.46 | Low | Low |
| **Combined R1–R3+R8** | **vK.10.6: 0.22** | **0.50–0.65** | **1 day** | **Low** |

---

## Bottom Line

**vK.10.6 is the project's most complete notebook** — excellent classification (AUC=0.91), comprehensive evaluation (12 features no other run has), and proof that the from-scratch model can learn with sufficient training time. But segmentation at F1=0.22 still trails v6.5's pretrained F1=0.41 by 50%.

The project is now one architectural change away from a strong submission: **port v6.5's pretrained ResNet34 into vK.10.6's codebase**. The evaluation suite, engineering infrastructure, and training config are all ready. The model just needs features it can't learn from scratch on 8,829 images.

**The path forward is synthesis:**
1. Start from vK.10.6's proven codebase (best engineering + evaluation)
2. Replace custom UNet with v6.5's `smp.Unet(encoder_name='resnet34', encoder_weights='imagenet')`
3. Add v8's improvements (ReduceLROnPlateau, per-sample Dice, expanded augmentations) with corrected hyperparameters
4. Push beyond 0.41 with ELA input and edge-aware loss
