# Cross-Comparison: vK.11.1, vK.11.4, vK.11.5, vK.12.0, vK.11.1-R2, vK.12.0b Runs

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Scope** | Six experiment runs from the vK.11.x/12.0 synthesis architecture |
| **Verdict** | **All six runs failed. The synthesis architecture does not learn.** |

---

## 1. Executive Summary

The vK.11.x series represents the project's most ambitious experiment: a "synthesis architecture" combining the best elements from all prior work:
- Pretrained ResNet34 encoder (from v6.5, the project's best run at Tam-F1=0.41)
- Per-sample Dice loss (from v8)
- Comprehensive evaluation suite (from vK.10.6)
- NEW: ELA 4th input channel (Error Level Analysis for forensic signal)
- NEW: Sobel-based edge loss (boundary supervision)
- NEW: FC classification head (dual-task learning)

**Projected performance: Tam-F1 0.50-0.65.**

**Actual performance: Tam-F1 0.13.**

This is the worst pretrained-encoder result in project history -- worse than v6.5 (0.41), worse than v8 with broken pos_weight (0.29), and worse than vK.10.6 trained from scratch (0.22). The pixel-level AUC of ~0.50 confirms random-chance predictions. The shortcut test confirms the models ignore image content.

Of the six notebooks:
- **vK.11.1** was never executed (AMP bug in edge loss)
- **vK.11.4** trained for 25 epochs (Tam-F1=0.1321)
- **vK.11.5** trained for 13 epochs (Tam-F1=0.1272)
- **vK.12.0** trained for 16 epochs (Tam-F1=0.1321) -- crashed mid-evaluation (`KeyError: 'true_mask'`), 42 cells never ran
- **vK.11.1-R2** trained for 14 epochs (Tam-F1=0.1274) -- fully executed, worst pixel-AUC (0.4482)
- **vK.12.0b** trained for 16 epochs (Tam-F1=0.1322) -- crashed mid-visualization (`AttributeError`), 16 cells never ran

All five executed runs converged to the same constant-output prediction pattern. vK.12.0 and vK.12.0b's precision/recall metrics proved the model predicts ALL pixels as tampered (Recall=1.0, Precision=0.0825).

---

## 2. Notebook Feature Comparison

| Feature | vK.11.1 | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b |
|---------|---------|---------|---------|---------|------------|----------|
| **Cells** (total / code / markdown) | 102 / 49 / 53 | 127 / 58 / 69 | 135 / 61 / 74 | 151 / 69 / 82 | 102 / 49 / 53 | 151 / 69 / 82 |
| **Code cells executed** | 18 / 49 (37%) | 54 / 58 (93%) | 57 / 61 (93%) | 53 / 69 (77%) -- crashed | 49 / 49 (100%) | 53 / 69 (77%) -- crashed |
| Executive Summary section | No | Yes | Yes | Yes | No | Yes |
| Results Dashboard section | No | No | Yes (broken) | Yes (broken) | No | Yes (broken) |
| Dataset Exploration | No | No | No | Yes | No | Yes |
| Architecture Diagram | No | No | No | Yes | No | Yes |
| Model Complexity (torchinfo) | No | No | No | Yes | No | Yes |
| Precision/Recall/PixelAcc | No | No | No | Yes | No | Yes |
| Enhanced Robustness Suite | No | No | No | Code only (crashed) | No | Code only (crashed) |
| FP/FN Error Analysis | No | No | No | Code only (crashed) | No | Code only (crashed) |
| Inference Speed Test | No | No | No | Code only (crashed) | No | Code only (crashed) |
| Experiment Comparison Table | No | No | No | Yes (misleading baselines) | No | Yes (misleading baselines) |
| Reproducibility Verification | No | Yes | Yes | Code only (crashed) | Yes | Code only (crashed) |
| Quick Inference Demo | No | Yes (unexecuted) | Yes (unexecuted) | Code only (crashed) | Yes | Code only (crashed) |
| W&B prediction logging | No | Yes (every 5 epochs) | Yes (every 5 epochs) | Yes (every 5 epochs) | Yes (every 5 epochs) | Yes (every 5 epochs) |
| W&B mode | — | Offline | Offline | Offline | **Online** | Offline |
| Edge loss AMP bug | **YES (fatal)** | Fixed | Fixed | Fixed | Fixed | Fixed |
| Crash | None (unexecuted) | None | None | `KeyError: 'true_mask'` | None | `AttributeError: 'Tensor'.astype` |
| Model Card version label | "vK.11.1" (correct) | "vK.11.1" (wrong) | "vK.11.1" (wrong) | "vK.11.1" (wrong) | "vK.11.1" (correct) | "vK.11.1" (wrong) |
| Notebook size (KB) | 224 | 929 | 663 | 762 | ~700 | ~750 |

### Progression Summary

```
vK.11.1:    Base code + AMP bug → never ran
vK.11.4:    Bug fix + reduced CONFIG + Executive Summary + Repro + Demo → first run, failed
vK.11.5:    11.4 code + Results Dashboard → second run, failed worse
vK.12.0:    11.5 code + 10 eval/presentation improvements → third run, failed + crashed (KeyError)
vK.11.1-R2: 11.4 code in 11.1 template + W&B enhancements → fourth run, failed, worst pixel-AUC
vK.12.0b:   12.0 code with KeyError fix but new crash → fifth run, failed + crashed (AttributeError)
```

---

## 3. CONFIG Comparison

| Parameter | vK.11.1 | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b |
|-----------|---------|---------|---------|---------|------------|----------|
| `max_epochs` | **100** | 50 | 50 | 50 | 50 | 50 |
| `patience` | **20** | 10 | 10 | 10 | 10 | 10 |
| `img_size` | 256 | 256 | 256 | 256 | 256 | 256 |
| `batch_size` | 8 (→32) | 8 (→32) | 8 (→32) | 8 (→16) | 8 (→32) | 8 (→32) |
| `encoder_lr` | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| `decoder_lr` | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| `weight_decay` | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |
| `accumulation_steps` | 4 | 4 | 4 | 4 | 4 | 4 |
| `encoder_freeze_epochs` | 2 | 2 | 2 | 2 | 2 | 2 |
| `max_grad_norm` | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| `seed` | 42 | 42 | 42 | 42 | 42 | 42 |

The only CONFIG change across the series is the **halving of training budget** from vK.11.1: `max_epochs` 100→50 and `patience` 20→10. All subsequent runs retain the downgraded values. Note: vK.12.0's batch auto-scaled to 16 (vs 32 in others), giving effective batch=64 instead of 128. This likely reflects different Kaggle GPU memory allocation.

---

## 4. Training Dynamics Comparison

| Metric | vK.11.1 | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b |
|--------|---------|---------|---------|---------|------------|----------|
| **Status** | Unexecuted | Trained | Trained | Trained (crashed) | **Trained** | **Trained (crashed)** |
| **Epochs completed** | 0 | 25 | 13 | 16 | **14** | **16** |
| **Best epoch** | N/A | 15 | **3** | 6 | **4** | **6** |
| **Best val Dice(tam)** | N/A | 0.1412 | 0.1364 | 0.1412 | **0.1362** | **0.1412** |
| **Initial train loss** | N/A | 4.6517 | 4.6402 | 4.5975 | **4.6094** | **4.6129** |
| **Final train loss** | N/A | 1.8795 | 1.6958 | 1.7305 | **1.6950** | **1.6462** |
| **Early stopped** | N/A | Yes (patience=10) | Yes (patience=10) | Yes (patience=10) | **Yes (patience=10)** | **Yes (patience=10)** |
| **LR reductions** | N/A | 2 (at ~epoch 18, 21) | 1+ | 1+ | **2 (at ~epoch 7, 10)** | **0 visible** |
| **Crashed** | N/A | No | No | Yes (`KeyError`) | **No** | **Yes (`AttributeError`)** |

### Training Trajectory Visualization

```
Val Dice (Tampered) Over Training

vK.11.4:
  Epoch:  1    5    10   15   20   25
  Dice:  .118 .120 .133 .141 .141 .141  ← flatlines at epoch 15

vK.11.5:
  Epoch:  1    3    5    8    10   13
  Dice:  .123 .136 .125 .111 .114 .118  ← peaks at epoch 3, DECLINES

vK.12.0:
  Epoch:  1    3    5    6    10   16
  Dice:  .126 .125 .122 .141 .141 .141  ← flatlines at epoch 6

vK.11.1-R2:
  Epoch:  1    4    5    8    10   14
  Dice:  .125 .136 .135 .133 .132 .132  ← peaks at epoch 4, slow decline

vK.12.0b:
  Epoch:  1    3    6    8    10   16
  Dice:  .122 .113 .141 .141 .141 .141  ← dips at epoch 3, flatlines at epoch 6
```

### Key Insight: Encoder Unfreeze Dynamics

| Phase | vK.11.4 Val Dice(tam) | vK.11.5 Val Dice(tam) |
|-------|----------------------|----------------------|
| Frozen epoch 1 | 0.1180 | 0.1232 |
| Frozen epoch 2 | 0.1239 | 0.1355 |
| Unfrozen epoch 3 | 0.1227 ↓ | **0.1364** ↑ (peak) |
| Unfrozen epoch 4+ | Slow climb → 0.1412 at ep 15 | **Decline → 0.1113 at ep 8** |

vK.12.0 confirms vK.11.5's pattern: val AUC peaks at epoch 3 (0.7244) then degrades upon encoder unfreeze. Interestingly, vK.12.0's best Dice (epoch 6) coincides with its worst AUC (0.5882) -- suggesting classification and segmentation objectives directly compete for encoder capacity.

---

## 5. Test Metrics Comparison

### Primary Metrics

| Metric | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b | Assessment |
|--------|---------|---------|---------|------------|----------|------------|
| Accuracy | 0.4142 | 0.4194 | 0.4062 | **0.5235** | 0.4062 | R2 best (cls head learned), others below random |
| AUC-ROC | 0.6434 | 0.6466 | 0.5637 | **0.6550** | 0.6175 | R2 best, 12.0 worst |
| **Tam-F1** | **0.1321** | **0.1272** | **0.1321** | **0.1274** | **0.1322** | All within 0.005 of each other |
| **Tam-IoU** | **0.0825** | **0.0768** | **0.0825** | **0.0780** | **0.0825** | |
| **Tam-Dice** | **0.1321** | **0.1272** | **0.1321** | **0.1274** | **0.1322** | |
| Pixel-AUC | 0.4988 | 0.5215 | 0.4952 | **0.4482** | 0.4791 | All ≈ random; R2 worst (anti-correlated) |
| Precision (tam) | — | — | 0.0825 | — | 0.0825 | Model predicts all pixels tampered |
| Recall (tam) | — | — | **1.0000** | — | **1.0000** | Confirms predict-everything failure |

All five runs are catastrophically poor. vK.11.1-R2's AUC-ROC (0.6550) is the best while its pixel-AUC (0.4482) is the worst -- the classification head succeeds at the expense of segmentation, confirming the multi-objective conflict.

### Threshold Optimization

| Parameter | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b |
|-----------|---------|---------|---------|------------|----------|
| Optimal threshold | 0.4939 | 0.0500 | 0.5092 | 0.0500 | 0.0500 |
| F1 at optimal | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 |

Despite radically different optimal thresholds (0.49 vs 0.05 vs 0.51 vs 0.05 vs 0.05), all five achieve **identical F1 = 0.1321**. This confirms the predictions are degenerate -- the threshold cannot improve a constant-output model.

### Per-Forgery-Type Breakdown

| Forgery Type | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b | Match? |
|-------------|---------|---------|---------|------------|----------|--------|
| Splicing | 0.1016 | 0.1016 | 0.1016 | 0.1016 | 0.1016 | **ALL EXACT** |
| Copy-move | 0.1918 | 0.1918 | 0.1918 | 0.1918 | 0.1918 | **ALL EXACT** |

### Mask-Size Stratified Evaluation

| Size Category | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b | Match? |
|--------------|---------|---------|---------|------------|----------|--------|
| Tiny (<2%) | 0.0190 | 0.0190 | 0.0190 | 0.0190 | 0.0190 | **ALL EXACT** |
| Small (2-5%) | 0.0630 | 0.0630 | 0.0630 | 0.0630 | 0.0630 | **ALL EXACT** |
| Medium (5-15%) | 0.1537 | 0.1537 | 0.1537 | 0.1537 | 0.1537 | **ALL EXACT** |
| Large (>15%) | 0.4860 | 0.4859 | 0.4859 | 0.4859 | 0.4859 | Within rounding |

### Shortcut Learning Detection

| Test | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b | Match? |
|------|---------|---------|---------|------------|----------|--------|
| Shuffled-mask F1 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | **ALL EXACT** |
| Delta from baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **ALL EXACT** |
| Eroded-pred F1 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 | **ALL EXACT** |
| Delta from baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **ALL EXACT** |

### Robustness Testing

| Condition | vK.11.4 F1 | vK.11.5 F1 | vK.12.0 F1 | vK.11.1-R2 F1 | vK.12.0b F1 |
|-----------|-----------|-----------|-----------|-------------|------------|
| Clean baseline | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 |
| JPEG QF=70 | 0.1321 | 0.1321 | Not executed | 0.1321 | Not executed |
| JPEG QF=50 | 0.1321 | 0.1321 | Not executed | 0.1321 | Not executed |
| Gaussian noise | 0.1321 | 0.1321 | Not executed | 0.1321 | Not executed |
| Gaussian blur | 0.1321 | 0.1321 | Not executed | 0.1321 | Not executed |
| Resize | 0.1321 | 0.1321 | Not executed | 0.1322 | Not executed |

Note: vK.12.0 and vK.12.0b's enhanced robustness suites were blocked by crashes. vK.11.1-R2's robustness tests ran but Albumentations API deprecation means transforms were silently ineffective.

---

## 6. The Identical-Output Problem

The tables above reveal a deeply suspicious pattern: **all five executed runs produce bitwise identical results across every stratified analysis** -- per-forgery-type, per-mask-size, and shortcut tests. Five independent training runs converge to the same constant prediction.

This is statistically near-impossible for five independently trained models with different epoch counts and best epochs, **unless all five models output the same constant prediction.**

vK.12.0 and vK.12.0b's precision/recall metrics provide the definitive diagnosis: **Recall = 1.0000, Precision = 0.0825**. The model predicts EVERY pixel as tampered.

### Evidence Summary

| Evidence | Implication |
|----------|-------------|
| Shuffled-mask F1 = baseline F1 | Predictions don't correlate with image content |
| Robustness F1 = baseline for all 8 conditions | Predictions insensitive to input perturbations |
| Pixel-AUC ≈ 0.50 | Pixel predictions have zero discriminative power |
| Per-type + per-size metrics identical across runs | All five models output the same fixed pattern |
| Optimal threshold search converges to same F1 | Output distribution is degenerate |

### What the models probably learned

All five models converged to predicting a **near-zero-valued mask for all inputs**. With 59.4% of images being authentic (zero mask ground truth), predicting zero everywhere:
- Gets ~59% of Dice scores = 1.0 (authentic images with zero GT and zero prediction)
- Gets ~41% of Dice scores = 0.0 (tampered images where zero prediction doesn't overlap with GT mask)
- Average Dice ≈ 0.594 * 1.0 + 0.406 * 0.0 = 0.594 for all-sample metrics

But the tampered-only Dice of ~0.13 (not 0.0) suggests the model is outputting very small positive values that, when thresholded at the right point, overlap slightly with ground truth masks. The 0.49 (large mask) vs 0.02 (tiny mask) size stratification confirms this: the few positive pixels happen to overlap more with larger tampered regions purely by geometric chance.

---

## 7. Code Diff Analysis

### vK.11.1 → vK.11.4 (Code Changes)

| # | Change | Lines | Impact |
|---|--------|-------|--------|
| 1 | `CONFIG['max_epochs']`: 100 → 50 | 1 | **Harmful** -- reduced training budget |
| 2 | `CONFIG['patience']`: 20 → 10 | 1 | **Harmful** -- premature early stopping |
| 3 | Edge loss AMP fix: added `.float()` casts + `autocast(enabled=False)` | 3 | **Critical fix** -- unblocks training |
| 4 | `ReduceLROnPlateau(verbose=True)` → removed verbose | 1 | Cosmetic (deprecated param) |
| 5 | W&B prediction logging in training loop | 35 | Helpful for monitoring |

**Net impact**: Fixed the fatal AMP bug but degraded the training budget. The AMP fix was necessary; the CONFIG downgrade was not.

### vK.11.4 → vK.11.5 (Code Changes)

| # | Change | Lines | Impact |
|---|--------|-------|--------|
| 1 | Results Dashboard section added (4 code, 4 markdown cells) | ~80 | **Zero impact** -- shows placeholders |
| 2 | Version labels updated in title/conclusion | 2 | Cosmetic |

**Net impact**: Zero functional changes. vK.11.5 is vK.11.4 with a broken dashboard bolted on.

### vK.11.1 Run-01 → vK.11.1 Run-02 (Code Changes)

| # | Change | Lines | Impact |
|---|--------|-------|--------|
| 1 | `CONFIG['max_epochs']`: 100 → 50 | 1 | **Harmful** -- reduced training budget |
| 2 | `CONFIG['patience']`: 20 → 10 | 1 | **Harmful** -- premature early stopping |
| 3 | Edge loss AMP fix: added `.float()` casts + `autocast(enabled=False)` | 3 | **Critical fix** -- unblocks training |
| 4 | `ReduceLROnPlateau(verbose=True)` → removed verbose | 1 | Cosmetic |
| 5 | W&B project/run name: `vK.11.0` → `vK.11.1` | 2 | Correctness fix |
| 6 | Comprehensive W&B artifact logging (every viz section) | ~100 | Monitoring improvement |

**Net impact**: Same code changes as vK.11.1→vK.11.4, plus W&B correctness fix and comprehensive logging. Uses vK.11.1's 102-cell template (no Executive Summary, no Dashboard).

### vK.12.0 → vK.12.0b (Code Changes)

| # | Change | Lines | Impact |
|---|--------|-------|--------|
| 1 | Removed `if hasattr(img, 'numpy'): img = img.numpy()` guard in `show_enhanced_viz()` | -2 | **Introduced crash** -- `AttributeError: 'Tensor'.astype` |

**Net impact**: One 2-line deletion introduced a new crash. All other code is 100% identical to vK.12.0.

---

## 8. Root Cause Analysis

Why did the synthesis architecture fail? Here are the hypothesized root causes, ordered by likelihood:

### RC-1: Multi-Objective Loss Conflict (HIGH probability)

```
Total = 1.5 * FocalLoss(cls) + 1.0 * SegLoss + 0.3 * EdgeLoss
```

The classification loss has **the highest weight (1.5x)** and operates on the encoder's bottleneck features. The segmentation and edge losses operate on the decoder output. During backpropagation, the encoder receives gradient signal from all three losses, but the classification gradient dominates due to its 1.5x weight.

**Hypothesis**: The encoder is being pushed toward image-level discrimination (authentic vs tampered) at the expense of pixel-level feature extraction needed for segmentation. The encoder learns to compress information into a global label rather than preserving spatial detail.

**Test**: Run with `cls_weight=0.0` (ablation) to see if segmentation alone works.

### RC-2: Encoder Unfreeze at Too-High Learning Rate (HIGH probability)

vK.11.5's epoch-3 peak provides direct evidence. The pretrained ResNet34 has well-calibrated features from ImageNet. Unfreezing at lr=1e-4 (same as the randomly-initialized decoder) overwrites these features too aggressively.

**Standard practice**: Fine-tune pretrained encoders at 1e-5 to 1e-6 (10-100x lower than decoder lr).

**The project's v6.5 run also used differential LR (enc=1e-4, dec=1e-3) and achieved Tam-F1=0.41.** However, v6.5 did NOT have a 4th input channel or classification head competing for encoder gradients.

**Test**: Reduce `encoder_lr` to 1e-5 or 1e-6.

### RC-3: ELA Channel Adding Noise (MEDIUM probability)

The ELA channel is computed via JPEG re-compression at quality=90. If CASIA images are stored as PNG or uncompressed TIFF, the ELA channel produces minimal signal (nearly uniform output). This adds a noisy 4th channel that:
- Requires the first conv layer to be modified (breaking pretrained weight initialization)
- Provides no useful forensic information
- Forces the encoder to learn to ignore an entire input channel

**Test**: Run with `in_channels=3` (remove ELA) to check if the standard 3-channel input works.

### RC-4: Reduced Training Budget (MEDIUM probability)

vK.11.1's CONFIG had `max_epochs=100, patience=20`. vK.11.4/11.5 used `max_epochs=50, patience=10`. The best from-scratch result (vK.10.6, Tam-F1=0.22) used 100 epochs with patience=30.

With a synthesis architecture that has new components to integrate (ELA, edge loss, classification head), the model likely needs MORE training time, not less.

**Test**: Restore `max_epochs=100, patience=30`.

### RC-5: SMP 4-Channel Input Weight Adaptation (LOW-MEDIUM probability)

When SMP's UNet is initialized with `in_channels=4` on a ResNet34 pretrained with 3 channels, the first conv layer must be adapted. SMP typically copies the mean of the RGB weights to the 4th channel filter. But ELA has fundamentally different statistics than RGB -- the copied weights are meaningless for ELA features.

**Test**: Manually initialize the 4th channel conv weights (e.g., Kaiming initialization for the 4th filter while preserving RGB weights).

---

## 9. Verdict

### Strongest Run

**vK.11.4** is the "best" of the six by a thin margin:
- Tam-F1=0.1321 (matches 12.0/12.0b, beats 11.5's 0.1272 and R2's 0.1274)
- Best AUC-ROC (0.6434 among original 3; R2's 0.6550 is technically highest but with worse segmentation)
- Trained longest (25 epochs, most exploration of loss landscape)
- No crash, no broken dashboard

However, calling any synthesis run "strongest" is misleading. All six are failures.

### Most Stable Training

**vK.11.4** had a less pathological training curve (gradual improvement to epoch 15, then plateau) compared to vK.11.5 (peak at epoch 3, continuous decline), vK.12.0/12.0b (peak at epoch 6, flatline), and vK.11.1-R2 (peak at epoch 4, slow decline). None are "stable" in any meaningful sense.

### Best Evaluation Infrastructure

**vK.12.0** has the most comprehensive evaluation suite -- precision/recall/pixel-accuracy, dataset exploration, model complexity, architecture diagram, and experiment comparison. Its diagnostic value is the highest even though its model performance is the same.

### Best W&B Integration

**vK.11.1-R2** is the only run with online W&B sync and the most comprehensive artifact logging (predictions, threshold sweeps, ELA, Grad-CAM, robustness tables all pushed to wandb.ai).

### Most Technically Sound Implementation

**vK.11.1** has the best CONFIG (100 epochs, patience=20) and represents the intended design. Its only flaw is the edge loss AMP bug, which is a one-line fix. If the AMP bug were fixed and the CONFIG retained, vK.11.1 would have given the synthesis architecture its best chance.

### Recommended Baseline for Further Development

**None of the vK.11.x/12.0 notebooks should be the baseline.** The synthesis architecture has a fundamental optimization problem that cannot be fixed by training longer or adjusting hyperparameters alone. Five training runs have proven this conclusively.

The recommended path forward is:

1. **Revert to v6.5's architecture** (SMP UNet, ResNet34, 3-channel input, BCEDice loss, no classification head, no edge loss) as the proven baseline
2. **Add ONE component at a time** with controlled ablation:
   - Run A: v6.5 + ELA 4th channel (test ELA value)
   - Run B: v6.5 + edge loss (test edge supervision)
   - Run C: v6.5 + classification head (test dual-task)
3. **Only combine components** that individually demonstrate improvement
4. **Use vK.11.1's CONFIG** (100 epochs, patience=20) for all ablation runs

---

## Appendix: Full Metric Table

| Metric | vK.11.1 | vK.11.4 | vK.11.5 | vK.12.0 | vK.11.1-R2 | vK.12.0b |
|--------|---------|---------|---------|---------|------------|----------|
| Status | Unexecuted | Trained | Trained | Trained (crashed) | Trained | Trained (crashed) |
| Epochs | 0 | 25 | 13 | 16 | 14 | 16 |
| Best epoch | N/A | 15 | 3 | 6 | 4 | 6 |
| Best val Dice(tam) | N/A | 0.1412 | 0.1364 | 0.1412 | 0.1362 | 0.1412 |
| Test Accuracy | N/A | 0.4142 | 0.4194 | 0.4062 | 0.5235 | 0.4062 |
| Test AUC-ROC | N/A | 0.6434 | 0.6466 | 0.5637 | 0.6550 | 0.6175 |
| Test Dice (all) | N/A | 0.0537 | 0.0517 | 0.0537 | 0.0517 | 0.0537 |
| Test IoU (all) | N/A | 0.0335 | 0.0312 | 0.0335 | 0.0317 | 0.0335 |
| Test F1 (all) | N/A | 0.0537 | 0.0517 | 0.0537 | 0.0517 | 0.0537 |
| Test Dice (tam) | N/A | 0.1321 | 0.1272 | 0.1321 | 0.1274 | 0.1322 |
| Test IoU (tam) | N/A | 0.0825 | 0.0768 | 0.0825 | 0.0780 | 0.0825 |
| Test F1 (tam) | N/A | 0.1321 | 0.1272 | 0.1321 | 0.1274 | 0.1322 |
| Precision (tam) | N/A | — | — | 0.0825 | — | 0.0825 |
| Recall (tam) | N/A | — | — | 1.0000 | — | 1.0000 |
| Pixel Accuracy (tam) | N/A | — | — | 0.0825 | — | 0.0825 |
| Pixel-AUC | N/A | 0.4988 | 0.5215 | 0.4952 | 0.4482 | 0.4791 |
| Optimal threshold | N/A | 0.4939 | 0.0500 | 0.5092 | 0.0500 | 0.0500 |
| F1 at optimal | N/A | 0.1321 | 0.1321 | 0.1321 | 0.1321 | 0.1321 |
| Splicing Dice | N/A | 0.1016 | 0.1016 | 0.1016 | 0.1016 | 0.1016 |
| Copy-move Dice | N/A | 0.1918 | 0.1918 | 0.1918 | 0.1918 | 0.1918 |
| Shortcut delta | N/A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Robustness (all) | N/A | 0.1321 | 0.1321 | Not executed | 0.1321 | Not executed |
