# Technical Audit: vR.P.3 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-3-ela-as-input-replace-rgb-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 25 epochs (all ran, no early stopping), best at epoch 25 |
| **Version** | vR.P.3 — ELA as Input (Replace RGB) |
| **Parent** | vR.P.2 (gradual encoder unfreeze, RGB input, Pixel F1=0.5117) |
| **Change** | Replace RGB input with ELA (JPEG Q=90) + ELA-specific normalization + revert to frozen body + BN unfreeze |
| **Status** | **PARTIALLY EXECUTED — Cell 22 NameError crashed cells 22-27, MODEL NOT SAVED** |

---

## 1. Notebook Overview & Experiment Goal

vR.P.3 tests a fundamental hypothesis: **does ELA provide a better forensic signal than RGB for pixel-level tampering localization?** Instead of feeding raw RGB pixels to the pretrained ResNet-34 encoder, this version computes ELA (Error Level Analysis) maps from JPEG recompression at Q=90 and uses these 3-channel ELA images as input.

This requires replacing ImageNet normalization with **ELA-specific normalization** (mean/std computed from 500 training samples), since ELA pixel distributions are starkly different from natural images. The encoder freeze strategy reverts from P.2's aggressive unfreeze (layer3+layer4) to P.3's conservative **frozen body + BN unfreeze** — only BatchNorm layers adapt to the new ELA distribution while all convolutional weights remain frozen at ImageNet values.

**Result:** The breakthrough run of the pretrained series. Pixel F1 jumps from 0.5117 (P.2) to **0.6920** — a **+17.8pp improvement**, the largest single-variable gain in either track. Image accuracy reaches **86.79%**, the best in the pretrained series.

**Critical issue:** A `NameError: name 'denormalize' is not defined` in the visualization cell (cell 22) crashed the remaining cells. The model was **never saved** — prediction visualizations, per-image statistics, and the `.pth` file were never generated.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | ResNet-34 (ImageNet, **FROZEN body + BN UNFROZEN**) |
| Input | **ELA 384x384** (JPEG Q=90, brightness-scaled) |
| Normalization | **ELA-specific**: mean=[0.0497, 0.0418, 0.0590], std=[0.0663, 0.0570, 0.0756] |
| Loss | SoftBCEWithLogitsLoss + DiceLoss (mode=binary, from_logits=True) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5, single param group) |
| Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | patience=7, val_loss |
| Seed | 42 |
| AMP | Enabled (autocast + GradScaler) |
| TF32 | Enabled (no effect on P100 Pascal GPU) |
| num_workers | 2, persistent_workers=True |
| drop_last | True (train only) |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### Freeze Strategy

| Component | Status | Rationale |
|-----------|--------|-----------|
| Encoder conv weights (all layers) | **FROZEN** | Preserve ImageNet features |
| Encoder BatchNorm layers | **UNFROZEN** | Adapt running statistics to ELA distribution |
| Decoder (all) | **TRAINABLE** | Learn to segment from ELA features |
| Segmentation head | **TRAINABLE** | Final 1x1 conv output |

### Parameters

| Category | Count |
|----------|-------|
| Total | 24,436,369 |
| Trainable | 3,168,721 (13.0%) |
| — Encoder BN | 17,024 |
| — Decoder | 3,151,552 |
| — Segmentation head | 145 |
| Frozen (encoder conv) | 21,267,648 |
| Data:param ratio | 1:359 |

### ELA Normalization (computed from 500 training samples)

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.0497 | 0.0663 |
| G | 0.0418 | 0.0570 |
| B | 0.0590 | 0.0756 |

Note: These values are ~10x smaller than ImageNet normalization, reflecting that ELA images are mostly dark (low error) with bright spots at tampering boundaries.

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **Largest single-variable improvement in either track: +17.8pp Pixel F1** | 0.6920 vs P.2's 0.5117 — dwarfs all previous gains (next best was P.2's +5.7pp over P.1) |
| S2 | **Best Pixel IoU in pretrained series: 0.5291** | +0.1852 from P.2 (0.3439) — nearly doubles |
| S3 | **Best Pixel AUC: 0.9528** | +0.0840 from P.2 — first model above 0.95 |
| S4 | **Best image accuracy in pretrained series: 86.79%** | +17.75pp from P.2 (69.04%), +16.64pp from P.1 (70.15%) |
| S5 | **Lowest FP rate in pretrained series: 2.7%** | Only 30/1124 authentic images falsely flagged (P.1: 22.6%, P.2: 19.7%) |
| S6 | **Best image ROC-AUC: 0.9502** | +0.2306 from P.2 (0.7196) — dramatic improvement in classification calibration |
| S7 | **Conservative freeze works better than aggressive unfreeze** | P.3 (frozen body + BN only) outperforms P.2 (layer3+4 unfrozen) across all metrics |
| S8 | **Model still improving at epoch 25** | Val metrics trending upward — more epochs would likely yield even better results |
| S9 | **Steady convergence** | Monotonic improvement with expected LR reduction plateaus, no catastrophic instability |
| S10 | **ELA-specific normalization correctly computed** | Mean ~0.05, std ~0.06 — appropriate for the sparse, dark ELA distribution |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **CRITICAL** | `NameError: name 'denormalize' is not defined` crashed cells 22-27 — **MODEL NOT SAVED** |
| W2 | **CRITICAL** | No prediction visualizations generated — cannot inspect what the model actually predicts |
| W3 | **MAJOR** | Best epoch is 25 (the last epoch) — model was still improving, 25 epochs is likely insufficient |
| W4 | **MAJOR** | Pixel recall only 0.5905 — still misses ~41% of tampered pixels |
| W5 | **MAJOR** | Image-level FN rate 28.6% — nearly 1 in 3 tampered images go undetected |
| W6 | MODERATE | Train-val gap growing: train_loss=0.2381 vs val_loss=0.4109 at epoch 25, ratio 1.73x |
| W7 | MODERATE | Val loss oscillates significantly (e.g., 0.4976 → 0.5054 → 0.4705 → 0.5127 in epochs 9-12) |
| W8 | MINOR | Function naming inconsistency: `denormalize_ela()` defined but `denormalize()` called — copy-paste from RGB notebook |

---

## 5. Major Issues

### 5.1 CRITICAL: Model Not Saved (W1, W2)

The notebook defines a function `denormalize_ela()` in the preprocessing section but the visualization cell calls `denormalize()` without the `_ela` suffix. This is a copy-paste error from the RGB-based parent notebook (vR.P.2). The NameError crashes cell 22, and since cells 23-27 depend on prior execution, they also fail. Cell 27 contains the model save code (`torch.save()`), so **the trained model weights are permanently lost**.

This is the single most damaging bug in the pretrained series: the best-performing model cannot be loaded, deployed, or submitted for the assignment.

### 5.2 MAJOR: Insufficient Training Duration (W3)

The model achieved its best val_loss (0.4109) at epoch 25 — the very last epoch. The training curve shows continued improvement:
- Epochs 22-25 val_loss: 0.4137 → 0.4249 → 0.4188 → 0.4109 (trending down)
- Epochs 22-25 pixel F1: 0.7243 → 0.7100 → 0.7163 → 0.7243 (trending up)

With `max_epochs=25` and `patience=7`, early stopping never triggered because the model kept finding new bests. The val pixel F1 of 0.7243 at epoch 25 is significantly better than the 0.6920 test F1, suggesting more training could push test performance higher.

### 5.3 MAJOR: Pixel Recall Gap (W4)

Despite the dramatic improvement, pixel recall (0.5905) remains the weakest metric. The model is precise (0.8356) but conservative — it identifies tampered regions accurately where it does predict, but misses 41% of tampered pixels entirely. This suggests the ELA signal highlights obvious tampering boundaries but the model struggles with subtle forgeries.

---

## 6. Minor Issues

### 6.1 Val Loss Oscillation (W7)

Validation loss shows characteristic oscillation with amplitude ~0.1 (e.g., epoch 9: 0.4976, epoch 12: 0.5127, epoch 14: 0.4919). This is typical of small validation sets (1,892 images) and batch-level variance, but it makes the LR scheduler unreliable — it may reduce LR based on noise rather than genuine convergence.

### 6.2 Growing Overfitting Gap (W6)

By epoch 25: train_loss=0.238, val_loss=0.411, ratio=1.73x. While much better than P.2's severe overfitting (ratio 2.9x at epoch 14), the gap is growing. The frozen encoder (only BN unfrozen) provides good regularization, but 3.17M trainable parameters on 8,829 images still allows memorization.

---

## 7. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR |
|-------|-----------|----------|----------|-----|-----|
| 1 | 0.9169 | 0.7792 | 0.4051 | 0.2540 | 1e-3 |
| 3 | 0.6738 | 0.6397 | 0.5460 | 0.3755 | 1e-3 |
| 5 | 0.5424 | 0.5592 | 0.5941 | 0.4225 | 1e-3 |
| 9 | 0.4466 | 0.4976 | 0.6523 | 0.4840 | 1e-3 |
| 11 | 0.4105 | 0.4705 | 0.6829 | 0.5185 | 1e-3 |
| 16 | 0.3174 | 0.4687 | 0.6709 | 0.5048 | 5e-4 |
| 17 | 0.3140 | 0.4271 | 0.7108 | 0.5514 | 5e-4 |
| 22 | 0.2570 | 0.4137 | 0.7243 | 0.5677 | 2.5e-4 |
| **25** (best) | **0.2381** | **0.4109** | **0.7243** | **0.5678** | **2.5e-4** |

**LR schedule:** 1e-3 (epochs 1-15) → 5e-4 (epochs 16-21) → 2.5e-4 (epochs 22-25)

**Key observations:**
- Strongest gains in epochs 1-11 (Pixel F1: 0.40 → 0.68)
- LR reduction at epoch 16 produced the best single-epoch gain: F1 jumped from 0.6709 to 0.7108
- Second LR reduction at epoch 22 yielded another F1 jump: 0.7064 → 0.7243
- Model was still improving at epoch 25 — val_loss hit its best and pixel F1 tied its best

---

## 8. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.2 | Delta from P.1 |
|--------|-------|----------------|----------------|
| Pixel Precision | 0.8356 | +0.2073 | +0.2021 |
| **Pixel Recall** | **0.5905** | **+0.1588** | **+0.2360** |
| **Pixel F1** | **0.6920** | **+0.1803** | **+0.2374** |
| **Pixel IoU** | **0.5291** | **+0.1852** | **+0.2349** |
| **Pixel AUC** | **0.9528** | **+0.0840** | **+0.1019** |

### Image-Level (Classification)

| Metric | Value | Delta from P.2 | Delta from P.1 |
|--------|-------|----------------|----------------|
| **Test Accuracy** | **86.79%** | **+17.75pp** | **+16.64pp** |
| **Macro F1** | **0.8560** | **+0.1887** | **+0.1693** |
| **ROC-AUC** | **0.9502** | **+0.2306** | **+0.1717** |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.8326 | 0.9733 | 0.8975 | 1,124 |
| Tampered | 0.9482 | 0.7139 | 0.8145 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 1,094 (TN) | 30 (FP) |
| **Tp** | 220 (FN) | 549 (TP) |

- **FP rate: 2.7%** (30/1124) — best in series, 8.7x lower than P.1's 22.6%
- **FN rate: 28.6%** (220/769) — improved from P.2's 47.5%
- **Tampered precision: 0.9482** — when the model flags an image as tampered, it's right 94.8% of the time

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"The authors present an exciting result — a +18pp pixel F1 improvement — and then immediately drive it off a cliff with a `NameError` that prevents saving the model. One must admire the commitment to snatching defeat from the jaws of victory.

The hypothesis is sound: ELA compresses the forensic signal into a more learnable representation than raw RGB. The frozen-body-plus-BN-unfreeze strategy is elegant and the results validate it convincingly. But the execution is sloppy. The `denormalize_ela()` vs `denormalize()` naming mismatch is a trivial copy-paste error that would be caught by running the notebook once before submission. The fact that the best model in the series has no saved weights is, to put it diplomatically, suboptimal.

Furthermore, the 25-epoch ceiling is concerning. The model was clearly still improving — the best epoch IS the last epoch. This is not early stopping convergence; this is the researcher running out of patience before the model does. A simple `max_epochs=50` would have cost negligible additional compute and likely pushed Pixel F1 above 0.70 on test.

The results themselves are genuinely impressive: 86.79% image accuracy (up from 69% in the parent), 2.7% FP rate (down from 20%), and near-0.95 pixel AUC. The ELA-as-input insight is the single most valuable finding in this ablation series. It just deserves a notebook that can actually save its own model."

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction (pixel masks) | **PARTIAL** | Model generates masks but crash prevents saving/loading |
| Train/val/test split | **PASS** | 70/15/15 stratified, seed=42 |
| Standard metrics (F1, IoU, AUC) | **PASS** | Full pixel-level + image-level suite computed |
| Visual results (Original/GT/Predicted/Overlay) | **FAIL** | Cell 22 crash — no visualizations generated |
| Model weights (.pth file) | **FAIL** | Cell 27 never executed — model not saved |
| Architecture explanation | **PASS** | Detailed pipeline diagram in markdown cells |
| Single notebook execution | **FAIL** | Cells 22-27 crash — notebook does not run end-to-end |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 13 | 15 | Excellent: ELA input innovation, conservative freeze strategy, proper BN adaptation |
| Dataset | 14 | 15 | Proper GT masks, ELA normalization computed correctly, stratified split |
| Methodology | 14 | 20 | Sound single-variable ablation, but model crash + insufficient epochs penalized heavily |
| Evaluation | 16 | 20 | Comprehensive metrics computed, but no visualizations due to crash |
| Documentation | 11 | 15 | Good inline markdown and pipeline diagram, but crash prevents full results documentation |
| Assignment Alignment | 10 | 15 | Core metrics present, but model not saved = cannot submit for assignment |
| **Total** | **78** | **100** | |

---

## 12. Final Verdict: **STRONG POSITIVE** — Score: 78/100

**Pixel F1: 0.6920 (+0.1803 from parent P.2, +0.2374 from baseline P.1)**

vR.P.3 is the most important experiment in the pretrained series. The switch from RGB to ELA input produces a +17.8pp improvement in Pixel F1 — 3x larger than all previous gains combined. Every metric improves dramatically: pixel AUC crosses 0.95, image accuracy reaches 86.79%, and the FP rate drops to 2.7%.

The result validates a key insight: **input representation matters more than encoder architecture or training strategy.** P.2's aggressive encoder unfreeze with 23M trainable parameters on RGB input reached 0.5117. P.3's conservative freeze with 3.17M trainable parameters on ELA input reaches 0.6920. The forensic signal in ELA images is simply a better match for the localization task.

**Score penalized from potential ~88 to 78** due to:
- **CRITICAL bug:** Model not saved (cost: -6 points across Methodology, Evaluation, Assignment Alignment)
- **Insufficient training:** Best epoch = last epoch, model still improving (cost: -4 points in Methodology)

### Recommended Next Steps

1. **Re-run P.3 with bug fix** — rename `denormalize()` to `denormalize_ela()` and increase `max_epochs` to 50
2. **Use P.3's ELA input as the new default** for all future pretrained experiments
3. **Investigate pixel recall gap** — 0.5905 recall means 41% of tampered pixels are missed; attention mechanisms or higher resolution may help
