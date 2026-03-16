# Technical Audit: vR.P.9 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-9-focal-dice-loss-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 25 epochs, best at epoch 21 |
| **Version** | vR.P.9 — Focal + Dice Loss |
| **Parent** | vR.P.3 (ELA, frozen+BN, BCE+Dice, Pixel F1=0.6920) |
| **Change** | Replace BCE loss with Focal Loss (alpha=0.25, gamma=2.0) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview & Experiment Goal

vR.P.9 tests whether **Focal Loss** improves localization by addressing the class imbalance at the pixel level. In image tampering detection, the vast majority of pixels are authentic (typically >80% of each image). Standard BCE treats all pixels equally, while Focal Loss down-weights easy-to-classify pixels and focuses learning on hard boundary pixels.

**Loss function:**
- **P.3 (parent):** SoftBCEWithLogitsLoss + DiceLoss
- **P.9:** FocalLoss(alpha=0.25, gamma=2.0, from_logits=True) + DiceLoss(mode=binary, from_logits=True)

Everything else is identical to P.3: UNet + ResNet-34 (frozen+BN), ELA input, Adam optimizer.

**Result:** Pixel F1 reaches **0.6923** (+0.03pp from P.3 — essentially unchanged). However, Pixel AUC **regressed** from 0.9528 to 0.9323 (-0.0205), and Image ROC-AUC dropped from 0.9502 to 0.9076 (-0.0426). The Focal Loss hypothesis is **not confirmed**.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | ResNet-34 (ImageNet, frozen weights, BN unfrozen) |
| Input | ELA 384x384, ImageNet normalization |
| Loss | **FocalLoss(alpha=0.25, gamma=2.0) + DiceLoss** (from_logits=True) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5, decoder + BN only) |
| Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | patience=7, val_loss |
| Seed | 42 |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### Focal Loss Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| alpha | 0.25 | Balances positive/negative class weights |
| gamma | 2.0 | Down-weights easy examples by factor (1-p)^gamma |
| from_logits | True | Operates on raw model outputs, numerically stable |

### Parameters

| Category | Count |
|----------|-------|
| Total | 24,439,617 |
| Trainable (decoder + BN) | 3,174,833 (13.0%) |
| Frozen (encoder conv weights) | 21,264,784 |
| Model file size | 123.4 MB |

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **Pixel recall slightly improved: 0.5922** | +0.0017 over P.3's 0.5905 |
| S2 | **Image accuracy improved: 87.16%** | +0.37pp over P.3's 86.79% |
| S3 | **Clean single-variable ablation** | Only the loss function changed |
| S4 | **Proper early stopping** | Best at epoch 21, patience exhausted at epoch 25 |
| S5 | **Same model size as P.3** | No additional parameters (123.4 MB) |
| S6 | **Clean execution** | All cells pass, model saved, full evaluation suite |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **MAJOR** | Pixel AUC regressed: 0.9323 vs P.3's 0.9528 (-0.0205) |
| W2 | **MAJOR** | Image ROC-AUC regressed: 0.9076 vs P.3's 0.9502 (-0.0426) |
| W3 | **MAJOR** | Training volatility increased |
| W4 | MODERATE | Pixel F1 essentially unchanged: +0.03pp |
| W5 | MODERATE | FP rate increased: 3.2% vs P.3's 2.7% |
| W6 | MINOR | Focal loss hyperparameters (alpha, gamma) not tuned — defaults used |

---

## 5. Major Issues

### 5.1 MAJOR: AUC Regression (W1, W2)

The most concerning result is the AUC degradation at both pixel and image levels. AUC measures the model's ability to rank predictions — a drop indicates the raw probability outputs are less well-calibrated.

**Pixel AUC:** 0.9528 → 0.9323 (-0.0205, -2.15%)
**Image ROC-AUC:** 0.9502 → 0.9076 (-0.0426, -4.48%)

**Interpretation:** Focal Loss's (1-p)^gamma weighting artificially concentrates predictions near the extremes (0 and 1), reducing the dynamic range of intermediate probabilities.

### 5.2 MAJOR: Training Volatility (W3)

The training curves for P.9 show more oscillation in validation loss compared to P.3's smooth descent. Focal Loss introduces additional gradient modulation that makes optimization less stable.

---

## 6. Minor Issues

### 6.1 Untuned Hyperparameters (W6)

The Focal Loss alpha=0.25 and gamma=2.0 are standard defaults from the RetinaNet paper. These were designed for object detection, not forensic segmentation. A grid search would provide stronger evidence.

### 6.2 FP Rate Increase (W5)

FP rate increased from 2.7% to 3.2% — marginal but in the wrong direction.

---

## 7. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR |
|-------|-----------|----------|----------|-----|-----|
| 1 | 0.7124 | 0.6481 | 0.4186 | 0.2648 | 1e-3 |
| 5 | 0.3952 | 0.4217 | 0.5798 | 0.4081 | 1e-3 |
| 10 | 0.2918 | 0.3614 | 0.6302 | 0.4601 | 1e-3 |
| 15 | 0.2341 | 0.3378 | 0.6654 | 0.4984 | 5e-4 |
| 20 | 0.2012 | 0.3289 | 0.6887 | 0.5252 | 2.5e-4 |
| **21** (best) | **0.1965** | **0.3264** | **0.6923** | **0.5294** | **2.5e-4** |
| 25 (final) | 0.1823 | 0.3412 | 0.6841 | 0.5198 | 1.25e-4 |

**Key observations:**
- **Best at epoch 21** — earlier than P.3's epoch 25
- **Loss scales not directly comparable** — Focal Loss produces smaller absolute values than BCE
- **Post-peak degradation** — epochs 22-25 show increasing val loss

---

## 8. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.3 |
|--------|-------|----------------|
| Pixel Precision | 0.8329 | -0.0027 |
| Pixel Recall | 0.5922 | +0.0017 |
| **Pixel F1** | **0.6923** | **+0.0003** |
| Pixel IoU | 0.5294 | +0.0003 |
| Pixel AUC | 0.9323 | **-0.0205** |

### Image-Level (Classification)

| Metric | Value | Delta from P.3 |
|--------|-------|----------------|
| Test Accuracy | 87.16% | +0.37pp |
| Macro F1 | 0.8606 | +0.0046 |
| ROC-AUC | 0.9076 | **-0.0426** |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.8413 | 0.9680 | 0.9003 | 1,124 |
| Tampered | 0.9369 | 0.7282 | 0.8194 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 1,088 (TN) | 36 (FP) |
| **Tp** | 207 (FN) | 562 (TP) |

- **FP rate: 3.2%** (36/1,124) — slightly worse than P.3's 2.7%
- **FN rate: 26.9%** (207/769) — improved over P.3's 28.6%

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"Focal Loss. The loss function that conquered object detection and then proceeded to disappoint everyone who thought it was a universal solution. The authors hypothesize that down-weighting easy pixels will help the model focus on hard boundary regions. Reasonable hypothesis. Clean experimental design.

The result: +0.03pp Pixel F1. That's not an improvement, that's thermal noise. The stock market fluctuates more than that during a coffee break.

But the real story is in the AUC metrics. Pixel AUC drops 2.15%. Image ROC-AUC craters by 4.48%. The model's probability outputs are less calibrated, less informative, less useful for anything beyond binary thresholding. Focal Loss improved recall by 0.0017 (literally 1-2 pixels per image) while destroying the probability distribution.

The fundamental problem: forensic segmentation is not the same as object detection. In object detection, you have rare objects against a sea of background. In image tampering, the tampering IS the signal — it's not rare within the tampered region, it's dense. Focal Loss's (1-p)^gamma weighting assumes the imbalance is the problem. But in this dataset, the imbalance is already handled by the Dice Loss component. Adding Focal Loss is treating a disease the patient doesn't have.

At least the experiment is clean. A well-executed negative result is still a result. File this under 'confirmed: BCE > Focal for forensic segmentation.'"

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction (pixel masks) | **PASS** | Model generates 384x384 binary masks |
| Train/val/test split | **PASS** | 70/15/15 stratified, seed=42 |
| Standard metrics (F1, IoU, AUC) | **PASS** | Full pixel-level + image-level suite |
| Visual results (Original/GT/Predicted/Overlay) | **PASS** | All visualization cells executed |
| Model weights (.pth file) | **PASS** | Saved: 123.4 MB |
| Architecture explanation | **PASS** | Focal Loss motivation and parameters documented |
| Single notebook execution | **PASS** | End-to-end on Kaggle |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 13 | 15 | Same UNet as P.3 — loss swap doesn't change architecture |
| Dataset | 14 | 15 | Same proven pipeline as P.3 |
| Methodology | 12 | 20 | Single-variable ablation is clean (+3), AUC regression not investigated (-4), defaults not tuned (-4) |
| Evaluation | 17 | 20 | Comprehensive metrics. AUC regression noted but not analyzed in depth (-3) |
| Documentation | 11 | 15 | Focal Loss motivation documented. Could better explain AUC regression. |
| Assignment Alignment | 11 | 15 | All deliverables present. AUC regression weakens practical utility. |
| **Total** | **78** | **100** | |

---

## 12. Final Verdict: **NEUTRAL** (+0.03pp from P.3) — Score: 78/100

**Pixel F1: 0.6923 (+0.0003 from P.3's 0.6920 — within NEUTRAL threshold)**

vR.P.9 demonstrates that Focal Loss does not improve forensic segmentation when combined with Dice Loss. The pixel-level binary predictions are essentially unchanged, while the probability calibration (AUC) significantly degrades.

### Key Insight: Focal Loss is Counterproductive for Forensic Segmentation

| Metric | P.3 (BCE+Dice) | P.9 (Focal+Dice) | Delta | Assessment |
|--------|----------------|-------------------|-------|------------|
| Pixel F1 | 0.6920 | 0.6923 | +0.03pp | Unchanged |
| Pixel AUC | **0.9528** | 0.9323 | **-0.0205** | Regressed |
| Image ROC-AUC | **0.9502** | 0.9076 | **-0.0426** | Regressed |
| Training stability | Smooth | Volatile | — | Worse |

**Recommended conclusion:** BCE+Dice remains the optimal loss for this architecture. Future loss experiments should explore Lovász-Softmax or Boundary Loss rather than weighted cross-entropy variants.
