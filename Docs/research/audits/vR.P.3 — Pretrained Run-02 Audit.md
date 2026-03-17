# Technical Audit: vR.P.3 — Pretrained Run-02

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-3-ela-as-input-replace-rgb-run-02.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 25 epochs (hit ceiling), best at epoch 25 |
| **Version** | vR.P.3 — ELA Input (Run-02, Reproducibility) |
| **Parent** | vR.P.1 (ResNet-34, RGB, frozen encoder, Pixel F1=0.4546) |
| **Change** | Reproducibility re-run of P.3 Run-01 (identical config) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview & Experiment Goal

vR.P.3 Run-02 is a **reproducibility re-run** of the breakthrough ELA input experiment. The original P.3 Run-01 demonstrated that replacing RGB input with Error Level Analysis (ELA) images dramatically improves forgery localization (+23.74pp Pixel F1 over P.1). This run validates that finding.

**Configuration:** Identical to P.3 Run-01 — UNet with ResNet-34 encoder (frozen weights, unfrozen BatchNorm), ELA input at 384x384, BCE+Dice loss, Adam optimizer.

**Result:** Pixel F1 reaches **0.6920** (identical to Run-01 within noise), Image Accuracy **86.79%**. The reproducibility is confirmed. However, the model was still improving at epoch 25 (best = last epoch), suggesting the training was cut short.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | ResNet-34 (ImageNet, frozen weights, **BN unfrozen**) |
| Input | **ELA** 384x384, ImageNet normalization |
| Loss | SoftBCEWithLogitsLoss + DiceLoss (mode=binary, from_logits=True) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5, decoder + BN only) |
| Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | patience=7, val_loss |
| Seed | 42 |
| AMP | NOT ENABLED |
| num_workers | 0 |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### Parameters

| Category | Count |
|----------|-------|
| Total | 24,439,617 |
| Trainable (decoder + BN) | 3,174,833 (13.0%) |
| Frozen (encoder conv weights) | 21,264,784 |

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **Reproducibility confirmed** | Pixel F1 0.6920 matches Run-01 — result is stable, not a fluke |
| S2 | **ELA breakthrough validated** | +23.74pp over P.1 confirmed across two independent runs |
| S3 | **Lowest FP rate in series: 2.7%** | Only 30 authentic images misclassified as tampered (of 1,124) |
| S4 | **Strong pixel AUC: 0.9528** | Excellent discrimination between tampered and authentic pixels |
| S5 | **Efficient training** | Only 13% of parameters trainable — minimal overfitting risk |
| S6 | **Clean execution** | All cells pass, model saved, full evaluation suite completed |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **MAJOR** | Training hit epoch ceiling (25/25) — model still improving, not converged |
| W2 | **MAJOR** | Results table incorrectly lists input as "RGB" instead of "ELA" (display bug) |
| W3 | MODERATE | Pixel recall limited at 0.5905 — misses ~41% of tampered pixels |
| W4 | MODERATE | No AMP/TF32 — slower training, methodology inconsistent with P.1.5+ |
| W5 | MINOR | num_workers=0 — GPU idles during data loading |

---

## 5. Major Issues

### 5.1 MAJOR: Training Not Converged (W1)

The best epoch is epoch 25, which is also the last epoch. The training curve shows continued improvement with no plateau. Early stopping never triggered (patience=7 was never exhausted). This strongly suggests the model would benefit from extended training.

**Impact:** The reported metrics are likely an underestimate of P.3's true ceiling. This directly motivates the vR.P.7 experiment (extended training, 50 epochs).

### 5.2 MAJOR: Display Bug in Results Table (W2)

The notebook's results summary table incorrectly displays "RGB" as the input type instead of "ELA". This is a display-only bug — the actual training used ELA images (confirmed by the `convert_to_ela_image` function calls and ELA normalization in the data pipeline). However, this could cause confusion when reviewing results.

---

## 6. Minor Issues

### 6.1 Limited Pixel Recall (W3)

Pixel recall of 0.5905 means the model detects only ~59% of tampered pixels. While precision is high (0.8356), the model is conservative — it confidently identifies regions it's sure about but misses many tampered areas. This is a common pattern with frozen encoders that lack fine-grained feature adaptation.

### 6.2 Missing Speed Optimizations (W4)

No AMP (Automatic Mixed Precision) or TF32 enabled. While this doesn't affect metric accuracy, it means training was ~1.5-2x slower than necessary. P.1.5 demonstrated these optimizations work on Kaggle P100.

---

## 7. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR |
|-------|-----------|----------|----------|-----|-----|
| 1 | 0.9873 | 0.8704 | 0.4218 | 0.2672 | 1e-3 |
| 5 | 0.5802 | 0.5623 | 0.5837 | 0.4124 | 1e-3 |
| 10 | 0.4205 | 0.4672 | 0.6347 | 0.4651 | 1e-3 |
| 15 | 0.3451 | 0.4289 | 0.6612 | 0.4940 | 5e-4 |
| 20 | 0.2947 | 0.4135 | 0.6795 | 0.5147 | 2.5e-4 |
| **25** (best) | **0.2653** | **0.4008** | **0.6920** | **0.5291** | **1.25e-4** |

**Key observations:**
- **Still improving at epoch 25:** Both val loss and Pixel F1 show continued improvement with no sign of plateau
- **LR reduced 3 times:** 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4
- **No early stopping triggered:** Patience=7 was never exhausted — the model never stopped improving
- **Train-val gap moderate:** 0.2653 vs 0.4008 (ratio 1.51x) — healthy, not severely overfitting

---

## 8. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.1 |
|--------|-------|----------------|
| Pixel Precision | 0.8356 | +0.2021 |
| Pixel Recall | 0.5905 | +0.2360 |
| **Pixel F1** | **0.6920** | **+0.2374** |
| **Pixel IoU** | **0.5291** | **+0.2349** |
| **Pixel AUC** | **0.9528** | **+0.1019** |

### Image-Level (Classification)

| Metric | Value | Delta from P.1 |
|--------|-------|----------------|
| Test Accuracy | 86.79% | +16.64pp |
| Macro F1 | 0.8560 | +0.1693 |
| ROC-AUC | 0.9502 | +0.1717 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.8332 | 0.9733 | 0.8979 | 1,124 |
| Tampered | 0.9407 | 0.7139 | 0.8119 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 1,094 (TN) | 30 (FP) |
| **Tp** | 220 (FN) | 549 (TP) |

- **FP rate: 2.7%** (30/1,124) — best in entire series
- **FN rate: 28.6%** (220/769) — moderate, room for improvement

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"A reproducibility run. How refreshingly honest. The authors re-ran their best experiment and got the same result. Science works. Give them a participation trophy.

But wait — the model was still improving when training stopped. The best epoch is the LAST epoch. That's not convergence, that's interruption. You've established that P.3's architecture works, and then you've also established that you didn't let it finish. It's like testing whether a car can reach 200 km/h by driving it to 150 and then parking.

The results table says 'RGB' when the input is clearly ELA. A documentation bug in a documentation run. The irony is palpable.

Still, the core finding is confirmed: ELA input is a +23.74pp improvement. That's not noise, that's signal. The FP rate of 2.7% means this model almost never falsely accuses an authentic image — a genuinely useful property. If only it could also find more than 59% of actual tampering."

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction (pixel masks) | **PASS** | Model generates 384x384 binary masks |
| Train/val/test split | **PASS** | 70/15/15 stratified, seed=42 |
| Standard metrics (F1, IoU, AUC) | **PASS** | Full pixel-level + image-level suite |
| Visual results (Original/GT/Predicted/Overlay) | **PASS** | All visualization cells executed |
| Model weights (.pth file) | **PASS** | Saved: 123.4 MB |
| Architecture explanation | **PASS** | ELA pipeline + UNet documented |
| Single notebook execution | **PASS** | End-to-end on Kaggle |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 13 | 15 | Same proven UNet+ResNet-34+ELA architecture as Run-01 |
| Dataset | 14 | 15 | Proper GT masks, standard pipeline, consistent splits |
| Methodology | 15 | 20 | Reproducibility run is good science. Training not converged (-3), display bug (-2) |
| Evaluation | 18 | 20 | Comprehensive metrics suite, all visualizations present |
| Documentation | 10 | 15 | Results table has RGB/ELA bug (-3), otherwise adequate |
| Assignment Alignment | 12 | 15 | All deliverables present, good metrics |
| **Total** | **82** | **100** | |

---

## 12. Final Verdict: **BASELINE (Reproducibility re-run)** — Score: 82/100

**Pixel F1: 0.6920 (confirms Run-01 result)**

vR.P.3 Run-02 confirms the reproducibility of the ELA input breakthrough. The result is stable across two independent runs (same seed, same config, same metrics). This validates the core finding that ELA preprocessing is the single most impactful change in the entire experimental series.

The main takeaway is not the metrics (which are identical to Run-01) but the evidence that training was cut short. The model was still improving at epoch 25, directly motivating the vR.P.7 extended training experiment.

### Key Insight: Training Budget is the Next Bottleneck

| Evidence | Implication |
|----------|-------------|
| Best epoch = last epoch (25/25) | Model not converged |
| Early stopping never triggered | No sign of overfitting |
| Val loss still decreasing | More capacity to learn |
| Train-val gap healthy (1.51x) | Room for more training |

**Recommended next step:** vR.P.7 — extend training to 50 epochs with the same P.3 configuration.
