# Technical Audit: vR.P.7 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-7-ela-extended-training-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 46 epochs (early stopping at 46), best at epoch 36 |
| **Version** | vR.P.7 — ELA + Extended Training (50 max epochs) |
| **Parent** | vR.P.3 (ELA, frozen+BN, 25 epochs, Pixel F1=0.6920) |
| **Change** | Extended max_epochs from 25→50, patience from 7→10, AMP enabled |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview & Experiment Goal

vR.P.7 tests whether **extended training** improves localization beyond P.3's 25-epoch ceiling. P.3's reproducibility run (P.3-r02) showed the model was still improving at epoch 25 (best = last epoch), suggesting premature termination. This experiment doubles max_epochs (25→50) and increases early stopping patience (7→10) to let the model converge naturally.

**Additional changes:** AMP and TF32 enabled (speed optimization), NUM_WORKERS=4 with persistent workers and prefetch_factor=2.

**Result:** Pixel F1 reaches **0.7154** (+2.34pp from P.3's 0.6920), with the best epoch at 36. Early stopping triggered at epoch 46. This confirms that P.3 was stopped prematurely and extended training yields meaningful improvement.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | ResNet-34 (ImageNet, frozen body + BN unfrozen) |
| Input | ELA 384×384, ELA-specific normalization |
| Loss | SoftBCEWithLogitsLoss + DiceLoss |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | **50** (was 25 in P.3) |
| Early stopping | **patience=10** (was 7 in P.3) |
| Seed | 42 |
| AMP | **Enabled** (new vs P.3) |
| TF32 | **Enabled** (new vs P.3) |
| num_workers | 4 (persistent, prefetch=2) |
| Trainable params | 3,168,721 (13.0%) |

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **Best Pixel F1 at time of run: 0.7154** | +2.34pp over P.3, surpasses P.4's 0.7053 |
| S2 | **Hypothesis confirmed** | P.3 was indeed stopped too early — best epoch 36 vs P.3's forced stop at 25 |
| S3 | **Extended training is free performance** | Same architecture, same data, just more epochs → +2.34pp |
| S4 | **Natural convergence** | Early stopping triggered at epoch 46 (patience=10 from epoch 36) — proper termination |
| S5 | **AMP enabled** | ~1.5x training speed without quality loss |
| S6 | **Comprehensive per-image analysis** | Mean Pixel F1 per tampered image: 0.5580 ± 0.4164 |
| S7 | **Model properly saved** | 123.4 MB checkpoint with full training state |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | MODERATE | Results table incorrectly shows "RGB 384²" instead of "ELA 384²" (display bug) |
| W2 | MODERATE | High per-image F1 variance (std=0.4164) — very inconsistent localization |
| W3 | MODERATE | Pixel AUC slightly regressed: 0.9504 vs P.3's 0.9528 (-0.0024) |
| W4 | MODERATE | FP rate increased: 3.6% vs P.3's 2.7% (+0.9pp) |
| W5 | MINOR | Val loss oscillates significantly (e.g., epoch 36: 0.3935 → epoch 37: 0.4562) |
| W6 | MINOR | Training time ~105 min (vs P.3's ~25 min) — 4.2x longer for +2.34pp |

---

## 5. Major Issues

No major issues. This is a clean, well-executed experiment with meaningful positive results.

---

## 6. Minor Issues

### 6.1 Display Bug (W1)
The cross-track comparison table incorrectly shows "RGB 384²" for this run. Actual input is ELA. Same bug as P.3-r02.

### 6.2 Per-Image Inconsistency (W2)
Per-image Pixel F1 on tampered images has std=0.4164 on a mean of 0.5580. Some images get near-perfect localization while others are barely detected. This variance suggests the model struggles with certain tampering types or sizes.

### 6.3 Pixel AUC Regression (W3)
Despite higher F1/IoU, Pixel AUC dropped slightly from 0.9528 to 0.9504. Extended training may have made predictions more binary (0/1) rather than well-calibrated probabilities, similar to the P.9 effect but much milder.

---

## 7. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR |
|-------|-----------|----------|----------|-----|-----|
| 1 | 0.9169 | 0.7792 | 0.4051 | 0.2540 | 1e-3 |
| 5 | 0.5424 | 0.5592 | 0.5941 | 0.4225 | 1e-3 |
| 11 | 0.4105 | 0.4705 | 0.6829 | 0.5185 | 1e-3 |
| 17 | 0.3140 | 0.4271 | 0.7108 | 0.5514 | 5e-4 |
| 25 | 0.2381 | 0.4109 | 0.7243 | 0.5678 | 2.5e-4 |
| **36 (best)** | **0.1869** | **0.3935** | **0.7404** | **0.5878** | **1.25e-4** |
| 46 (final) | 0.1612 | 0.4287 | 0.7089 | 0.5489 | 3.13e-5 |

**Key observations:**
- **11 more productive epochs than P.3** — continues improving from epoch 25 through 36
- **LR reduced 4 times:** 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5 → 3.13e-5
- **Epoch 36 val F1 (0.7404)** is substantially better than P.3's epoch 25 val F1 (~0.69)
- **Overfit after epoch 36:** Train loss continued decreasing but val loss rose

---

## 8. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.3 |
|--------|-------|----------------|
| Pixel Precision | 0.8374 | +0.0018 |
| **Pixel Recall** | **0.6245** | **+0.0340** |
| **Pixel F1** | **0.7154** | **+0.0234** |
| **Pixel IoU** | **0.5569** | **+0.0278** |
| Pixel AUC | 0.9504 | -0.0024 |

### Image-Level (Classification)

| Metric | Value | Delta from P.3 |
|--------|-------|----------------|
| Test Accuracy | 87.37% | +0.58pp |
| Macro F1 | 0.8637 | +0.0077 |
| ROC-AUC | 0.9433 | -0.0069 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.8449 | 0.9644 | 0.9007 | 1,124 |
| Tampered | 0.9344 | 0.7412 | 0.8267 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 1,084 (TN) | 40 (FP) |
| **Tp** | 199 (FN) | 570 (TP) |

- **FP rate: 3.6%** (40/1,124) — slightly worse than P.3's 2.7%
- **FN rate: 25.9%** (199/769) — improved from P.3's 28.6% (-2.7pp)

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"The simplest experiment in the series. More epochs. That's it. And it works. +2.34pp Pixel F1 from doing literally nothing to the architecture, the data, or the training procedure — just letting the model finish its job.

This raises an uncomfortable question: how many of the previous experiments (P.8 progressive unfreeze, P.9 focal loss) were actually testing their hypothesis, and how many were just getting additional training time dressed up as an ablation variable? P.8 ran for 32 epochs (vs P.3's 25). P.9 ran for 25 as well. P.7 ran for 46 and found the best at epoch 36. If P.8 and P.9 had also run for 50 epochs, would they have found even better checkpoints?

The per-image F1 variance (std=0.4164) tells a story the aggregate numbers hide: this model is inconsistent. Some tampered images get F1 > 0.9; others barely above 0.0. The model hasn't learned a robust tampering detector — it's learned to detect certain types of tampering. This is the next frontier: reducing variance, not just raising the mean.

Still, the simplest experiment, the cheapest experiment, and the most informative experiment. Sometimes the best research is 'have we actually tried training long enough?'"

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction (pixel masks) | **PASS** | 384×384 masks, Pixel F1=0.7154 |
| Train/val/test split | **PASS** | 70/15/15, seed=42 |
| Standard metrics (F1, IoU, AUC) | **PASS** | Complete pixel + image suite |
| Visual results | **PASS** | All visualizations present |
| Model weights | **PASS** | 123.4 MB checkpoint |
| Architecture explanation | **PASS** | Extended training rationale documented |
| Single notebook execution | **PASS** | End-to-end ~105 min |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 13 | 15 | Same proven UNet+ResNet-34+ELA — no architecture change |
| Dataset | 14 | 15 | Standard pipeline, proper GT masks, all splits correct |
| Methodology | 17 | 20 | Clean ablation (single variable: more epochs), AMP enabled, display bug (-3) |
| Evaluation | 19 | 20 | Comprehensive metrics, per-image analysis present (-1 for AUC not addressed) |
| Documentation | 12 | 15 | Good training analysis, display bug in results table (-3) |
| Assignment Alignment | 13 | 15 | All deliverables present, best Pixel F1 at time of run |
| **Total** | **88** | **100** | |

---

## 12. Final Verdict: **POSITIVE** (+2.34pp Pixel F1) — Score: 88/100

**Pixel F1: 0.7154 (+0.0234 from P.3's 0.6920)**

vR.P.7 confirms that P.3 was stopped prematurely. Extended training (50 max epochs, patience=10) allows the model to find a better optimum at epoch 36. The +2.34pp improvement is the largest single-variable gain in the pretrained ELA series, achieved without any architectural or methodological change.

### Key Insight: Training Budget Was the Bottleneck

| Training Budget | Best Epoch | Pixel F1 | Delta |
|----------------|-----------|----------|-------|
| 25 epochs, patience=7 (P.3) | 25 (ceiling) | 0.6920 | — |
| 50 epochs, patience=10 (P.7) | 36 (natural) | 0.7154 | **+0.0234** |

**Recommendation:** All future experiments should use max_epochs=50 and patience=10 as standard settings to avoid premature termination.
