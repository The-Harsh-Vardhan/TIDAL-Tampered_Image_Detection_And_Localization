# vR.P.7 --- Pretrained Run-01 (Extended 50-Epoch) Audit

## Overview

| Field | Value |
|-------|-------|
| **Version** | vR.P.7 |
| **Run** | Run-01 (50-epoch extended) |
| **Track** | Pretrained Localization |
| **Change** | Extended training: 50 epochs (from 25), patience 10 (from 7), NUM_WORKERS 4 |
| **Parent** | vR.P.3 (ELA input, frozen body + BN unfrozen) |
| **GPU** | Tesla P100-PCIE-16GB |

**Note:** `vr-p-7-ela-extended-training-01-run-01.ipynb` and `vr-p-7-ela-extended-training-run-01.ipynb` are **byte-identical duplicates**. This audit covers both files.

---

## Experiment Goal

Test the hypothesis that P.3 was still improving at epoch 25 by extending training to 50 epochs with patience 10. Also fixes the `denormalize` NameError bug from P.3.

---

## Training Summary

| Metric | Value |
|--------|-------|
| Epochs completed | 46 / 50 |
| Best epoch | **36** |
| Best val loss | 0.3935 |
| Best val Pixel F1 | 0.7404 |
| Best val IoU | 0.5878 |
| Early stopping | Yes (epoch 46, patience 10) |
| LR schedule | 1e-3 -> 5e-4 (ep16) -> 2.5e-4 (ep22) -> 1.25e-4 (ep35) -> 6.25e-5 (ep41) -> 3.13e-5 (ep45) |

---

## Test Results

### Pixel-Level (Localization)

| Metric | Value |
|--------|-------|
| **Pixel F1** | **0.7154** |
| **Pixel IoU** | **0.5569** |
| Pixel Precision | 0.8374 |
| Pixel Recall | 0.6245 |
| Pixel AUC | 0.9504 |

### Image-Level (Classification)

| Metric | Value |
|--------|-------|
| Image Accuracy | 87.37% |
| Macro F1 | 0.8637 |
| ROC-AUC | 0.9433 |
| Au P/R/F1 | 0.8449 / 0.9644 / 0.9007 |
| Tp P/R/F1 | 0.9344 / 0.7412 / 0.8267 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|---|---|---|
| **Au** | 1084 | 40 |
| **Tp** | 199 | 570 |

---

## Comparison with Parent (vR.P.3) and Series Leader (vR.P.10)

| Metric | P.3 (25ep) | P.7 (50ep) | Delta | P.10 (CBAM) |
|--------|-----------|-----------|-------|-------------|
| Pixel F1 | 0.6920 | 0.7154 | **+2.34pp** | 0.7277 |
| Pixel IoU | 0.5291 | 0.5569 | +2.78pp | 0.5719 |
| Pixel AUC | 0.9528 | 0.9504 | -0.24pp | 0.9573 |
| Image Acc | 86.79% | 87.37% | +0.58pp | 87.32% |

**Verdict: POSITIVE (+2.34pp Pixel F1 from extended training alone)**

---

## Strengths

1. **Confirmed P.3 was undertrained** --- best epoch moved from 25 to 36, validating the hypothesis
2. **+2.34pp Pixel F1** from training budget alone (no architecture change)
3. **Clean single-variable experiment** --- only epochs and patience changed
4. **Fixed denormalize bug** from P.3
5. **Model continued improving through 5 LR reductions**, showing the scheduler was effective

---

## Weaknesses

1. **Still below P.10** (0.7154 vs 0.7277) --- training budget alone can't match architectural improvement
2. **Diminishing returns beyond epoch 36** --- 10 more epochs with 3 LR reductions produced no improvement
3. **Pixel AUC slightly regressed** (-0.24pp from P.3)
4. **Training gap widens** --- train_loss=0.16 vs val_loss=0.39 at epoch 46 suggests mild overfitting

---

## Roast

A solid, well-executed experiment that confirms P.3 needed more training time. The +2.34pp Pixel F1 gain is genuine and scientifically clean. However, this is not a surprising result --- it was obvious from P.3's training curve that the model was still improving at epoch 25.

The real question is: why wasn't this the default from the start? Training for 25 epochs with a model that's clearly still learning is a known experimental pitfall. The fix is trivial (increase epochs), and the "discovery" is just correcting an earlier inadequacy.

**Verdict:** Necessary correction, not an innovative experiment. Well-executed but low on novelty.

---

## Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture Implementation | 13 | /15 | Same proven UNet+ResNet-34 |
| Dataset Handling | 14 | /15 | Proper ELA pipeline |
| Experimental Methodology | 17 | /20 | Clean single-variable; confirmed hypothesis |
| Evaluation Quality | 19 | /20 | Full metrics suite |
| Documentation Quality | 12 | /15 | Good changelog, bug fix documented |
| Assignment Alignment | 13 | /15 | Full pipeline, model saved |

### **Final Score: 88/100**
