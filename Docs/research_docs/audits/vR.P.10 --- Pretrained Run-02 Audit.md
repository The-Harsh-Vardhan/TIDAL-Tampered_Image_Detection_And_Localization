# vR.P.10 --- Pretrained Run-02 Audit

## Overview

| Field | Value |
|-------|-------|
| **Version** | vR.P.10 |
| **Run** | Run-02 (Reproducibility) |
| **Track** | Pretrained Localization |
| **Change** | Focal+Dice loss + CBAM attention in UNet decoder |
| **Parent** | vR.P.3 (ELA input, frozen body + BN unfrozen) |
| **GPU** | Tesla P100-PCIE-16GB |
| **Framework** | PyTorch 2.9.0 + SMP 0.5.0 |

---

## Experiment Goal

Reproducibility verification of vR.P.10 Run-01 --- confirm that the CBAM attention + Focal+Dice results are deterministic and reproducible across independent Kaggle sessions.

---

## Test Results

### Pixel-Level (Localization)

| Metric | Run-01 | Run-02 | Delta |
|--------|--------|--------|-------|
| Pixel F1 | 0.7277 | 0.7277 | 0.0000 |
| Pixel IoU | 0.5719 | 0.5719 | 0.0000 |
| Pixel AUC | 0.9573 | 0.9573 | 0.0000 |
| Pixel Precision | 0.8611 | 0.8611 | 0.0000 |
| Pixel Recall | 0.6300 | 0.6300 | 0.0000 |

### Image-Level (Classification)

| Metric | Run-01 | Run-02 | Delta |
|--------|--------|--------|-------|
| Image Accuracy | 87.32% | 87.32% | 0.0000 |
| Macro F1 | 0.8615 | 0.8615 | 0.0000 |
| ROC-AUC | 0.9633 | 0.9633 | 0.0000 |

### Confusion Matrix (identical)

| | Pred Au | Pred Tp |
|---|---|---|
| **Au** | 1102 | 22 |
| **Tp** | 218 | 551 |

---

## Training Summary

| Metric | Run-01 | Run-02 |
|--------|--------|--------|
| Best epoch | 24 | 24 |
| Best val loss | 0.3589 | 0.3589 |
| Epochs completed | 25 | 25 |
| LR schedule changes | ep16 (5e-4), ep23 (2.5e-4) | ep16 (5e-4), ep23 (2.5e-4) |

**Every single metric, epoch log, and LR schedule change is identical across both runs.**

---

## Strengths

1. **Perfect reproducibility** --- same results to 4 decimal places across independent Kaggle sessions
2. **Validates SEED=42 determinism** across the entire pipeline (data split, model init, training, evaluation)
3. **Confirms P.10 as the current series leader** (Pixel F1 = 0.7277)
4. **Clean execution** --- no errors, all cells passed

---

## Weaknesses

1. **No new information generated** --- this is purely confirmatory
2. **Does not test cross-GPU reproducibility** (both runs on P100)
3. **Changelog/discussion markdown not updated for P.10** (still references P.3 comparisons)

---

## Roast

**As a strict conference reviewer:**

Reproducibility is the bare minimum for scientific credibility, not a research contribution. This re-run confirms the pipeline is deterministic, which is necessary but not novel. The clean execution and bit-identical results validate the experimental infrastructure.

However, true reproducibility testing would include: (a) different hardware (T4 vs P100), (b) different PyTorch versions, (c) different random seeds. This only tests "same code, same hardware, same seed" --- which is just testing that CUDA is deterministic with `cudnn.deterministic=True`.

**Verdict:** Adequate reproducibility check. Not independently meaningful as an experiment.

---

## Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture Implementation | 15 | /15 | Same proven CBAM + UNet + ResNet-34 |
| Dataset Handling | 14 | /15 | Proper ELA pipeline, stratified split |
| Experimental Methodology | 16 | /20 | Reproducibility confirmed; but only same-hardware |
| Evaluation Quality | 19 | /20 | Full pixel+image metrics, CM, visualizations |
| Documentation Quality | 10 | /15 | Discussion markdown not updated for P.10 |
| Assignment Alignment | 14 | /15 | Full pipeline, model saved |

### **Final Score: 88/100**
