# vR.P.14 --- Pretrained Run-01 Audit

## Overview

| Field | Value |
|-------|-------|
| **Version** | vR.P.14 |
| **Run** | Run-01 |
| **Track** | Pretrained Localization |
| **Change** | Test-Time Augmentation (4 views: orig + hflip + vflip + hvflip) |
| **Parent** | vR.P.3 (ELA input, frozen body + BN unfrozen) |
| **GPU** | Tesla P100-PCIE-16GB |
| **Framework** | PyTorch 2.9.0 + SMP 0.5.0 |

---

## Experiment Goal

Test whether averaging predictions from 4 geometric views (original, horizontal flip, vertical flip, both flips) at inference time improves localization accuracy. Training is identical to P.3 --- TTA is evaluation-only.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 384 |
| BATCH_SIZE | 16 |
| ENCODER | ResNet-34 (frozen body + BN unfrozen) |
| LEARNING_RATE | 1e-3 |
| ELA_QUALITY | 90 |
| EPOCHS | 25 |
| PATIENCE | 7 |
| NUM_WORKERS | 2 |
| Loss | BCE + Dice (same as P.3) |
| Split | 70/15/15 stratified |
| TTA_VIEWS | 4 (orig, hflip, vflip, hvflip) |

---

## Training Summary

| Metric | Value |
|--------|-------|
| Epochs completed | 25 / 25 (all) |
| Best epoch | 25 |
| Best val loss | 0.4109 |
| Best val Pixel F1 | 0.7243 |
| Best val IoU | 0.5678 |
| Early stopping | No (patience never exceeded) |
| LR schedule | 1e-3 -> 5e-4 (ep16) -> 2.5e-4 (ep22) |

---

## Test Results

### Pixel-Level --- TTA vs No-TTA Comparison

| Metric | No TTA | With TTA | Delta |
|--------|--------|----------|-------|
| **Pixel F1** | **0.6919** | 0.6388 | **-5.32pp** |
| **Pixel IoU** | **0.5290** | 0.4693 | **-5.97pp** |
| Pixel Precision | 0.8356 | 0.8355 | -0.02pp |
| Pixel Recall | 0.5904 | 0.5170 | **-7.34pp** |
| Pixel AUC | 0.9528 | **0.9618** | **+0.90pp** |

### Image-Level

**NOT AVAILABLE** --- Cell 18 crashed with `NameError: name 'test_probs' is not defined`. Cells 18-27 never executed.

### Confusion Matrix

**NOT AVAILABLE** --- blocked by cell 18 crash.

---

## Comparison with Parent (vR.P.3)

| Metric | P.3 | P.14 (No TTA) | P.14 (With TTA) |
|--------|-----|----------------|------------------|
| Pixel F1 | 0.6920 | 0.6919 | 0.6388 |
| Pixel IoU | 0.5291 | 0.5290 | 0.4693 |
| Pixel AUC | 0.9528 | 0.9528 | 0.9618 |

No-TTA metrics are identical to P.3 (confirming same training). TTA **degraded** all hard-threshold metrics.

**Verdict: NEGATIVE (TTA hurt Pixel F1 by -5.32pp)**

---

## Strengths

1. **Clean experimental design** --- training is unchanged, isolating TTA's effect perfectly
2. **Lossless geometric transforms** --- no interpolation artifacts in the augmentation
3. **Both TTA and no-TTA computed** in the same pass for fair comparison
4. **AUC improved** (+0.90pp), showing probability calibration is slightly better
5. **Perfect reproducibility** --- no-TTA metrics match P.3 exactly

---

## Weaknesses

1. **TTA degraded Pixel F1 by 5.32pp** --- the opposite of what was expected
2. **Recall crashed by 7.34pp** --- averaging pushes borderline pixels below 0.5 threshold
3. **Run is INCOMPLETE** --- cell 18 crash prevented image-level metrics, confusion matrix, visualizations, and model save
4. **4x inference cost** for worse results
5. **No threshold optimization** --- TTA averaging shifts the probability distribution; the 0.5 threshold is no longer optimal

---

## Major Issues

1. **Critical code bug:** Cell 18 references `test_probs` and `test_labels` which don't exist --- should be `preds_tta` and `labels_arr` (from cell 17). This prevented 10 cells from executing, losing image-level metrics, visualizations, and model save.
2. **TTA failure not diagnosed:** The run shows TTA hurts at threshold=0.5, but doesn't test whether it helps with an optimized threshold. The AUC improvement suggests TTA improves ranking quality even though it worsens binary classification at the default boundary.
3. **Threshold-dependent evaluation only:** All reported metrics (except AUC) use threshold=0.5. TTA smooths probabilities, requiring threshold recalibration. This wasn't done.

---

## Minor Issues

1. No visualization of TTA prediction averaging (e.g., showing how individual views differ)
2. Training completed all 25 epochs with best at epoch 25 --- model was likely still improving
3. No per-image analysis of where TTA helped vs hurt

---

## Roast

**As a strict conference reviewer:**

The idea was sound in principle --- TTA is a standard technique that provides free accuracy gains in classification and semantic segmentation. But applying it naively to this problem reveals a fundamental misunderstanding of how TTA interacts with probability averaging and hard thresholds.

The core problem: averaging probability maps from 4 views pulls predictions toward the mean. For pixels near the 0.5 decision boundary (which is most of the boundary region), this averaging crosses them below threshold, producing **smaller predicted masks** with higher precision but much lower recall. The AUC improvement (+0.9pp) proves the probability rankings are actually better --- if you had recalibrated the threshold, you might have seen improvement.

**The code bug in cell 18 is inexcusable for a research notebook.** Variables from cell 17 (`preds_tta`, `labels_arr`) were not carried through to cell 18's image-level evaluation code, which still references the old P.3 variable names (`test_probs`, `test_labels`). This is a clear sign the TTA evaluation code was bolted onto the base notebook without end-to-end testing. As a result, 40% of the notebook's evaluation is missing, and the model was never saved.

**Verdict:** Interesting negative result, poorly executed. The finding that "naive TTA at threshold=0.5 hurts segmentation" is genuinely useful, but the missing threshold optimization and broken post-evaluation code significantly undermine the experiment's scientific value.

---

## Assignment Alignment

| Deliverable | Status |
|-------------|--------|
| Pixel-level prediction | Yes (TTA and no-TTA) |
| GT mask comparison | Yes (pixel metrics computed) |
| Standard metrics (F1, IoU, AUC) | Partial (pixel done, image-level MISSING) |
| Visual results | MISSING (cell crash) |
| Model weights saved | MISSING (cell crash) |
| Single notebook | Yes |
| Localization masks | Yes |

---

## Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture Implementation | 12 | /15 | Same UNet+ResNet-34; TTA implementation is clean |
| Dataset Handling | 14 | /15 | Proper ELA pipeline, stratified split |
| Experimental Methodology | 10 | /20 | Negative result not investigated (threshold opt missing); critical code bug |
| Evaluation Quality | 10 | /20 | Pixel metrics complete; image-level MISSING due to crash |
| Documentation Quality | 10 | /15 | Good TTA explanation; incomplete results |
| Assignment Alignment | 8 | /15 | No model save, no visualizations, missing image metrics |

### **Final Score: 64/100**
