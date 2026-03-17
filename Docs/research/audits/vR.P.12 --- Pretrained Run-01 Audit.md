# vR.P.12 --- Pretrained Run-01 Audit

## Overview

| Field | Value |
|-------|-------|
| **Version** | vR.P.12 |
| **Run** | Run-01 |
| **Track** | Pretrained Localization |
| **Change** | ELA + Data Augmentation (Albumentations) + Focal+Dice Loss |
| **Parent** | vR.P.3 (ELA input, frozen body + BN unfrozen) |
| **GPU** | Tesla P100-PCIE-16GB |
| **Framework** | PyTorch 2.9.0 + SMP 0.5.0 + Albumentations 2.0.8 |

---

## Experiment Goal

Test whether Albumentations data augmentation (6 safe geometric/photometric transforms applied to both ELA image and mask) improves localization when combined with Focal+Dice loss, while preserving the ELA forensic signal.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 384 |
| BATCH_SIZE | 16 |
| ENCODER | ResNet-34 (frozen body + BN unfrozen) |
| LEARNING_RATE | 1e-3 |
| ELA_QUALITY | 90 |
| FOCAL_ALPHA / GAMMA | 0.25 / 2.0 |
| EPOCHS | 50 |
| PATIENCE | 10 |
| NUM_WORKERS | 4 (prefetch=2) |
| Loss | Focal + Dice |
| Split | 70/15/15 stratified |
| Augmentation | HFlip(0.5), VFlip(0.3), Rotate90(0.5), ShiftScaleRotate(0.3), GaussBlur(0.1), BrightContrast(0.2) |

---

## Training Summary

| Metric | Value |
|--------|-------|
| Epochs completed | 45 / 50 |
| Best epoch | 35 |
| Best val loss | 0.3495 |
| Best val Pixel F1 | 0.7057 |
| Best val IoU | 0.5452 |
| Early stopping | Yes (epoch 45, patience 10) |
| LR schedule | 1e-3 -> 5e-4 (ep23) -> 2.5e-4 (ep40) -> 1.25e-4 (ep44) |

**Notable:** Two severe val loss spikes at epochs 19 (0.5703) and 21 (0.7560, F1 crashed to 0.2332), indicating training instability.

---

## Test Results

### Pixel-Level (Localization)

| Metric | Value |
|--------|-------|
| **Pixel F1** | **0.6968** |
| **Pixel IoU** | **0.5347** |
| Pixel Precision | 0.8269 |
| Pixel Recall | 0.6021 |
| Pixel AUC | 0.9502 |

### Image-Level (Classification)

| Metric | Value |
|--------|-------|
| Test Accuracy | 88.48% |
| Macro F1 | 0.8756 |
| ROC-AUC | 0.9427 |
| Au Precision / Recall / F1 | 0.8528 / 0.9742 / 0.9095 |
| Tp Precision / Recall / F1 | 0.9524 / 0.7542 / 0.8418 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|---|---|---|
| **Au** | 1095 | 29 |
| **Tp** | 189 | 580 |

---

## Comparison with Parent (vR.P.3)

| Metric | P.3 | P.12 | Delta |
|--------|-----|------|-------|
| Pixel F1 | 0.6920 | 0.6968 | **+0.48pp** |
| Pixel IoU | 0.5291 | 0.5347 | +0.56pp |
| Pixel AUC | 0.9528 | 0.9502 | -0.26pp |
| Image Acc | 86.79% | 88.48% | +1.69pp |
| Image AUC | 0.9528 | 0.9427 | -1.01pp |

**Verdict: NEUTRAL (+0.48pp Pixel F1, but AUC regressed and training unstable)**

---

## Strengths

1. **Augmentation pipeline is well-designed** --- all 6 transforms are geometrically safe for ELA signal preservation
2. **Joint image+mask augmentation** correctly applies identical transforms to both
3. **Training-only augmentation** --- val/test remain clean for fair evaluation
4. **Image accuracy improved** (+1.69pp) suggesting augmentation helps classification
5. **Clean execution** --- no errors, early stopping triggered correctly
6. **Good documentation** in notebook markdown cells

---

## Weaknesses

1. **Minimal Pixel F1 improvement** (+0.48pp) --- augmentation barely moved the localization needle
2. **Training instability** --- two severe val loss spikes (epoch 19: 0.57, epoch 21: 0.76) suggest augmented training can destabilize the frozen encoder
3. **AUC regression** on both pixel (-0.26pp) and image (-1.01pp) levels
4. **45 epochs is expensive** for marginal improvement --- 3x the training cost of P.3 for +0.48pp
5. **Multiple confounds** --- augmentation AND loss change (Focal+Dice) vs P.3's BCE+Dice, making it impossible to isolate augmentation's effect
6. **Val F1 peaked at 0.7057 but test F1 was 0.6968** --- generalization gap

---

## Major Issues

1. **Confounded experiment:** P.12 changes TWO variables from P.3 (augmentation + loss function). Since P.9 showed Focal+Dice is essentially neutral (+0.03pp), the +0.48pp is likely attributable to augmentation alone, but this cannot be confirmed.
2. **Training instability not investigated:** The epoch 21 crash (F1: 0.7 -> 0.23) is a red flag. The model recovered, but the cause was not diagnosed. Possible causes: aggressive ShiftScaleRotate creating edge artifacts, or GaussianBlur degrading ELA signal.

---

## Minor Issues

1. Per-image pixel F1 for tampered images has high variance (mean=0.55, std=0.42) --- augmentation didn't reduce prediction inconsistency
2. Tampered recall (75.4%) is still below 80%, meaning 1 in 4 tampered images are missed
3. No ablation of individual augmentation transforms to identify which helped/hurt

---

## Roast

**As a strict conference reviewer:**

The hypothesis was reasonable --- augmentation should help a data-starved model generalize. But the execution confounds two variables (augmentation + loss change), violating single-variable ablation methodology. The +0.48pp Pixel F1 gain is statistically questionable given the training instability observed.

The training curve raises serious concerns: two catastrophic val loss spikes (epoch 21: F1 crashed from 0.66 to 0.23) suggest the augmentation pipeline may be damaging the frozen encoder's learned features in unexpected ways. This was not investigated.

The augmentation transforms are textbook-safe for natural images, but the interaction with ELA (which is NOT a natural image) deserved more careful analysis. GaussianBlur on ELA data quite literally smooths out the compression artifacts you're trying to detect. RandomBrightnessContrast changes the brightness scaling that ELA relies on. These should have been tested individually.

**Verdict:** Marginally useful but poorly isolated. The experiment answers "does augmentation + Focal+Dice help ELA localization?" with a tepid "barely." It would have been more informative to test augmentation alone vs P.3.

---

## Assignment Alignment

| Deliverable | Status |
|-------------|--------|
| Pixel-level prediction | Yes |
| GT mask comparison | Yes |
| Standard metrics (F1, IoU, AUC) | Yes |
| Visual results | Yes |
| Model weights saved | Yes |
| Single notebook | Yes |
| Localization masks | Yes |

---

## Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture Implementation | 13 | /15 | Same proven UNet+ResNet-34; augmentation well-integrated |
| Dataset Handling | 14 | /15 | Proper joint aug, stratified split, ELA stats |
| Experimental Methodology | 13 | /20 | Confounded (2 variables), instability not investigated |
| Evaluation Quality | 19 | /20 | Full pixel+image metrics, confusion matrix, visualizations |
| Documentation Quality | 12 | /15 | Good inline docs, augmentation table, changelog |
| Assignment Alignment | 13 | /15 | Full localization pipeline, model saved |

### **Final Score: 84/100**
