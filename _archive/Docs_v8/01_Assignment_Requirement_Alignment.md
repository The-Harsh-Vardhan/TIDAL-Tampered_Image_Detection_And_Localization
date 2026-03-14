# 01 — Assignment Requirement Alignment

## Purpose

Map each formal assignment requirement to the current implementation status, identify gaps, and specify what v8 must add.

---

## Requirement Matrix

### 1. Dataset Selection & Preparation

| Requirement | Status | Evidence | Gap |
|---|---|---|---|
| Use publicly available dataset with authentic/tampered/masks | ✅ Met | CASIA v2.0 via Kaggle, 12,614 pairs | None |
| Data pipeline: cleaning, preprocessing, mask alignment | ✅ Met | Mask binarization >0, 384×384 resize, pair validation | Mask quality not independently audited |
| Proper train/val/test split | ✅ Met | Stratified 70/15/15 by forgery type, 0 leaks by path | No content-based near-duplicate check |
| Data augmentation for robustness | ⚠️ Partial | HFlip + VFlip + RandomRotate90 only | No photometric/noise/compression augmentation |

**v8 Actions:**
- Add photometric augmentations (ColorJitter, ImageCompression, GaussNoise)
- Run perceptual hash check for near-duplicate leak detection
- Report augmentation ablation

### 2. Model Architecture & Learning

| Requirement | Status | Evidence | Gap |
|---|---|---|---|
| Train a model to predict tampered regions | ✅ Met | SMP U-Net / ResNet34, pixel-level mask output | Copy-move F1=0.31 is near-failure |
| Architecture choice is up to you | ✅ Met | U-Net chosen with documented rationale | No comparison baseline (DeepLabV3+) mentioned |
| Loss function choice is up to you | ✅ Met | BCE + Dice combined loss | No pos_weight, batch-level Dice, no boundary loss |
| Runnable on Colab T4 or similar | ✅ Met | Kaggle 2×T4, ~24.4M params | Also needs Colab variant tested |

**v8 Actions:**
- Add BCE `pos_weight` (P0 fix)
- Consider per-sample Dice computation
- Document why U-Net chosen over DeepLabV3+ explicitly

### 3. Testing & Evaluation

| Requirement | Status | Evidence | Gap |
|---|---|---|---|
| Localization performance metrics | ✅ Met | Pixel-F1, Pixel-IoU, threshold sweep | Mixed-set inflated; tampered-only F1=0.41 underreported |
| Image-level detection accuracy | ✅ Met | Accuracy=0.8246, AUC=0.8703 | Heuristic max-probability, not a learned head |
| Visual results: Original / GT / Predicted / Overlay | ✅ Met | 4-panel visualizations, best/typical/worst | Overlay quality could be improved |

**v8 Actions:**
- Lead with tampered-only metrics (P0 fix)
- Add boundary metrics (Boundary F1)
- Add per-mask-size performance breakdown
- Consider a learned image-level classification head

### 4. Deliverables & Documentation

| Requirement | Status | Evidence | Gap |
|---|---|---|---|
| Single Colab Notebook | ⚠️ Partial | Single Kaggle notebook exists | Colab variant not verified for current run |
| Dataset explanation in notebook | ✅ Met | Markdown cells describe CASIA, splits, preprocessing | — |
| Architecture description | ✅ Met | Model rationale documented | — |
| Training strategy | ✅ Met | CONFIG system, hyperparameters documented | — |
| Evaluation results | ✅ Met | Full metric suite reported | — |
| Clear visualizations | ✅ Met | Multi-panel, Grad-CAM, failure cases | — |
| Model weights provided | ✅ Met | Checkpoint saved as artifact | — |

**v8 Actions:**
- Verify the notebook runs end-to-end on Colab T4
- Ensure docs and notebook version are aligned (currently v6.5 notebook vs v6 docs)

### 5. Bonus Points

| Requirement | Status | Evidence | Gap |
|---|---|---|---|
| Robustness against distortions | ✅ Met | JPEG, noise, blur, resize suite | All degrade F1 by ~13%, 4 conditions plateau identically |
| Subtle tampering: copy-move | ❌ Failed | Copy-move F1=0.3105 | Near-failure on assignment's explicit bonus area |
| Subtle tampering: splicing from similar textures | ⚠️ Partial | Splicing F1=0.5901 (moderate) | No texture-similarity stratification |

**v8 Actions:**
- Investigate copy-move failure mode (priority analysis)
- Add targeted copy-move augmentation or loss weighting
- Report per-forgery-type metrics prominently

---

## Overall Compliance Assessment

| Category | Score | Comment |
|---|---|---|
| Dataset & Pipeline | 8/10 | Solid, but augmentation is minimal |
| Architecture & Learning | 6/10 | Functional but loss design has significant gaps |
| Testing & Evaluation | 7/10 | Comprehensive framework, but primary metric is misleading |
| Deliverables | 7/10 | Complete, but doc-notebook version mismatch |
| Bonus (Robustness) | 5/10 | Present but reveals weaknesses |
| Bonus (Subtle tampering) | 3/10 | Copy-move is near-failure |

**Estimated overall: 6.0/10** — Meets all minimum requirements but critical gaps in loss design, metric reporting, and forgery coverage weaken the submission.
