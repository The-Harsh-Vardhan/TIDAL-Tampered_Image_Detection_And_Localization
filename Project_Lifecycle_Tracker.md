# Project Lifecycle Tracker

## Tampered Image Detection & Localization

**Project:** Big Vision Internship Assignment
**Author:** Harsh Vardhan
**Duration:** Multi-version development across 60+ experiment iterations
**Latest Version:** vR.P.41 (Pretrained Track) / vK.12.0 (Kaggle Track) / vR.1.7 (ETASR Track)
**Best Model:** vR.P.19 — Multi-Quality RGB ELA 9-Channel (Pixel F1 = 0.7965)
**W&B Dashboard:** [Tampered Image Detection & Localization](https://wandb.ai/tampered-image-detection-and-localization/Tampered%20Image%20Detection%20&%20Localization/reports/Tampered-Image-Detection-Localization--VmlldzoxNjIyMjMxNg?accessToken=35b8v807ums5jnxtg6z8wieul1ylpetxrv2x4n7k9tr39mwf79ngtqs8w6d6tuaa)

---

## 1. Project Overview

### Objective

Develop a deep learning model that **detects** whether an image has been tampered and **localizes** the manipulated region at pixel level by producing a binary segmentation mask.

### Problem Definition

Image tampering (copy-move, splicing, inpainting) is increasingly difficult to detect visually. This project builds an automated system that:

1. **Classifies** an image as authentic or tampered (binary image-level decision)
2. **Localizes** the tampered region by predicting a pixel-level mask highlighting altered areas

### Expected Outputs

| Output | Format | Description |
|--------|--------|-------------|
| Classification label | Binary (0/1) | Authentic vs tampered |
| Segmentation mask | 384x384 single-channel | Pixel-level tamper probability map |
| Visual overlays | RGB image | Original / GT / Predicted / Overlay comparison |

### Assignment Requirements

From the Big Vision Internship Assignment specification:

1. **Dataset Selection & Preparation** — Use publicly available datasets (CASIA, Coverage, CoMoFoD, or Kaggle). Handle cleaning, preprocessing, mask alignment. Proper train/val/test splits. Apply augmentation for robustness.
2. **Model Architecture & Learning** — Train a model to predict tampered regions. Architecture and loss function choice is open. Must be runnable on Google Colab T4 GPU.
3. **Testing & Evaluation** — Thorough localization and detection evaluation using standard metrics. Clear visual results (Original, GT, Predicted, Overlay).
4. **Deliverables & Documentation** — Single Colab Notebook with dataset explanation, architecture description, training strategy, hyperparameter choices, evaluation results, and visualizations. Provide notebook link, trained weights, and scripts.
5. **Bonus** — Robustness testing against distortions (JPEG, resize, crop, noise). Detecting subtle copy-move and splicing artifacts.

---

## 2. Technology Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| Language | Python 3.10+ | Primary development language |
| Deep Learning | PyTorch 2.x | Model training and inference |
| Segmentation Library | Segmentation Models PyTorch (SMP) | Pretrained encoder-decoder architectures |
| Augmentation | Albumentations | Image and mask augmentation pipeline |
| Experiment Tracking | Weights & Biases (W&B) | Metrics logging, visualization, artifact storage |
| Image Processing | OpenCV (cv2) | ELA computation, image I/O, contour detection |
| Data Analysis | Pandas, NumPy | Dataset management, metric computation |
| Visualization | Matplotlib, Seaborn | Training curves, prediction grids, evaluation plots |
| Model Analysis | torchinfo | Parameter counting, layer-by-layer model summary |
| Explainability | Custom Grad-CAM | Encoder attention heatmaps |
| Hardware | Kaggle T4 GPU (15 GB VRAM) | Training and evaluation |
| Notebooks | Jupyter / Kaggle / Google Colab | Development environment |

---

## 3. Dataset Information

### Primary Dataset: CASIA 2.0 Upgraded

| Property | Value |
|----------|-------|
| **Source** | Kaggle (`harshv777/casia2-0-upgraded-dataset`) |
| **Total valid pairs** | 12,614 images |
| **Authentic images** | 7,491 (59.4%) |
| **Tampered images** | 5,123 (40.6%) |
| **Copy-move forgeries** | 3,295 (26.1% of total) |
| **Splicing forgeries** | 1,828 (14.5% of total) |
| **Mask availability** | Ground truth binary masks for all tampered images |
| **Split strategy** | 70 / 15 / 15 stratified by label |
| **Training set** | 8,829 images |
| **Validation set** | 1,892 images |
| **Test set** | 1,893 images |
| **Input resolution** | 384x384 (vR.P track) / 256x256 (vK track) / 128x128 (vR ETASR track) |

### Dataset Challenges

- **Class imbalance at pixel level**: Most pixels in tampered images are still authentic, making segmentation difficult
- **Small tampered regions**: Many images have <2% tampered area, challenging detection
- **Forgery type imbalance**: Copy-move (3,295) significantly outnumbers splicing (1,828)
- **JPEG compression artifacts**: Dataset contains JPEG artifacts that can act as shortcuts

---

## 4. Research Tracks Overview

The project evolved through four parallel research tracks, each addressing different aspects of the problem:

| Track | Versions | Architecture | Best Metric | Experiments | Status |
|-------|----------|-------------|-------------|-------------|--------|
| **Track A: vK (Kaggle Baseline)** | vK.1–vK.12.0 | Custom UNet → TamperDetector | Tam-F1 = 0.4101 (v6.5) | 25 source + 22 runs | Superseded |
| **Track B: vR (ETASR Paper)** | vR.0–vR.1.7 | ETASR CNN (classification) | Test Acc = 90.23% (vR.1.6) | 11 source + 16 runs | Completed |
| **Track C: vR.P (Pretrained Ablation)** | vR.P.0–vR.P.41 | UNet + ResNet-34 (SMP) | **Pixel F1 = 0.7965 (vR.P.19)** | 41 source + 37 W&B runs | **Primary track** |
| **Track D: v0x (Early Exploration)** | Approaches 1–5 | Various | N/A | 5 approaches | Completed |

### Track Lineage

```
v0x (Exploration)
 |-- Approach 1: Literature Review
 |-- Approach 2: Kaggle Baseline ---------> vK track (vK.1 -> vK.12)
 |-- Approach 3: Research Paper CNN ------> vR track (vR.0 -> vR.1.7)
 |-- Approach 4: Pretrained Model --------> vR.P track (vR.P.0 -> vR.P.41)
 |-- Approach 5: FakeShield (abandoned)

vK track:  vK.1 -> vK.3 -> vK.7 -> vK.10 -> vK.11 -> vK.12
vR track:  vR.0 -> vR.1.1 -> vR.1.3 -> vR.1.4 -> vR.1.5 -> vR.1.6(BEST) -> vR.1.7
vR.P track: P.0 -> P.1 -> P.3(ELA) -> P.7 -> P.10(CBAM) -> P.15(Multi-Q)
            -> P.19(BEST) -> P.20-P.28 -> P.30-P.30.4 -> P.40-P.41
```

---

## 5. Model Architecture Evolution

### Best Architecture: UNet + ResNet-34 + Multi-Quality RGB ELA (vR.P.19)

```
Input Image (384x384x3 RGB)
    |
    v
ELA Preprocessing (3 quality levels: Q=75, Q=85, Q=95)
    |-- Each quality produces an RGB ELA map (384x384x3)
    |-- Concatenate: 3 qualities x 3 channels = 9 channels
    |
    v
Input Tensor (384x384x9)
    |
    v
+------------------+
| ResNet-34 Encoder |  (ImageNet pretrained, frozen, BN unfrozen)
| (first conv       |   Modified for 9 input channels
|  adapted for 9ch) |
+--------+---------+
         |
    +----+----+
    |         |
    v         v
+--------+  +-------------+
| UNet   |  | Max pixel   |
| Decoder|  | probability |
| (skip  |  | -> Image    |
| conn.) |  | classification
+---+----+  +------+------+
    |              |
    v              v
Seg Mask       Class Label
(384x384x1)   (Auth/Tampered)
```

### Architecture Across All Tracks

| Track | Model | Encoder | Pretrained | Input | Resolution | Loss | Best Result |
|-------|-------|---------|------------|-------|------------|------|-------------|
| vK.1–vK.3 | Custom UNet | 5-layer CNN | No | RGB 3ch | 256x256 | CE+BCE | Tam-F1=0.20 |
| v6.5 | SMP UNet | ResNet-34 | ImageNet | RGB 3ch | 384x384 | BCEDice | Tam-F1=0.41 |
| vK.11+ | TamperDetector | ResNet-34 (SMP) | ImageNet | RGB+ELA 4ch | 256x256 | Focal+BCE+Dice+Edge | Pending |
| vR.0–vR.1.7 | ETASR CNN | Custom 3-layer | No | ELA 3ch | 128x128 | CrossEntropy | Acc=90.23% |
| **vR.P.0–P.6** | **SMP UNet** | **ResNet-34** | **ImageNet** | **RGB/ELA 3ch** | **384x384** | **BCEDice** | **F1=0.7053** |
| **vR.P.7–P.18** | **SMP UNet** | **ResNet-34** | **ImageNet** | **ELA 3ch** | **384x384** | **BCEDice/Focal** | **F1=0.7329** |
| **vR.P.19** | **SMP UNet** | **ResNet-34** | **ImageNet** | **Multi-Q RGB ELA 9ch** | **384x384** | **BCEDice** | **F1=0.7965** |
| vR.P.20–P.28 | SMP UNet | ResNet-34 | ImageNet | Various experimental | 384x384 | BCEDice | F1=0.7762 |
| vR.P.30–P.30.4 | SMP UNet | ResNet-34 | ImageNet | Multi-Q ELA + CBAM | 384x384 | Various | F1=0.7762 |
| vR.P.40–P.41 | SMP UNet | Custom Inception | No | Multi-Q RGB ELA 9ch | 384x384 | BCEDice | Not yet run |

---

## 6. Experiment Version History

### Track C: vR.P Pretrained Ablation (Primary Track)

| Version | Change | Pixel F1 | IoU | Pixel AUC | Test Acc | Epochs | Verdict |
|---------|--------|----------|-----|-----------|----------|--------|---------|
| **vR.P.0** | ResNet-34 frozen, RGB (divg07, no GT) | 0.3749 | 0.2307 | 0.8486 | 70.63% | 24 | Baseline (no GT) |
| **vR.P.1** | Dataset fix + GT masks (sagnikkayalcse52) | 0.4546 | 0.2942 | 0.8509 | 70.15% | 25 | Proper baseline |
| **vR.P.1.5** | Speed optimizations | 0.4227 | 0.2680 | 0.8560 | 71.05% | 23 | NEUTRAL |
| **vR.P.2** | Gradual unfreeze (layer3+layer4) | 0.5117 | 0.3439 | 0.8688 | 69.04% | 14 | POSITIVE |
| **vR.P.3** | ELA input (replace RGB, BN unfrozen) | 0.6920 | 0.5291 | 0.9528 | 86.79% | 25 | STRONG POSITIVE |
| **vR.P.4** | 4-channel RGB+ELA | 0.7053 | 0.5447 | 0.9433 | 84.42% | 25 | NEUTRAL |
| **vR.P.5** | ResNet-50 encoder | 0.5137 | 0.3456 | 0.8828 | 72.00% | 25 | POSITIVE |
| **vR.P.6** | EfficientNet-B0 encoder | 0.5217 | 0.3529 | 0.8708 | 70.68% | 23 | POSITIVE |
| **vR.P.7** | Extended training (50 epochs) | 0.7154 | 0.5569 | 0.9504 | 87.37% | 46 | POSITIVE |
| **vR.P.8** | Progressive unfreeze (layer4 only) | 0.6985 | 0.5367 | 0.9541 | 87.59% | 32 | NEUTRAL |
| **vR.P.9** | Focal+Dice loss | 0.6923 | 0.5294 | 0.9323 | 87.16% | 25 | NEUTRAL |
| **vR.P.10** | Focal+Dice + CBAM attention | 0.7277 | 0.5719 | 0.9573 | 87.32% | 25 | POSITIVE |
| **vR.P.12** | Augmentation + Focal+Dice | 0.6968 | 0.5347 | 0.9502 | 88.48% | 45 | NEUTRAL |
| **vR.P.14/14b** | Test-Time Augmentation (TTA) | 0.6388 | 0.4693 | 0.9618 | 87.43% | 25 | NEGATIVE |
| **vR.P.15** | Multi-Quality ELA (Q=75/85/95, gray) | 0.7329 | 0.5785 | 0.9608 | 87.53% | 25 | POSITIVE |
| **vR.P.16** | DCT spatial map baseline | 0.3209 | 0.1911 | 0.7778 | 61.60% | 18 | NEGATIVE |
| **vR.P.17** | ELA + DCT spatial fusion (6ch) | 0.7302 | 0.5751 | 0.9431 | 87.06% | 25 | POSITIVE |
| **vR.P.18** | JPEG compression robustness | INVALID | — | — | — | — | INVALID |
| **vR.P.19** | **Multi-Q RGB ELA (9ch)** | **0.7965** | **0.6615** | **0.9665** | **—** | **25** | **SERIES BEST** |
| vR.P.20 | ELA magnitude + chrominance direction | 0.7439 | 0.5923 | 0.9571 | — | 25 | POSITIVE |
| vR.P.23 | Chrominance channel analysis | 0.5981 | 0.4268 | 0.9211 | — | 25 | NEGATIVE |
| vR.P.24 | Noiseprint forensic features | 0.6285 | 0.4583 | 0.9012 | — | 25 | NEGATIVE |
| vR.P.26 | Segmentation + classification head | — | — | — | — | — | Experimental |
| vR.P.27 | JPEG compression augmentation | 0.7523 | 0.6029 | 0.9581 | — | 25 | POSITIVE |
| vR.P.28 | Cosine annealing LR scheduler | 0.7601 | 0.6132 | 0.9623 | — | 25 | POSITIVE |
| vR.P.30 | Multi-Q ELA + CBAM attention | 0.7714 | 0.6280 | 0.9641 | — | 25 | POSITIVE |
| **vR.P.30.1** | Multi-Q ELA + CBAM (50 epochs) | **0.7762** | **0.6344** | **0.9651** | **—** | **50** | **2nd BEST** |
| vR.P.30.2 | Multi-Q ELA + CBAM + unfreeze | 0.7721 | 0.6289 | 0.9638 | — | 25 | POSITIVE |
| vR.P.30.3 | Multi-Q ELA + CBAM + Focal+Dice | 0.7698 | 0.6258 | 0.9629 | — | 25 | POSITIVE |
| vR.P.30.4 | Multi-Q ELA + CBAM + augmentation | 0.7745 | 0.6321 | 0.9645 | — | 25 | POSITIVE |
| vR.P.40.1–41 | Custom Inception encoders | — | — | — | — | — | Not yet run |

### Track B: vR ETASR Classification

| Version | Change | Test Accuracy | Status |
|---------|--------|--------------|--------|
| **vR.0** | Baseline paper reproduction | — | Initial |
| **vR.1.1** | Proper 70/15/15 split + fixed metrics | 88.38% | Honest baseline |
| **vR.1.2** | Data augmentation | REJECTED | Incompatible with architecture |
| **vR.1.3** | Class weights | 88.59% | Marginal improvement |
| **vR.1.4** | BatchNormalization | 89.12% | Stable training |
| **vR.1.5** | ReduceLROnPlateau | 89.75% | Better convergence |
| **vR.1.6** | Deeper CNN (3rd Conv2D) | **90.23%** | **ETASR BEST** |
| **vR.1.7** | GlobalAveragePooling | 89.91% | NEUTRAL |

### Track A: vK Kaggle Baseline

| Version | Architecture | Tam-F1 | Status |
|---------|-------------|--------|--------|
| vK.3 | Custom UNet (scratch) | ~0.20 (est.) | Completed |
| **v6.5** | **SMP ResNet-34** | **0.4101** | **Best vK segmentation** |
| v8 | SMP ResNet-34 | 0.2949 | Regressed (pos_weight bug) |
| **vK.10.6** | Custom UNet (scratch) | 0.2213 | Best from-scratch |
| vK.11.0–12.0 | TamperDetector (SMP+ELA) | Pending | Not yet run |

---

## 7. Key Findings and Insights

### Input Preprocessing Impact (Most Important Finding)

| Input Type | Best F1 | Example Version | Improvement vs RGB |
|------------|---------|-----------------|-------------------|
| Raw RGB (frozen encoder) | 0.4546 | vR.P.1 | Baseline |
| ELA single-quality (Q=90) | 0.6920 | vR.P.3 | **+23.74pp** |
| 4-channel RGB+ELA | 0.7053 | vR.P.4 | +25.07pp |
| Multi-Quality ELA grayscale (Q=75/85/95) | 0.7329 | vR.P.15 | +27.83pp |
| **Multi-Quality RGB ELA 9ch (Q=75/85/95)** | **0.7965** | **vR.P.19** | **+34.19pp** |
| DCT spatial map alone | 0.3209 | vR.P.16 | -13.37pp |
| Noiseprint features | 0.6285 | vR.P.24 | +17.39pp |

**Conclusion:** ELA preprocessing is the single most impactful variable. Multi-quality RGB ELA at 9 channels (vR.P.19) outperforms every other input configuration by a significant margin.

### Encoder Architecture

- ResNet-34 is sufficient — neither ResNet-50 (+5.91pp over RGB baseline but below ELA) nor EfficientNet-B0 improved over ResNet-34 when combined with ELA
- Pretrained encoders are essential for small datasets (8,829 training images)
- Frozen encoder with unfrozen BatchNorm is the optimal strategy

### Training Strategy

- Extended training (50 epochs) provides +2-3pp improvement over 25 epochs
- CBAM attention helps +3-5pp but is secondary to input quality
- Focal+Dice loss does not consistently outperform BCE+Dice
- TTA actually hurts (vR.P.14: -5.32pp) — averaging smooths out precise localization
- Data augmentation gives marginal benefits for segmentation (+0.48pp)

### Diminishing Returns

After Pixel F1 ~0.78, combining improvements shows diminishing returns:
- vR.P.30.1 (Multi-Q ELA + CBAM + 50ep) = 0.7762 (did not surpass P.19's 0.7965)
- This suggests the model approaches the ceiling of what the dataset/architecture can support

---

## 8. Evaluation Strategy

### Primary Metrics (vR.P Track)

| Metric | Type | Description |
|--------|------|-------------|
| **Pixel F1** | Segmentation | Primary metric — harmonic mean of pixel precision and recall |
| **IoU (Jaccard)** | Segmentation | Intersection-over-Union of predicted vs GT mask |
| **Pixel AUC** | Segmentation | Threshold-independent pixel-level quality |
| **Image Accuracy** | Classification | Correct tampered/authentic classification |
| **Tampered F1 (cls)** | Classification | F1 for the tampered class |
| **Macro F1 (cls)** | Classification | Average F1 across both classes |

### Critical Insight: Tampered-Only Metrics

Mixed-set metrics are inflated because authentic images have all-zero GT masks, scoring perfect Dice/IoU/F1. From vR.P.3 onward, all metrics are computed on tampered images only.

---

## 9. Reproducibility

### Constants Across All vR.P Experiments

| Parameter | Value |
|-----------|-------|
| Dataset | CASIA v2.0 (sagnikkayalcse52, from P.1 onward) |
| Random seed | 42 |
| Split | 70/15/15 stratified by label |
| Resolution | 384x384 |
| Decoder | UNet (SMP default, skip connections from all 4 encoder stages) |
| Batch size | 16 |
| Max epochs | 25 (unless explicitly ablated) |
| Early stopping | patience=7, monitor=val_loss |
| Framework | PyTorch + SMP |

### Single-Variable Ablation Methodology

Every experiment changes exactly **one variable** from its parent version. This isolates the effect of each modification and enables clear causal attribution.

### Ablation Discipline Exceptions

Four experiments deviate from the single-variable protocol by changing multiple variables simultaneously:

| Experiment | Changes Made | Note |
|-----------|-------------|------|
| vR.P.8 | Progressive encoder unfreeze + 50 epochs (was 25) | 2 simultaneous changes |
| vR.P.12 | Data augmentation + CBAM (inherited) | 2 simultaneous changes |
| vR.P.13 | Focal+Dice loss + inherits P.12's combined changes | 3 compounded changes from P.3 |
| vR.P.28 | Cosine annealing LR + different training schedule | 2 simultaneous changes |

These results remain valid but should be interpreted as combined-effect measurements, not isolated ablations. See [`AUDIT_REPORT.md`](Notebooks/research_tracks/vR.P/pretrained_ablation_experiments/final%20runs/AUDIT_REPORT.md) for the full notebook audit.

---

## 10. Error Analysis

### Common Failure Modes

| Failure Mode | Severity | Mitigation |
|-------------|----------|------------|
| Very small tampered regions (<2% area) | High | Focal loss, mask-size stratified eval |
| Copy-move with similar textures | High | Multi-quality ELA captures compression artifacts |
| Low contrast manipulations | Medium | ELA preprocessing highlights compression inconsistencies |
| JPEG double compression artifacts | Medium | Multi-quality ELA at Q=75/85/95 |
| Mask boundary imprecision | Medium | CBAM attention in decoder |
| DCT features catastrophic failure | Critical | vR.P.16 proved DCT alone is insufficient |

### Key Debugging History

| Issue | Root Cause | Resolution |
|-------|------------|------------|
| v8 regression (-28% F1) | pos_weight=30.01 bug | Identified in audit; not propagated |
| vK.10.3b collapse | Early stopping killed from-scratch training | Extended to 100 epochs |
| vR.P.18 INVALID | Checkpoint not found during evaluation | Run invalidated |
| Mixed-set metric inflation | Authentic images score perfect masks | Switched to tampered-only metrics |
| TTA degradation (P.14) | Averaging smooths precise boundaries | TTA abandoned for localization |

---

## 11. Lessons Learned

### Architecture and Training

1. **ELA is the single most impactful variable.** Switching from RGB to ELA input boosted Pixel F1 by +23.74pp — more than any architectural change.
2. **Multi-quality ELA captures more forensic signal.** Using 3 JPEG quality levels (75/85/95) instead of 1 added another +10pp.
3. **Pretrained encoders are essential for small datasets.** Custom CNNs plateau at F1~0.22 (100 epochs) vs pretrained ResNet-34 at F1~0.45 (25 epochs).
4. **ResNet-34 is sufficient.** Neither ResNet-50 nor EfficientNet-B0 improved results with ELA input.
5. **CBAM attention helps but is secondary.** +3-5pp improvement, but input quality matters more.
6. **TTA hurts localization.** Averaging predictions smooths away precise mask boundaries.

### Methodology

7. **Single-variable ablation is essential.** Without it, v8's regression would have been attributed to architecture changes instead of a pos_weight bug.
8. **Mixed-set metrics are misleading.** Always report tampered-only metrics.
9. **Diminishing returns are real.** After F1~0.78, combining improvements does not compound linearly.
10. **Alternative forensic features underperform ELA.** DCT (catastrophic), YCbCr, and Noiseprint all fell short.

---

## 12. Future Improvements

### High Priority

| Improvement | Expected Impact |
|-------------|----------------|
| Cross-dataset evaluation (Coverage, CoMoFoD) | Test generalization beyond CASIA |
| Vision Transformer encoders (Swin, ConvNeXt) | Potentially capture global manipulation patterns |
| Multi-scale input pipeline (256 + 512) | Better handling of varying tampered region sizes |

### Medium Priority

| Improvement | Expected Impact |
|-------------|----------------|
| SRM (Steganalysis Rich Model) filters | Noise-level inconsistency detection |
| Ensemble of RGB + ELA + frequency models | Complementary feature fusion |
| CRF post-processing | Sharper mask boundaries |
| Larger/better datasets | Break the F1~0.80 ceiling |

---

## 13. Repository Structure

```
submission/                              # Final deliverables for review
    final_notebook.ipynb                 # vR.P.19 (best model)
    submission_report.md
    model_weights_link.txt

Notebooks/
    final/                               # Best notebooks curated
    research_tracks/
        v0x/documentation_experiments/   # Early exploration (5 approaches)
        vK/kaggle_baseline_experiments/  # Kaggle track (25 source, 22 runs)
        vR/research_paper_experiments/   # ETASR track (11 source, 16 runs)
        vR.P/pretrained_ablation_experiments/  # Primary track (41 source, 22 runs)

experiments/
    wandb_runs/                          # 37 W&B exported run notebooks
    wandb_tracking/                      # W&B infrastructure

Docs/
    submission_report/                   # Clean submission documents
    research_docs/                       # Full research documentation
        ablation_study/                  # Ablation plans, audits, leaderboards

scripts/                                 # Python build/utility scripts
models/                                  # Pretrained weight analysis
configs/                                 # Training configs, sweep definitions
data_access/                             # Dataset info and download links
_archive/                                # Historical artifacts
```

---

## Appendix A: vR.P Experiment Tree

```
vR.P.0 (RGB, divg07, no GT)
  |
vR.P.1 (dataset fix, GT masks) ----+---- vR.P.5 (ResNet-50)
  |                                 |---- vR.P.6 (EfficientNet-B0)
vR.P.1.5 (speed optimizations)
  |
vR.P.2 (gradual unfreeze)
  |
vR.P.3 (ELA input) ----+---- vR.P.7 (extended 50ep, F1=0.7154)
  |                     |---- vR.P.8 (progressive unfreeze, F1=0.6985)
  |                     |---- vR.P.9 (Focal+Dice, F1=0.6923)
  |                     |---- vR.P.10 (CBAM, F1=0.7277)
  |                     |---- vR.P.16 (DCT only — CATASTROPHIC)
  |
vR.P.4 (4ch RGB+ELA, F1=0.7053)
  |
vR.P.12 (augmentation, F1=0.6968)
vR.P.14/14b (TTA — NEGATIVE)
vR.P.15 (Multi-Q ELA gray, F1=0.7329)
  |
vR.P.17 (ELA+DCT fusion, F1=0.7302)

vR.P.19 (Multi-Q RGB ELA 9ch, F1=0.7965) *** SERIES BEST ***
  |
  +---- vR.P.20 (magnitude+chrominance, F1=0.7439)
  +---- vR.P.23 (chrominance only — NEGATIVE)
  +---- vR.P.24 (Noiseprint — NEGATIVE)
  +---- vR.P.27 (JPEG augmentation, F1=0.7523)
  +---- vR.P.28 (cosine annealing, F1=0.7601)
  +---- vR.P.30 (Multi-Q + CBAM, F1=0.7714)
           |
           +---- vR.P.30.1 (50ep, F1=0.7762) *** 2nd BEST ***
           +---- vR.P.30.2 (unfreeze, F1=0.7721)
           +---- vR.P.30.3 (Focal+Dice, F1=0.7698)
           +---- vR.P.30.4 (augmentation, F1=0.7745)

vR.P.40.1–P.41 (Custom Inception encoders — not yet run)
```

---

*This document serves as the complete project history and experiment logbook for the Tampered Image Detection & Localization project. It tracks architectural decisions, performance metrics, debugging efforts, and lessons learned across all research tracks and 60+ experiment iterations.*
