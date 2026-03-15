# Image Tampering Detection & Localization

**A systematic ablation study using deep learning to detect and localize tampered regions in images.**

> **TL;DR**
> - Built a pixel-level tamper localization system using a UNet segmentation model with a pretrained ResNet-34 encoder
> - Best result: **Pixel F1 = 0.7329**, **Image Accuracy = 87.53%** on the CASIA v2.0 dataset
> - Key finding: **Input representation matters most** --- switching from raw RGB to Error Level Analysis (ELA) preprocessing produced the single largest improvement (+23.74 percentage points in Pixel F1)

**Dataset:** CASIA v2.0 (12,614 images)
**Experiments:** 17 controlled ablation experiments tracked with Weights & Biases
**Framework:** PyTorch + Segmentation Models PyTorch (SMP)

---

## 1. Problem Statement

Digital images can be manipulated in ways that are invisible to the human eye. Someone can splice a region from one photo into another, or copy-move a part of an image to hide or fabricate content. This is called **image tampering**.

Detecting whether an image has been tampered with is useful, but it is not enough. We also need to know **where** the tampering occurred --- which specific pixels were altered. This is called **tamper localization**.

This project tackles both tasks:
- **Detection:** Is this image authentic or tampered? (binary classification)
- **Localization:** Which pixels were manipulated? (pixel-level segmentation mask)

The goal is to build a model that takes an image as input and produces a binary mask highlighting the tampered regions.

---

## 2. What This Project Builds

The system works as a pipeline:

```
Input Image
    |
    v
ELA Preprocessing (Error Level Analysis)
    |-- Re-save image as JPEG at quality Q
    |-- Compare with original to find compression inconsistencies
    |-- Tampered regions show up as bright artifacts
    |
    v
UNet Segmentation Model (ResNet-34 encoder, pretrained on ImageNet)
    |
    v
Predicted Mask (384 x 384 pixels)
    |-- Each pixel gets a probability: tampered or authentic
    |-- Threshold at 0.5 to produce a binary mask
```

The key insight is that **ELA preprocessing** converts the image into a forensic representation where tampered regions become visible. The segmentation model then learns to identify these regions precisely.

---

## 3. Dataset

The project uses **CASIA v2.0**, a standard benchmark dataset for image forgery detection.

| Property | Value |
|----------|-------|
| Authentic images | 7,491 (59.4%) |
| Tampered images | 5,123 (40.6%) |
| **Total** | **12,614** |
| Tampering types | Splicing, copy-move |
| Ground truth | Binary masks for each tampered image |
| Image size | Resized to 384 x 384 |

The data is split using stratified sampling (preserving the authentic/tampered ratio):

| Split | Count | Purpose |
|-------|-------|---------|
| Training | ~8,830 (70%) | Model training |
| Validation | ~1,892 (15%) | Early stopping and learning rate scheduling |
| Test | ~1,892 (15%) | Final reported metrics |

All experiments use the same random seed (42) to ensure reproducibility.

---

## 4. Research Journey

This project did not arrive at its final approach on the first try. The path included multiple failed approaches, each teaching an important lesson.

### Attempt 1: Documentation-First (v0x)

The project started with a literature review and wrote extensive documentation before writing any code. Too many ideas from research papers were added at once, leading to a system that was complex on paper but untested.

**Lesson:** Documentation without experimentation leads to untested assumptions. Start coding early.

### Attempt 2: Kaggle Notebook Reproduction (vK.x.x)

A promising Kaggle notebook for image tampering detection was found and adapted. However, an audit revealed the notebook had **data leakage** --- test images were leaking into training, inflating results. After fixing the leakage, results dropped significantly because the model was training from scratch on only ~10,000 images.

**Lesson:** Always audit for data leakage. Training from scratch on small datasets is insufficient --- the model only learns low-level features like edges and corners.

### Attempt 3: Research Paper Baseline (vR.x.x)

The project pivoted to reproducing a published research paper (ETASR custom CNN). This produced a working classification model (best accuracy: 90.23%), but the architecture was **classification-only** --- it could say "tampered" or "authentic" but could not produce a localization mask.

**Lesson:** Verify that the base architecture supports all required tasks before investing in ablation studies.

### Attempt 4: Pretrained Ablation Study (vR.P.x.x) --- Final Approach

Armed with lessons from all prior failures, the project adopted:
- A **pretrained encoder** (ResNet-34 on ImageNet) instead of training from scratch
- A **UNet segmentation model** for pixel-level localization
- A **strict one-change-per-experiment** ablation discipline
- **Weights & Biases** for experiment tracking

This is the approach that produced all reported results below.

---

## 5. Final Approach

### Architecture

- **Model:** UNet (encoder-decoder segmentation network)
- **Encoder:** ResNet-34 pretrained on ImageNet
- **Encoder strategy:** All convolutional weights frozen; only BatchNorm layers unfrozen (adapts to forensic input). This keeps trainable parameters to ~3.17M out of 24.4M total.
- **Decoder:** 5 upsampling blocks (256, 128, 64, 32, 16 channels) with skip connections
- **Attention (best variant):** CBAM (Channel + Spatial Attention) in each decoder block --- adds only 11K parameters
- **Output:** Single-channel sigmoid mask at 384 x 384

### ELA Preprocessing

Error Level Analysis (ELA) works by re-saving an image as JPEG at a specific quality level, then comparing the re-saved version with the original. Authentic regions compress uniformly, but tampered regions show inconsistencies because they were compressed at a different level or not at all.

The best variant uses **Multi-Quality ELA**: three grayscale ELA maps at Q=75, Q=85, and Q=95, stacked as a 3-channel input. Each quality level reveals different types of artifacts.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (weight decay 1e-5) |
| Learning rate | 1e-3 |
| Loss | BCE + Dice |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 3) |
| Early stopping | Patience 7 on validation loss |
| Batch size | 16 |
| Max epochs | 25 (50 for extended runs) |
| Mixed precision | AMP + TF32 |
| Hardware | Kaggle T4/P100 GPUs |

---

## 6. Results

### Per-Metric Best Results

| Metric | Best Version | Value |
|--------|-------------|-------|
| **Pixel F1** | vR.P.15 (Multi-Q ELA) | **0.7329** |
| **Pixel IoU** | vR.P.15 (Multi-Q ELA) | **0.5785** |
| Pixel AUC | vR.P.14b (TTA) | 0.9618 |
| Image Accuracy | vR.P.12 (Augmentation) | 88.48% |
| Image Macro F1 | vR.P.12 (Augmentation) | 0.8756 |
| Image AUC | vR.P.10 (CBAM) | 0.9633 |

No single experiment dominates all metrics.

### Key Experiment Results

| Version | What Changed | Pixel F1 | IoU | Img Acc | Verdict |
|---------|-------------|----------|-----|---------|---------|
| P.1 | RGB baseline (proper GT masks) | 0.4546 | 0.2942 | 70.15% | Baseline |
| **P.3** | **ELA input (replacing RGB)** | **0.6920** | **0.5291** | **86.79%** | **Strong +** |
| P.7 | Extended training (50 epochs) | 0.7154 | 0.5569 | 87.37% | Positive |
| P.9 | Focal + Dice loss | 0.6923 | 0.5294 | 87.16% | Neutral |
| P.10 | CBAM attention in decoder | 0.7277 | 0.5719 | 87.32% | Positive |
| P.12 | Data augmentation | 0.6968 | 0.5347 | 88.48% | Neutral |
| **P.15** | **Multi-Quality ELA (Q=75/85/95)** | **0.7329** | **0.5785** | **87.53%** | **Positive** |
| P.16 | DCT spatial map (replacing ELA) | 0.3209 | 0.1911 | 61.60% | Negative |

### What Matters Most

The impact hierarchy across all experiments:

1. **Input representation** --- ELA preprocessing was the biggest single improvement (+23.74pp Pixel F1)
2. **Attention mechanisms** --- CBAM gave +3.57pp for only 11K extra parameters
3. **Training configuration** --- Extended training and augmentation helped modestly
4. **Loss function** --- Focal vs BCE made almost no difference (+0.03pp)

---

## 7. Ablation Study

Each experiment changed exactly one thing from the baseline, enabling clear cause-and-effect analysis.

| Change | Version | Pixel F1 Delta | Effect |
|--------|---------|---------------|--------|
| Multi-Quality ELA | P.15 | +4.09pp | Best overall --- captures forensic signals at multiple compression levels |
| ELA + DCT fusion | P.17 | +3.82pp | DCT adds complementary frequency information when combined with ELA |
| CBAM attention | P.10 | +3.57pp | Channel + spatial attention sharpens localization, best parameter efficiency |
| Extended training | P.7 | +2.34pp | 25 epochs was premature; best epoch was at epoch 36 |
| RGB + ELA fusion | P.4 | +1.33pp | RGB adds little value when ELA is already present |
| Progressive unfreeze | P.8 | +0.65pp | Modest gain; highest pixel precision (0.8857) |
| Data augmentation | P.12 | +0.48pp | Best image-level accuracy but modest pixel F1 gain |
| Focal + Dice loss | P.9 | +0.03pp | Essentially neutral |
| Test-time augmentation | P.14b | -5.32pp | Averaging smooth sharp boundaries --- hurts localization |
| DCT-only input | P.16 | -37.11pp | Block-level frequency features alone are insufficient |

---

## 8. Visual Results

Each experiment notebook generates a visualization grid showing:
- **Original image** --- the input photograph
- **Ground truth mask** --- the actual tampered region (white = tampered)
- **Predicted mask** --- the model's output
- **Overlay** --- predicted mask overlaid on the original image

The best models (P.15, P.10) produce sharp localization masks that closely match ground truth boundaries for splicing forgeries. False positives are rare --- the CBAM variant (P.10) achieves a false positive rate of only 2.0%. The model struggles most with subtle copy-move forgeries where source and target textures are very similar.

<!-- Visual comparison grids are available in the executed Kaggle notebook outputs for vR.P.10 and vR.P.15 -->

---

## 9. Key Learnings

- **Data leakage destroys credibility.** The Kaggle notebook appeared to work well but produced artificially inflated results. Always audit data pipelines before trusting results.
- **Pretrained models are essential for small datasets.** Training from scratch on ~10K images cannot learn meaningful features beyond edges and corners.
- **Input representation matters more than architecture.** ELA preprocessing was worth +23.74pp Pixel F1. No architectural change came close.
- **Change one thing at a time.** Early attempts that added multiple improvements simultaneously led to catastrophic failures. Strict ablation discipline is the only way to understand what actually works.
- **Negative results are valuable.** Knowing that DCT alone fails (-37.11pp) or that TTA hurts localization (-5.32pp) prevents wasted effort.
- **Verify the base approach supports all tasks.** An entire research track (vR.x.x) was discontinued because it could only classify, not localize.
- **Track experiments formally.** Using W&B made it possible to compare 17 experiments systematically with reproducible metrics.

---

## 10. Future Improvements

- **Combine best components:** Multi-Quality ELA + CBAM attention + extended training (the vR.P.30 series is testing this)
- **Additional forensic features:** SRM noise filters, YCbCr chrominance analysis
- **Higher resolution:** Test at 512 x 512 to capture finer manipulation boundaries
- **Cross-dataset evaluation:** Test on Columbia, Coverage, and NIST datasets to verify generalization
- **Transformer-based encoders:** Swin Transformer or ConvNeXt may capture long-range dependencies better than ResNet

---

## 11. Experiment Tracking & Reproducibility

All experiments are tracked with **Weights & Biases**, logging:
- Per-epoch metrics: train loss, val loss, val Pixel F1, val IoU, learning rate
- Final test metrics: Pixel F1, IoU, AUC, Image Accuracy, Macro F1, Image AUC
- Prediction visualization examples

Reproducibility was verified by running vR.P.3 and vR.P.10 twice each --- both produced identical metrics across independent runs using seed 42.

All notebooks were executed on Kaggle GPU instances (T4/P100). A centralized leaderboard notebook automatically aggregates results from W&B for comparison.
