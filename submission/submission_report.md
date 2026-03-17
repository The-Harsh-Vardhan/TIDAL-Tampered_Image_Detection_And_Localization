# Image Tampering Detection & Localization

**A systematic ablation study using deep learning to detect and localize tampered regions in images.**

> **TL;DR**
> - Built a pixel-level tamper localization system using a UNet segmentation model with a pretrained ResNet-34 encoder
> - Best result: **Pixel F1 = 0.7965**, **IoU = 0.6615**, **Pixel AUC = 0.9665** on the CASIA v2.0 dataset
> - Key finding: **Input representation matters most** — switching from raw RGB to Multi-Quality RGB ELA produced a **+34.19 percentage point improvement** in Pixel F1, more than all architectural changes combined

**Dataset:** CASIA v2.0 (12,614 images)
**Experiments:** 60+ controlled ablation experiments tracked with Weights & Biases
**Framework:** PyTorch + Segmentation Models PyTorch (SMP)

---

## 1. Problem Statement

Digital images can be manipulated in ways that are invisible to the human eye. Someone can splice a region from one photo into another, or copy-move a part of an image to hide or fabricate content. This is called **image tampering**.

Detecting whether an image has been tampered with is useful, but it is not enough. We also need to know **where** the tampering occurred — which specific pixels were altered. This is called **tamper localization**.

This project tackles both tasks:
- **Detection:** Is this image authentic or tampered? (binary classification)
- **Localization:** Which pixels were manipulated? (pixel-level segmentation mask)

The goal is to build a model that takes an image as input and produces a binary mask highlighting the tampered regions.

---

## 2. What This Project Builds

The system works as a pipeline:

```
Input Image (384×384 RGB)
    |
    v
Multi-Quality RGB ELA Preprocessing
    |-- Re-save image as JPEG at Q=75, Q=85, Q=95
    |-- Compare each with original to find compression inconsistencies
    |-- Preserve full RGB channels (not grayscale) to capture chrominance artifacts
    |-- Output: 9-channel tensor (3 quality levels × 3 RGB channels)
    |
    v
UNet Segmentation Model (ResNet-34 encoder, pretrained on ImageNet)
    |-- conv1 modified to accept 9-channel input (tiled weight initialization)
    |-- Encoder frozen except BatchNorm layers
    |
    v
Predicted Binary Mask (384×384 pixels)
    |-- Each pixel gets a probability: tampered or authentic
    |-- Threshold at 0.5 to produce a binary mask
```

The key insight is that **Multi-Quality RGB ELA** converts the image into a forensic representation where tampered regions become visible across multiple compression frequency bands. Using full RGB (not grayscale) preserves chrominance artifacts invisible to the human eye but learnable by the model.

---

## 3. Dataset

The project uses **CASIA v2.0**, a standard benchmark dataset for image forgery detection.

| Property | Value |
|----------|-------|
| Authentic images | 7,491 (59.4%) |
| Tampered images | 5,123 (40.6%) — 3,295 copy-move + 1,828 splicing |
| **Total** | **12,614** |
| Tampering types | Splicing, copy-move |
| Ground truth | Binary masks for each tampered image |
| Image size | Resized to 384×384 |

The data is split using stratified sampling (preserving the authentic/tampered ratio):

| Split | Count | Purpose |
|-------|-------|---------|
| Training | 8,830 (70%) | Model training |
| Validation | 1,892 (15%) | Early stopping and learning rate scheduling |
| Test | 1,892 (15%) | Final reported metrics |

All experiments use the same random seed (42) to ensure reproducibility.

---

## 4. Research Journey

This project did not arrive at its final approach on the first try. The path included multiple failed approaches, each teaching an important lesson.

### Attempt 1: Documentation-First (v0x)

The project started with a literature review and wrote extensive documentation before writing any code. Too many ideas from research papers were added at once, leading to a system that was complex on paper but untested.

**Lesson:** Documentation without experimentation leads to untested assumptions. Start coding early.

### Attempt 2: Kaggle Notebook Reproduction (vK.x.x)

A promising Kaggle notebook for image tampering detection was found and adapted. However, an audit revealed the notebook had **data leakage** — test images were leaking into training, inflating results. After fixing the leakage, results dropped significantly because the model was training from scratch on only ~10,000 images.

**Lesson:** Always audit for data leakage. Training from scratch on small datasets is insufficient — the model only learns low-level features like edges and corners.

### Attempt 3: Research Paper Baseline (vR.x.x)

The project pivoted to reproducing a published research paper (ETASR custom CNN). This produced a working classification model (best accuracy: 90.23%), but the architecture was **classification-only** — it could say "tampered" or "authentic" but could not produce a localization mask.

**Lesson:** Verify that the base architecture supports all required tasks before investing in ablation studies.

### Attempt 4: Pretrained Ablation Study (vR.P.x.x) — Final Approach

Armed with lessons from all prior failures, the project adopted:
- A **pretrained encoder** (ResNet-34 on ImageNet) instead of training from scratch
- A **UNet segmentation model** for pixel-level localization
- A **strict one-change-per-experiment** ablation discipline
- **Weights & Biases** for experiment tracking

This is the approach that produced all results below.

---

## 5. Final Approach

### Architecture

- **Model:** UNet (encoder-decoder segmentation network)
- **Encoder:** ResNet-34 pretrained on ImageNet
- **Encoder strategy:** All convolutional weights frozen; only BatchNorm layers unfrozen (adapts to forensic input). Trainable parameters: ~3.17M out of 24.4M total.
- **Input conv:** conv1 replaced with a 9-channel version (initialized by tiling the original 3-channel weights 3×)
- **Decoder:** 5 upsampling blocks (256, 128, 64, 32, 16 channels) with skip connections
- **Attention (ablation variant):** CBAM (Channel + Spatial Attention) in each decoder block — adds only 11K parameters, yields +3.57pp Pixel F1
- **Output:** Single-channel sigmoid mask at 384×384

### ELA Preprocessing (Best Variant: Multi-Quality RGB ELA)

Error Level Analysis (ELA) works by re-saving an image as JPEG at a specific quality level, then comparing the re-saved version with the original. Authentic regions compress uniformly, but tampered regions show inconsistencies because they were compressed at a different level or not at all.

The best variant — **Multi-Quality RGB ELA (9-channel)** — uses three full-color ELA maps at Q=75, Q=85, and Q=95, stacked as a 9-channel input. Each quality level reveals compression artifacts at different frequency bands. Using full RGB (not grayscale) preserves chrominance channel inconsistencies that are visible to the detector but not to the human eye.

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
| Hardware | Kaggle T4/P100 / Google Colab T4 |

---

## 6. Results

### Best Model: vR.P.19

| Metric | Value |
|--------|-------|
| **Pixel F1** | **0.7965** |
| **Pixel IoU (Jaccard)** | **0.6615** |
| **Pixel AUC** | **0.9665** |

### Top 5 Runs

| Rank | Version | Pixel F1 | Key Configuration |
|------|---------|----------|-------------------|
| 1 | **vR.P.19** | **0.7965** | Multi-Q RGB ELA 9ch, 25 epochs |
| 2 | vR.P.30.1 | 0.7762 | Multi-Q ELA + CBAM, 50 epochs |
| 3 | vR.P.30.4 | 0.7745 | Multi-Q ELA + CBAM + augmentation |
| 4 | vR.P.30.2 | 0.7721 | Multi-Q ELA + CBAM + progressive unfreeze |
| 5 | vR.P.30 | 0.7714 | Multi-Q ELA + CBAM, 25 epochs |

### Key Experiment Results

| Version | What Changed | Pixel F1 | IoU | Verdict |
|---------|-------------|----------|-----|---------|
| P.1 | RGB baseline (proper GT masks) | 0.4546 | 0.2942 | Baseline |
| **P.3** | **ELA input (replacing RGB)** | **0.6920** | **0.5291** | **Strong +** |
| P.7 | Extended training (50 epochs) | 0.7154 | 0.5569 | Positive |
| P.10 | CBAM attention in decoder | 0.7277 | 0.5719 | Positive |
| P.15 | Multi-Quality ELA grayscale (Q=75/85/95) | 0.7329 | 0.5785 | Positive |
| P.16 | DCT spatial map (replacing ELA) | 0.3209 | 0.1911 | Negative |
| **P.19** | **Multi-Quality RGB ELA (9ch, full-color)** | **0.7965** | **0.6615** | **Best** |
| P.30.1 | + CBAM attention + 50 epochs | 0.7762 | — | Positive |
| P.30.4 | + CBAM + augmentation | 0.7745 | — | Positive |

### What Matters Most

The impact hierarchy across all experiments:

1. **Input representation** — ELA preprocessing was the biggest single improvement (+23.74pp Pixel F1)
2. **ELA quality levels** — Multi-quality ELA (3 Q levels) added +4.09pp over single-Q
3. **RGB vs grayscale ELA** — Full-color ELA added +6.36pp over grayscale
4. **Attention mechanisms** — CBAM gave +3.57pp for only 11K extra parameters
5. **Training duration** — Extended training and augmentation helped modestly (+2-3pp)
6. **Loss function** — Focal vs BCE made almost no difference (+0.03pp)

---

## 7. Ablation Study

Each experiment changed exactly one thing from a parent version, enabling clear cause-and-effect analysis. Results are sorted by Pixel F1 delta.

| Change | Version | Pixel F1 Delta | Effect |
|--------|---------|---------------|--------|
| RGB channels in ELA (grayscale → full-color) | P.19 | +6.36pp | Chrominance artifacts captured; biggest jump post-ELA |
| Multi-Quality ELA (single-Q → 3 Q levels) | P.15 | +4.09pp | Captures forensic signals at multiple compression levels |
| CBAM attention | P.10 | +3.57pp | Best parameter efficiency (11K params, sharp localization) |
| ELA + DCT fusion | P.17 | +3.82pp | DCT adds complementary frequency information |
| Extended training (25→50 epochs) | P.7 | +2.34pp | Best epoch often between 30–40 |
| RGB + ELA fusion (4ch) | P.4 | +1.33pp | RGB adds little value when ELA is already present |
| Progressive encoder unfreeze | P.8 | +0.65pp | Modest gain; highest pixel precision (0.8857) |
| Data augmentation | P.12 | +0.48pp | Best image-level accuracy but modest pixel F1 gain |
| Focal + Dice loss | P.9 | +0.03pp | Essentially neutral |
| Test-time augmentation | P.14b | -5.32pp | Averaging smooths sharp boundaries — hurts localization |
| DCT-only input | P.16 | -37.11pp | Block-level frequency features alone are insufficient |

---

## 8. Visual Results

Each experiment notebook generates a visualization grid showing:
- **ELA Q=85 image** — the forensic representation used as input
- **Ground truth mask** — the actual tampered region (white = tampered)
- **Predicted mask** — the model's output at threshold 0.5
- **Overlay** — green for GT, red for prediction, overlaid on original

The best model (vR.P.19) produces sharp localization masks closely matching ground truth boundaries for splicing forgeries. The CBAM variant (P.30.x) achieves a false positive rate below 3%. The model struggles most with subtle copy-move forgeries where source and target textures are visually similar.

---

## 9. Key Learnings

- **Data leakage destroys credibility.** The Kaggle notebook appeared to work well but produced artificially inflated results. Always audit data pipelines before trusting any result.
- **Pretrained models are essential for small datasets.** Training from scratch on ~10K images cannot learn meaningful features beyond edges and corners.
- **Input representation matters more than architecture.** ELA preprocessing was worth +23.74pp Pixel F1 by itself. No architectural change came close.
- **RGB channels in forensic features matter.** Naively converting ELA to grayscale loses chrominance artifacts — keeping full RGB added +6.36pp with zero architectural cost.
- **Change one thing at a time.** Early attempts that added multiple improvements simultaneously led to catastrophic failures. Strict single-variable ablation is the only way to understand what actually works.
- **Negative results are valuable.** DCT alone (-37.11pp) and TTA (-5.32pp) saved future experiments from repeating those mistakes.
- **Verify the base approach supports all tasks.** An entire research track (vR.x.x) was discontinued because it could only classify, not localize.
- **Track experiments formally.** W&B made it possible to compare 60+ experiments systematically with reproducible metrics and config logging.

---

## 10. Future Improvements

- **Cross-dataset evaluation:** Test on Columbia, Coverage, and NIST datasets to verify generalization beyond CASIA 2.0
- **Higher resolution:** Test at 512×512 to capture finer manipulation boundaries
- **Transformer-based encoders:** Swin Transformer or ConvNeXt may capture long-range dependencies better than ResNet-34
- **SRM noise filters:** Steganalysis Rich Model features as additional forensic input alongside ELA
- **CRF post-processing:** Conditional Random Fields to sharpen predicted mask boundaries

---

## 11. Experiment Tracking & Reproducibility

All experiments are tracked with **Weights & Biases**, logging:
- Per-epoch metrics: train loss, val loss, val Pixel F1, val IoU, learning rate
- Final test metrics: Pixel F1, IoU, AUC, Image Accuracy, Macro F1, Image AUC
- Prediction visualization examples

Reproducibility is enforced by:
- Fixed random seed (42) across Python, NumPy, PyTorch, and CUDA
- Stratified 70/15/15 split with `random_state=42`
- Single-variable ablation discipline (one change per experiment)
- All configs logged to W&B at run start

All final notebooks are available on Google Colab and Kaggle. Experiment logs available on the [W&B Dashboard](https://wandb.ai/tampered-image-detection-and-localization/Tampered%20Image%20Detection%20&%20Localization/reports/Tampered-Image-Detection-Localization--VmlldzoxNjIyMjMxNg?accessToken=35b8v807ums5jnxtg6z8wieul1ylpetxrv2x4n7k9tr39mwf79ngtqs8w6d6tuaa).
