# W&B Report Text Content for TIDAL Ablation Study
# Copy-paste each section into the corresponding W&B report markdown block.
# ============================================================


# ============================================================
# SECTION 1: TITLE & OVERVIEW
# ============================================================

# Title: TIDAL: Systematic Ablation Study for Image Tamper Detection & Localization

## Overview

This report documents a **systematic ablation study** for pixel-level image tampering detection and localization on the **CASIA v2.0** benchmark dataset (12,614 images: 7,491 authentic + 5,123 tampered).

The project explored **4 experimental tracks** over **60+ controlled experiments**, following a strict **single-variable ablation discipline** — exactly one change per experiment — to determine what actually works and why.

### Best Model: vR.P.19 (Multi-Quality RGB ELA, 9-channel)

| Metric | Value |
|--------|-------|
| **Pixel F1** | **0.7965** |
| **IoU (Jaccard)** | **0.6615** |
| **Pixel AUC** | **0.9665** |

### Central Finding

**Input representation dominates all other factors.** Switching from raw RGB to Multi-Quality RGB ELA (9-channel) produced a **+34.19 percentage point** improvement in Pixel F1 — more than all other improvements combined.

**Impact Hierarchy:** Input representation >> Attention mechanisms > Training configuration > Loss function

📎 [GitHub Repository](https://github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization)


# ============================================================
# SECTION 2: KEY FINDING — INPUT REPRESENTATION DOMINATES
# ============================================================

## Key Finding: Input Representation is Everything

The single most important decision in this pipeline is **how the image is preprocessed before feeding it to the model**. No architectural change, training trick, or loss function comes close to the impact of switching from RGB to ELA.

### What is ELA?

**Error Level Analysis (ELA)** re-saves an image as JPEG at a given quality level and measures the pixel-wise difference from the original. Authentic regions compress uniformly (low residuals). Tampered regions — composited from different sources or saved at different quality levels — show inconsistent residuals.

### Multi-Quality RGB ELA (Best Variant)

Instead of a single quality level, we use **three quality levels** (Q=75, Q=85, Q=95), each producing a **3-channel RGB** ELA map. These are stacked into a **9-channel input tensor**.

| Quality | Sensitivity | Mean Residual | Std |
|---------|------------|---------------|-----|
| Q=75 (aggressive) | Strong manipulations | 0.0684 | 0.0656 |
| Q=85 (balanced) | Medium edits | 0.0605 | 0.0604 |
| Q=95 (gentle) | Subtle changes | 0.0402 | 0.0471 |

### Input Representation Progression

| Input Type | Channels | Pixel F1 | Δ from RGB |
|-----------|----------|----------|------------|
| RGB (P.1) | 3 | 0.4546 | — |
| ELA grayscale→RGB (P.3) | 3 | 0.6920 | +0.2374 |
| RGB + ELA fusion (P.4) | 4 | 0.7053 | +0.2507 |
| Multi-Q ELA grayscale (P.15) | 3 | 0.7329 | +0.2783 |
| ELA + DCT fusion (P.17) | 6 | 0.7302 | +0.2756 |
| **Multi-Q RGB ELA (P.19)** | **9** | **0.7965** | **+0.3419** |
| DCT only (P.16) | 3 | 0.3209 | −0.1337 |

**Key insight:** Keeping RGB color information in each ELA map (P.19) instead of converting to grayscale (P.15) was worth an additional **+6.36pp**. The color channels encode *which* RGB channels were manipulated during forgery.

> The bar chart below shows Pixel F1 grouped by input feature set. The dominance of `multi_quality_ela` is immediately visible.


# ============================================================
# SECTION 3: ABLATION PROGRESSION
# ============================================================

## Ablation Progression: The Path to 0.7965

Every experiment in the vR.P track changed **exactly one variable** from a known baseline. This discipline ensures clear cause-and-effect: if performance changes, we know exactly why.

### Main Evolution Path (Trunk)

```
P.0 (RGB, no GT)     →  F1 = 0.3749
  ↓ +0.0797 (fix dataset)
P.1 (RGB, proper GT)  →  F1 = 0.4546
  ↓ +0.2374 (switch to ELA)        ← BIGGEST SINGLE JUMP
P.3 (ELA input)       →  F1 = 0.6920
  ↓ +0.0409 (multi-quality)
P.15 (Multi-Q gray)   →  F1 = 0.7329
  ↓ +0.0636 (keep RGB color)
P.19 (Multi-Q RGB 9ch) →  F1 = 0.7965  ← BEST
```

### Complete Impact Hierarchy

Ranked by delta from the ELA baseline (P.3, F1=0.6920):

| Rank | Category | Modification | Δ F1 (pp) |
|------|----------|-------------|-----------|
| 1 | Input | Multi-Q RGB ELA 9ch (P.19) | +10.45 |
| 2 | Input | Multi-Q ELA grayscale (P.15) | +4.09 |
| 3 | Input | ELA + DCT fusion (P.17) | +3.82 |
| 4 | Architecture | CBAM attention (P.10) | +3.57 |
| 5 | Training | Extended training 50ep (P.7) | +2.34 |
| 6 | Input | RGB + ELA fusion (P.4) | +1.33 |
| 7 | Training | Progressive unfreeze (P.8) | +0.65 |
| 8 | Training | Data augmentation (P.12) | +0.48 |
| 9 | Loss | Focal + Dice (P.9) | +0.03 |
| 10 | Evaluation | Test-Time Aug (P.14b) | **−5.32** |
| 11 | Input | DCT only (P.16) | **−37.11** |

> The line chart below tracks Pixel F1 across experiments in chronological order. The parallel coordinates plot shows multi-metric trade-offs across all runs.


# ============================================================
# SECTION 4: TRAINING CURVES
# ============================================================

## Training Dynamics

The training curves reveal important patterns about how different configurations converge.

### Observations

- **P.3 (ELA baseline):** Best epoch was 25/25 — the training budget was the bottleneck, not the model. This motivated P.7 (extended training).
- **P.7 (50 epochs):** Optimum reached at epoch 36, gaining +2.34pp over P.3.
- **P.8 (Progressive unfreeze):** Best epoch was at epoch 23 — during the frozen phase, *before* any unfreezing. This confirmed that freezing the encoder is the right default.
- **P.10 (CBAM):** Converges faster than the P.3 baseline due to attention focusing the decoder on relevant features.
- **P.19 (Multi-Q RGB ELA):** Benefits from the richest input representation; the validation F1 curve is consistently above all other runs.

> The line plots below show `val_pixel_f1` and `val_loss` vs `epoch` for the top performing runs.


# ============================================================
# SECTION 5: ENCODER & ARCHITECTURE COMPARISON
# ============================================================

## Architecture Comparison: Input > Encoder Size

A critical finding is that **ELA preprocessing on a small encoder beats larger encoders on raw RGB**. The model architecture matters far less than what you feed it.

### Encoder Comparison (all on RGB input)

| Encoder | Trainable Params | Pixel F1 (RGB) |
|---------|-----------------|----------------|
| ResNet-34 (P.1) | 3.17M | 0.4546 |
| ResNet-50 (P.5) | 9.0M | 0.5137 |
| EfficientNet-B0 (P.6) | 2.24M | 0.5217 |
| **ResNet-34 + ELA (P.3)** | **3.17M** | **0.6920** |

ResNet-34 with ELA (3.17M params, F1=0.6920) **vastly outperforms** ResNet-50 with RGB (9.0M params, F1=0.5137). Spending compute on better input preprocessing is 40x more effective than buying a bigger encoder.

### CBAM: Most Parameter-Efficient Improvement

CBAM attention adds only **11,402 parameters** (0.36% of trainable) but yields **+3.57pp Pixel F1**. That's **0.313pp per 1K parameters** — about 300x more efficient than upgrading from ResNet-34 to ResNet-50.

### Model Configuration (Best)

| Component | Parameters | Trainable |
|-----------|-----------|-----------|
| ResNet-34 Encoder | 21.3M | BN only (frozen conv) |
| UNet Decoder (5 blocks) | ~3.17M | Yes |
| CBAM Modules (5 blocks) | 11,402 | Yes |
| **Total** | **24.4M** | **3.17M (13%)** |

> The scatter plot below shows trainable parameters vs Pixel F1. ELA runs (blue/green) cluster in the top-left: fewer parameters, higher performance.


# ============================================================
# SECTION 6: ERROR ANALYSIS
# ============================================================

## Error Analysis: What the Model Gets Right and Wrong

### False Positive / False Negative Rates

| Configuration | FP Rate | FN Rate | Change in FP |
|--------------|---------|---------|-------------|
| RGB baseline (P.1) | 22.6% | 40.4% | — |
| ELA baseline (P.3) | 2.7% | 28.6% | −19.9pp |
| CBAM (P.10) | 2.0% | 28.3% | −0.7pp |
| Augmentation (P.12) | 2.6% | 24.6% | −0.1pp |
| TTA (P.14b) | 1.2% | 29.3% | −1.5pp |

ELA dramatically cuts false positives (22.6% → 2.7%). The remaining errors are primarily **false negatives** — the model misses tampered regions.

### Precision-Recall Trade-off (P.15 Detailed)

| Metric | Value | Meaning |
|--------|-------|---------|
| Pixel Precision | 0.8409 | When it flags tampered, it's usually right |
| Pixel Recall | 0.6496 | It misses ~35% of tampered pixels |
| Pixel F1 | 0.7329 | Harmonic balance |

The model is **conservative** — high precision, moderate recall. This is appropriate for forensic applications where **false accusations are costly**.

### Failure Modes

The model struggles with:
1. **Subtle copy-move:** When source and target textures are very similar, ELA residuals are nearly identical
2. **Tiny tampered regions** (<2% of image): Lost in 384×384 downsampling
3. **Multi-generation JPEG:** Images re-saved multiple times have uniform artifacts everywhere, destroying the differential ELA signal

> The scatter plot below shows Pixel Precision vs Pixel Recall for all runs. Points in the upper-right have the best balance.


# ============================================================
# SECTION 7: PREDICTION VISUALIZATIONS
# ============================================================

## Prediction Visualizations

Each experiment logs a 4-panel visualization grid for sampled test images:
- **Original image:** The input photograph
- **Ground truth mask:** Binary mask showing actual tampered regions (white = tampered)
- **Predicted mask:** Model output after thresholding at 0.5
- **Overlay:** Predicted mask overlaid on the original image

### Qualitative Observations (Best Models: P.10, P.15, P.19)

- Sharp localization boundaries on large spliced regions
- False positive rates of 2.0–4.1% mean authentic images very rarely get false detections
- Best performance on splicing forgeries with different source compression levels
- Struggles with: subtle copy-move (similar textures), very small regions, multi-generation JPEG

> The image panels below show `prediction_examples` from the top-performing runs.


# ============================================================
# SECTION 8: RESEARCH JOURNEY — LEARNING FROM FAILURE
# ============================================================

## Research Journey: Learning from Failure

This project didn't start with the final approach. Three failed tracks provided critical lessons.

### Track 1: Documentation-First (v0x) — Failed
Wrote extensive documentation before any code. Too many ideas added simultaneously. **No executable experiments.**
> **Lesson:** Documentation without experimentation leads to untested assumptions.

### Track 2: Kaggle Notebook Reproduction (vK) — Failed
Found a Kaggle notebook claiming ~89.75% accuracy. Audit revealed **data leakage** (test set = validation set) and **segmentation failure** (model predicted all-zero masks, scoring Dice ~0.58 because 40% of images are authentic). After fixing leakage, best honest F1 = 0.4101.
> **Lesson:** Always audit for data leakage. Training from scratch on small datasets fails.

### Track 3: ETASR Paper Reproduction (vR) — Archived
Reproduced a published paper, achieving 90.23% classification accuracy (vs paper's 96.21% claim). But this was **classification-only** — no localization capability.
> **Lesson:** Verify the base architecture supports all required tasks before ablation.

### Track 4: Pretrained Ablation Study (vR.P) — Final
Combined all lessons: pretrained encoder + UNet segmentation + ELA input + single-variable ablation + W&B tracking. This produced all results shown in this report.


# ============================================================
# SECTION 9: REPRODUCIBILITY
# ============================================================

## Reproducibility

### Fixed Constants Across All Experiments

| Constant | Value |
|----------|-------|
| Dataset | CASIA v2.0 (12,614 images) |
| Split | 70/15/15 stratified (seed 42) |
| Resolution | 384×384 |
| Decoder | UNet (256/128/64/32/16) |
| Batch size | 16 |
| Optimizer | Adam (η=1e-3, wd=1e-5) |
| Framework | PyTorch + SMP |
| Seed | 42 |

### Verification Runs

Two experiments were repeated independently:

| Experiment | Run 1 F1 | Run 2 F1 | Match |
|-----------|---------|---------|-------|
| vR.P.3 (ELA baseline) | 0.6920 | 0.6920 | ✓ Identical |
| vR.P.10 (CBAM) | 0.7277 | 0.7277 | ✓ Identical |

All metrics are computed on **tampered images only** (from P.3 onward) to avoid inflation from all-zero authentic masks.


# ============================================================
# SECTION 10: CONCLUSION
# ============================================================

## Conclusion

Through 60+ controlled experiments across four research tracks, this study established:

1. **Input representation is the dominant factor.** RGB → Multi-Q RGB ELA = +34.19pp Pixel F1. No architecture or training change comes close.

2. **CBAM is the most parameter-efficient improvement.** +3.57pp for only 11K parameters (0.36% of trainable).

3. **Single-variable ablation works.** One change per experiment transforms ad-hoc experimentation into systematic science.

4. **Failures teach.** Data leakage detection, the Flatten-Dense bottleneck diagnosis, and the classification-only limitation discovery all came from failed tracks.

### Impact Hierarchy

```
Input representation  >>  Attention mechanisms  >  Training config  >  Loss function
```

### Links

- **GitHub:** [TIDAL Repository](https://github.com/The-Harsh-Vardhan/TIDAL-Tampered_Image_Detection_And_Localization)
- **Best Model:** vR.P.19 — Pixel F1 = 0.7965, IoU = 0.6615, AUC = 0.9665
