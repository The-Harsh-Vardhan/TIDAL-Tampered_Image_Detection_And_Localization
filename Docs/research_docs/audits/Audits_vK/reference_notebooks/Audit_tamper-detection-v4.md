# Audit: Tamper Detection v4

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `tamper_detection_v4.ipynb` (1.7 MB)

---

## Notebook Overview

The first notebook in the project's own lineage to use **SMP (Segmentation Models PyTorch)** with a pretrained ResNet34 encoder. This is the architectural prototype for all subsequent v6.5+ versions. However, it was only **partially executed** — training ran for 1 epoch, and evaluation/robustness sections were never run.

| Attribute | Value |
|---|---|
| Cell Count | 53 (39 code, 14 markdown) |
| Model | SMP UNet + ResNet34 (ImageNet pretrained) |
| Dataset | CASIA v2.0 (12,614 images) |
| Task | Pixel-level segmentation |
| Image Size | **512×512** |
| Execution | **Partial** — 1 epoch only, evaluation sections empty |

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CASIA v2.0 (Splicing Detection + Localization) |
| Tampered | 5,123 |
| Authentic | 7,491 |
| Total | 12,614 |
| Split | **85/7.5/7.5** (train_ratio=0.85) |
| Image Size | 512×512 |
| Input Channels | 3 (RGB only — no ELA) |

**Non-standard split ratio.** Uses 85/7.5/7.5 instead of the later 70/15/15 standard. This gives the largest training set but leaves very small validation and test sets.

**Augmentation (spatial only):**

| Transform | Parameters |
|---|---|
| Resize | 512×512 |
| HorizontalFlip | p=0.5 |
| VerticalFlip | p=0.5 |
| RandomRotate90 | p=0.5 |
| Normalize | ImageNet mean/std |

---

## Model Architecture Review

| Attribute | Value |
|---|---|
| Decoder | `smp.Unet` |
| Encoder | `resnet34` |
| Encoder Weights | `imagenet` (pretrained) |
| Input Channels | 3 (RGB) |
| Output Classes | 1 (binary mask) |
| Output | Raw logits, sigmoid applied during inference |

This is the **first SMP-based architecture** in the project lineage. Key design choices that carried forward to v6.5 and v8:
- Pretrained ResNet34 encoder (vs custom from-scratch UNet in vK.10.x)
- AdamW with differential LR (encoder: 1e-4, decoder: 1e-3)
- Gradient accumulation (4 steps → effective batch 16)

---

## Training Pipeline Review

| Component | Configuration |
|---|---|
| Optimizer | AdamW (encoder_lr=1e-4, decoder_lr=1e-3, wd=1e-4) |
| Loss | BCEDiceLoss (BCE + Dice, equal weight, smooth=1.0) |
| AMP | Enabled |
| Max Epochs | 50 |
| Patience | 10 |
| Batch Size | 4 (effective 16 via accumulation) |
| Gradient Accumulation | 4 steps |
| Max Grad Norm | 1.0 |
| LR Scheduler | **None** |
| pos_weight | **No** |

**Only 1 epoch was trained:**

| Epoch | Train Loss | Val Loss | Val F1 | Val IoU |
|---|---|---|---|---|
| 1 | 1.0415 | 1.0001 | **1.0000** | **1.0000** |

---

## Evaluation Metrics Review

| Metric | Value | Assessment |
|---|---|---|
| Val F1 (epoch 1) | **1.0000** | **BUG — impossible** |
| Val IoU (epoch 1) | **1.0000** | **BUG — impossible** |

**The F1=1.0 and IoU=1.0 at epoch 1 are a metric computation bug.** The validation function likely computes metrics over all images (including authentic), and for authentic images with all-zero masks, both prediction (near-zero after sigmoid) and ground truth are zero, causing the metric function to return 1.0 for each authentic sample. The metric is then dominated by the authentic majority.

**No threshold selection, evaluation, robustness, or save sections produced output** — their cells are empty, indicating the notebook was not run to completion.

---

## Visualization Assessment

No visualizations were produced. All visualization cells have empty outputs.

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| Architecture Choice | **Excellent** | SMP UNet + pretrained ResNet34 — the right approach |
| Differential LR | **Excellent** | Encoder 1e-4 / Decoder 1e-3 — protects pretrained features |
| Gradient Accumulation | **Good** | 4 steps for effective batch 16 |
| AMP | **Good** | Enabled for mixed precision |
| W&B Integration | **Good** | Guarded behind `USE_WANDB` flag |
| Execution | **Incomplete** | Only 1 epoch, evaluation never run |
| LR Scheduler | **Missing** | No schedule configured |
| pos_weight | **Missing** | No handling for class imbalance in masks |
| Metric Bug | **Critical** | F1=1.0 at epoch 1 |

---

## Strengths

1. **First to use SMP + pretrained encoder** — established the architecture that proved superior to from-scratch training
2. **Differential learning rates** — encoder_lr < decoder_lr protects ImageNet features
3. **Gradient accumulation** — enables effective batch size > physical batch size
4. **BCEDiceLoss** — combines pixel-level BCE with overlap-based Dice
5. **Clean CONFIG system** — all hyperparameters centralized
6. **512×512 resolution** — higher than later versions (384 in v6.5/v8, 256 in vK.10.x)

---

## Weaknesses

1. **Never fully executed** — only 1 training epoch, no evaluation results
2. **Metric computation bug** — F1=1.0 at epoch 1 is impossible and indicates flawed metric code
3. **No LR scheduler** — fixed learning rate throughout training
4. **No pos_weight** — binary masks are heavily class-imbalanced (most pixels are background)
5. **85/7.5/7.5 split** — non-standard, leaves very small validation/test sets
6. **No photometric augmentation** — only spatial transforms (flip, rotate)
7. **Image size 512** — may cause OOM on limited GPUs; later versions reduced to 384

---

## Critical Issues

1. **F1=1.0000 at Epoch 1.** The `compute_pixel_f1` function likely returns 1.0 for images where both prediction and ground truth are all-zero (authentic images). Since authentic images are the majority (~60%), the mean F1 is pulled to 1.0. The fix is to compute F1 only on tampered images (tampered-only metrics), which is what later versions (vK.10.6, vK.11.x) correctly implement.

2. **Partial execution.** The notebook demonstrates the right architecture but was never trained to convergence. Without results, it's impossible to know if this configuration would outperform v6.5 or v8. The 512×512 resolution in particular would have been valuable to test.

3. **No LR scheduler.** Without ReduceLROnPlateau or similar, the fixed LR may be too high for fine-tuning the pretrained encoder and too low for the decoder after initial epochs.

---

## Suggested Improvements

1. Fix the metric computation to report tampered-only F1/IoU
2. Add ReduceLROnPlateau or CosineAnnealing scheduler
3. Add pos_weight to BCE loss component
4. Run the full training + evaluation pipeline to completion
5. Consider reducing to 384×384 if 512 causes OOM
6. Add photometric augmentations (brightness, contrast, JPEG compression, noise)
7. Change split to 70/15/15 for larger validation/test sets

---

## Roast Section

Tamper Detection v4 is the blueprint that never became a building. It makes all the right architectural decisions — pretrained ResNet34, differential learning rates, gradient accumulation, AMP — and then runs for exactly one epoch before stopping. It's like writing a perfect recipe and then only preheating the oven.

The F1=1.0 at epoch 1 is the smoking gun. No model in the history of deep learning has achieved perfect pixel-level F1 on a tampered image dataset after one epoch of training. What happened is that the metric function gives perfect scores to authentic images (where both prediction and ground truth are zero), and since authentic images dominate the dataset, the mean F1 rounds up to 1.0. The notebook proudly logs "New best model (F1=1.0000)" and presumably the author thought "great, it's already perfect" — when in reality, the tampered-only F1 would be near zero.

This is the Goldilocks notebook of the reference collection: it's not too simple (like the ELA+RPA heuristic), not too complex (like the vK.10.x from-scratch UNet), but it was never run long enough to prove it's just right. The 512×512 resolution is actually an interesting choice that later versions abandoned for 384 (v6.5/v8) and 256 (vK.11.x) — we'll never know if the extra resolution would have helped.

**Bottom line:** The best architecture in the reference collection, let down by a metric bug that masked the need for further training and a run that ended after 1 epoch. This is the prototype that v6.5 and v8 successfully built upon — proving that the architecture was right, even if this specific notebook never proved it.
