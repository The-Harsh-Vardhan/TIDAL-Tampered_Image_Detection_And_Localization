# 10 — Project Timeline

## Purpose

Define the three-phase implementation roadmap. Complete each phase fully before moving to the next.

---

## Phase 1 — MVP (Minimum Viable Solution)

**Goal:** Simplest working solution that satisfies the assignment.

| # | Task | Details |
|---|---|---|
| 1 | Setup | Install deps, set seed, verify GPU |
| 2 | Download CASIA v2.0 | Kaggle API |
| 3 | Discover and validate pairs | Pair by filename; log excluded pairs |
| 4 | Binarize masks | Threshold at 128; all-zero for authentic |
| 5 | Stratified split | 85/7.5/7.5; seed=42; persist manifest |
| 6 | Implement TamperingDataset | On-the-fly loading with error checks |
| 7 | Define MVP transforms | Resize, HorizontalFlip, VerticalFlip, RandomRotate90, Normalize |
| 8 | Create DataLoaders | batch_size=4, num_workers=2 |
| 9 | Define model | smp.Unet(encoder_name="resnet34", in_channels=3, classes=1) |
| 10 | Implement BCE + Dice loss | BCEDiceLoss class |
| 11 | Configure optimizer | AdamW, encoder LR=1e-4, decoder LR=1e-3 |
| 12 | Training loop | AMP, gradient accumulation (4 steps), grad clipping |
| 13 | Validation | Pixel-F1 and IoU per epoch |
| 14 | Checkpointing | best_model.pt and last_checkpoint.pt to Drive |
| 15 | Early stopping | Patience=10 on validation F1 |
| 16 | Select threshold | Sweep on validation set |
| 17 | Evaluate on test set | F1, IoU, precision, recall, image accuracy, AUC |
| 18 | Visualize predictions | 4-column grid: original, GT, binary mask, overlay |
| 19 | Plot training curves | Loss, F1, IoU |

**Phase 1 does NOT include:** LR scheduler, photometric augmentations, W&B, SRM, robustness testing.

**Outcome:** A working tampering localization model with evaluation and visualizations.

---

## Phase 2 — Optimization

**Goal:** Improve performance and analysis depth.

| # | Task | Details |
|---|---|---|
| 1 | Add LR scheduler | CosineAnnealingWarmRestarts |
| 2 | Add photometric augmentations | BrightnessContrast, HueSaturation, GaussNoise, ImageCompression |
| 3 | ROC curve | Image-level ROC with AUC |
| 4 | F1-vs-threshold plot | Sweep and visualize |
| 5 | Optional: W&B logging | Loss, metrics, LR per epoch |

**Outcome:** Better model with richer evaluation.

---

## Phase 3 — Bonus Work

**Goal:** Earn bonus points with robustness testing and optional ablations.

| # | Task | Details |
|---|---|---|
| 1 | Robustness testing | JPEG, noise, resize degradations on test set |
| 2 | Robustness results | Table and bar chart |
| 3 | Forgery type breakdown | Splicing vs. copy-move F1 |
| 4 | Optional: encoder comparison | ResNet34 vs. EfficientNet-B0 |
| 5 | Optional: SRM ablation | RGB-only vs. RGB+SRM comparison |

**Outcome:** Robustness evaluation and optional ablations for bonus credit.

---

## Decision Rules

- If Phase 1 is incomplete, do not start Phase 2.
- If Phase 2 is incomplete, do not start Phase 3.
- Phase 1 alone satisfies the assignment requirements.

## Related Documents

- [01_Assignment_Overview.md](01_Assignment_Overview.md) — Requirements
- [12_Final_Submission_Checklist.md](12_Final_Submission_Checklist.md) — Verification
