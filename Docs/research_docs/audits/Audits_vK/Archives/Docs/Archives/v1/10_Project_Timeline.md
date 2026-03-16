# 10 — Project Timeline

## Purpose

This document defines the three-stage implementation roadmap. Each stage builds on the previous one. Complete each stage fully before moving to the next.

---

## Stage 1 — MVP (Minimum Viable Solution)

**Goal:** Produce the simplest working solution that satisfies the assignment requirements.

### Steps

| # | Task | Key details |
|---|---|---|
| 1 | Setup Colab environment | Install dependencies, mount Drive, set seed |
| 2 | Download CASIA v2.0 | Kaggle API; unzip to `/content/data/` |
| 3 | Discover and validate pairs | Pair tampered images to masks by filename; exclude 17 misaligned pairs |
| 4 | Binarize masks | Threshold at 128; generate all-zero masks for authentic images |
| 5 | Stratified train/val/test split | 85 / 7.5 / 7.5; stratify by forgery type; seed=42 |
| 6 | Implement `TamperingDataset` | PyTorch Dataset class with on-the-fly loading |
| 7 | Define transforms | Resize to 512×512, HorizontalFlip, VerticalFlip, RandomRotate90, Normalize |
| 8 | Create DataLoaders | batch_size=4, num_workers=2, pin_memory=True |
| 9 | Define U-Net model | SMP U-Net with ResNet34 encoder, in_channels=3, classes=1 |
| 10 | Implement BCE + Dice loss | `BCEDiceLoss` class |
| 11 | Configure optimizer | AdamW with differential LR (encoder: 1e-4, decoder: 1e-3) |
| 12 | Implement training loop | AMP, gradient accumulation (4 steps), gradient clipping |
| 13 | Implement validation | Per-epoch Pixel-F1 and IoU computation |
| 14 | Add checkpointing | Save `last_checkpoint.pt` and `best_model.pt` |
| 15 | Add early stopping | Patience = 10 epochs based on validation F1 |
| 16 | Train the model | Up to 50 epochs |
| 17 | Evaluate on test set | Pixel-F1, IoU, precision, recall, image accuracy, AUC |
| 18 | Visualize predictions | 4-column grid: original, GT mask, heatmap, overlay |
| 19 | Plot training curves | Loss, F1, IoU over epochs |

### Outcome

A working tampering localization model with quantitative results and visual output. This satisfies the core assignment.

---

## Stage 2 — Performance Optimization

**Goal:** Improve training stability, model performance, and evaluation depth.

### Steps

| # | Task | Key details |
|---|---|---|
| 1 | Add augmentations | RandomBrightnessContrast, HueSaturationValue, GaussNoise, ImageCompression |
| 2 | Add LR scheduler | CosineAnnealingWarmRestarts(T_0=10, T_mult=2) |
| 3 | Threshold calibration | Run Oracle-F1 search on validation set |
| 4 | Apply calibrated threshold | Re-evaluate test set with validation-optimal threshold |
| 5 | Add ROC curve | Image-level ROC with AUC |
| 6 | Add F1-vs-threshold plot | Sweep thresholds and plot mean F1 |
| 7 | Add optional edge loss | If boundaries are blurry, add edge loss (weight=0.5) |
| 8 | Optional: W&B logging | Log losses, metrics, and learning rate per epoch |

### Outcome

A better-performing model with richer evaluation and clearer analysis.

---

## Stage 3 — Bonus Improvements

**Goal:** Earn bonus points with robustness testing and optional ablations.

### Steps

| # | Task | Key details |
|---|---|---|
| 1 | Robustness testing | Evaluate trained model under JPEG, noise, and resize degradations |
| 2 | Plot robustness results | Bar chart with F1 per degradation |
| 3 | Forgery type breakdown | Report separate F1 for splicing vs. copy-move |
| 4 | Optional: SRM ablation | Train a second model with RGB+SRM (6 channels) and compare |
| 5 | Optional: encoder comparison | Compare ResNet34 vs. EfficientNet-B0 |

### Outcome

Robustness evaluation table and optional ablation results for bonus credit.

---

## Stage Dependency Chart

```
Stage 1 (MVP)
    ├── Core dataset pipeline
    ├── U-Net baseline model
    ├── Training with BCE + Dice
    ├── Evaluation metrics
    └── Prediction visualization
         │
         ▼
Stage 2 (Optimization)
    ├── Better augmentations
    ├── LR scheduler
    ├── Threshold calibration
    ├── Extended visualizations
    └── Optional W&B
         │
         ▼
Stage 3 (Bonus)
    ├── Robustness evaluation
    ├── Forgery type analysis
    └── Optional SRM / encoder ablation
```

## Decision Rule

- If Stage 1 is incomplete, do not start Stage 2.
- If Stage 2 is incomplete, do not start Stage 3.
- If time is limited, submit Stage 1 alone — it satisfies the assignment.

## Related Documents

- [01_Assignment_Overview.md](01_Assignment_Overview.md) — Requirements and success criteria
- [11_Final_Submission_Checklist.md](11_Final_Submission_Checklist.md) — Pre-submission verification
