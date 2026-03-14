# 11 — Final Submission Checklist

## Purpose

Use this checklist to verify the notebook is complete before submission.

---

## Core Requirements

### Dataset

- [ ] CASIA v2.0 downloaded and extracted inside the notebook
- [ ] Image-mask pairs correctly discovered by filename convention
- [ ] 17 misaligned pairs excluded with validation check
- [ ] Masks binarized at threshold 128
- [ ] All-zero masks generated for authentic images
- [ ] Stratified train/val/test split (85 / 7.5 / 7.5) with seed=42
- [ ] Split preserves authentic / splicing / copy-move ratios

### Data Pipeline

- [ ] `TamperingDataset` class implements `__getitem__` and `__len__`
- [ ] Images resized to 512×512 (bilinear interpolation)
- [ ] Masks resized to 512×512 (nearest-neighbor interpolation)
- [ ] ImageNet normalization applied to images
- [ ] Augmentations applied synchronously via `albumentations`
- [ ] DataLoaders configured with batch_size=4, num_workers=2, pin_memory=True

### Model

- [ ] U-Net from `segmentation_models_pytorch` with pretrained encoder
- [ ] Output is raw logits (no activation in model)
- [ ] Model instantiates and runs without error on T4

### Training

- [ ] BCE + Dice hybrid loss implemented
- [ ] AdamW optimizer with differential learning rates
- [ ] Mixed precision training (AMP) enabled
- [ ] Gradient accumulation (4 steps, effective batch=16)
- [ ] Gradient clipping (max_norm=1.0)
- [ ] Validation runs after each epoch
- [ ] Early stopping on validation Pixel-F1 (patience=10)
- [ ] Random seed set (42) for reproducibility

### Checkpointing

- [ ] `best_model.pt` saved when validation F1 improves
- [ ] `last_checkpoint.pt` saved every epoch
- [ ] Checkpoints saved to Google Drive (persist beyond session)
- [ ] Checkpoint contains model state, optimizer state, scheduler state, scaler state, metrics

### Evaluation

- [ ] Pixel-F1 computed and reported (mean ± std)
- [ ] Pixel-IoU computed and reported (mean ± std)
- [ ] Pixel precision and recall reported
- [ ] Image-level accuracy reported
- [ ] Image-level AUC-ROC reported
- [ ] All metrics computed on the **test set** using the best checkpoint
- [ ] Threshold selected on validation set, not test set

### Visualization

- [ ] 4-column prediction grid: original, GT mask, heatmap, overlay
- [ ] Grid includes best, median, worst, and authentic samples
- [ ] Training curves: loss, F1, IoU across epochs
- [ ] Best epoch annotated on F1 curve
- [ ] Figures saved as PNG files

---

## Stage 2 Additions

- [ ] Additional augmentations (brightness/contrast, hue/saturation, noise, JPEG)
- [ ] Learning rate scheduler (CosineAnnealingWarmRestarts)
- [ ] Threshold calibrated on validation set
- [ ] ROC curve plotted with AUC value
- [ ] F1-vs-threshold plot with oracle threshold marked

---

## Stage 3 Additions (Bonus)

- [ ] Robustness evaluation under JPEG compression (QF 50, 70)
- [ ] Robustness evaluation under Gaussian noise
- [ ] Robustness evaluation under resize degradation
- [ ] Robustness results table with F1 and delta from baseline
- [ ] Robustness bar chart visualization

### Optional Bonus

- [ ] Forgery type breakdown (splicing vs. copy-move F1)
- [ ] SRM ablation study (RGB-only vs. RGB+SRM)
- [ ] Encoder comparison (ResNet34 vs. EfficientNet-B0)

---

## Submission Artifacts

| Artifact | Location | Format |
|---|---|---|
| Colab notebook | Shared link (view access) | `.ipynb` |
| Best model weights | Google Drive | `best_model.pt` |
| Prediction figures | Saved in notebook outputs | PNG |
| Training curves | Saved in notebook outputs | PNG |

---

## Pre-Submission Verification

1. **Restart and run all** — Confirm the entire notebook runs end-to-end without errors.
2. **Check GPU** — Verify the notebook uses T4 GPU (`torch.cuda.get_device_name()`).
3. **Check outputs** — All evaluation tables and figures render correctly.
4. **Check sharing** — Notebook link is accessible (anyone with the link can view).
5. **Check weights** — Model checkpoint is accessible on Google Drive.

## Related Documents

- [01_Assignment_Overview.md](01_Assignment_Overview.md) — Deliverables and success criteria
- [10_Project_Timeline.md](10_Project_Timeline.md) — Implementation stages
