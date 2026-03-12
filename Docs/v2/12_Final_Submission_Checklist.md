# 12 — Final Submission Checklist

## Purpose

Pre-submission verification list. Items are grouped by priority: **MVP required**, **Phase 2 improvements**, and **Phase 3 bonus**.

---

## MVP Required (Phase 1)

### Dataset

- [ ] CASIA v2.0 downloaded inside the notebook
- [ ] Image-mask pairs discovered by filename convention
- [ ] Misaligned pairs detected dynamically and logged (not hardcoded)
- [ ] Masks binarized at threshold 128
- [ ] All-zero masks generated for authentic images
- [ ] Stratified train/val/test split (85/7.5/7.5) with seed=42
- [ ] Split manifest persisted for reproducibility

### Data Pipeline

- [ ] TamperingDataset class with error handling on file reads
- [ ] Images resized to 512x512 (bilinear)
- [ ] Masks resized to 512x512 (nearest-neighbor)
- [ ] ImageNet normalization applied
- [ ] MVP augmentations: HorizontalFlip, VerticalFlip, RandomRotate90
- [ ] DataLoaders: batch_size=4, num_workers=2, pin_memory=True

### Model

- [ ] smp.Unet with encoder_name="resnet34", in_channels=3, classes=1
- [ ] Model runs on T4 without OOM

### Training

- [ ] BCE + Dice loss
- [ ] AdamW with differential LR (encoder 1e-4, decoder 1e-3)
- [ ] AMP enabled
- [ ] Gradient accumulation (4 steps) with flush for partial final window
- [ ] Gradient clipping (max_norm=1.0)
- [ ] Validation after each epoch
- [ ] Early stopping on Pixel-F1 (patience=10)
- [ ] Seed set to 42

### Checkpointing

- [ ] best_model.pt saved on validation F1 improvement
- [ ] last_checkpoint.pt saved every epoch
- [ ] Checkpoints saved to Google Drive
- [ ] Checkpoint includes: model, optimizer, scaler, epoch, best_f1, best_epoch

### Evaluation

- [ ] Threshold selected on validation set
- [ ] Pixel-F1 reported (mean +/- std) — mixed-set and tampered-only
- [ ] Pixel-IoU reported (mean +/- std) — mixed-set and tampered-only
- [ ] Pixel precision and recall reported
- [ ] Image accuracy reported
- [ ] Image AUC-ROC reported
- [ ] All metrics on test set using best checkpoint

### Visualization

- [ ] 4-column grid: original, GT mask, **binary predicted mask**, overlay
- [ ] Grid includes best, median, worst, and authentic samples
- [ ] Training curves: loss, F1, IoU
- [ ] Best epoch annotated on F1 curve

---

## Phase 2 Improvements (Optional)

- [ ] LR scheduler (CosineAnnealingWarmRestarts)
- [ ] Photometric augmentations (brightness/contrast, hue/saturation, noise, JPEG)
- [ ] ROC curve plotted with AUC
- [ ] F1-vs-threshold plot with oracle threshold marked
- [ ] W&B integration (if used)

---

## Phase 3 Bonus (Optional)

- [ ] Robustness evaluation: JPEG QF 50/70
- [ ] Robustness evaluation: Gaussian noise (light/heavy)
- [ ] Robustness evaluation: resize degradation (images only, not masks)
- [ ] Robustness results table
- [ ] Robustness bar chart
- [ ] Forgery type breakdown (splicing vs. copy-move)
- [ ] Encoder comparison (ResNet34 vs. EfficientNet-B0)
- [ ] SRM ablation study

---

## Pre-Submission Verification

1. **Restart and run all** — entire notebook runs end-to-end without errors.
2. **GPU verified** — torch.cuda.get_device_name() shows T4.
3. **Outputs rendered** — all tables and figures display correctly.
4. **Sharing enabled** — notebook link accessible (anyone with link can view).
5. **Weights accessible** — model checkpoint downloadable from Drive.

## Related Documents

- [01_Assignment_Overview.md](01_Assignment_Overview.md) — Deliverables
- [10_Project_Timeline.md](10_Project_Timeline.md) — Phase definitions
