# Project Timeline

Three-phase execution plan. Phase 1 must be complete before Phase 2. Phase 3 items are independently completable once the MVP is stable.

**Reference notebook:** `tamper_detection_v5.1_kaggle.ipynb`
**Environment:** Kaggle T4 GPU

---

## Phase 1 — MVP (Minimum Viable Solution)

Must be complete before any other phase. Satisfies all core assignment requirements.

| # | Task | Document |
|---|---|---|
| 1 | Install dependencies | 08 |
| 2 | Set seed, verify GPU | 08 |
| 3 | Download CASIA Splicing Detection + Localization via Kaggle dataset slug | 02 |
| 4 | Discover image–mask pairs with case-insensitive directory walk and dimension validation | 02 |
| 5 | Validate alignment and log exclusions (unknown-type, unreadable, dimension-mismatch) | 02 |
| 6 | Binarize masks (threshold > 0) | 02 |
| 7 | Stratified split 70/15/15, persist `split_manifest.json` | 02 |
| 8 | Verify no data leakage across splits (assertion block) | 02 |
| 9 | Create `TamperingDataset` class | 04 |
| 10 | Define transforms at 384 × 384 (spatial + photometric) | 04 |
| 11 | Create DataLoaders with seeded generator and `worker_init_fn` | 04 |
| 12 | Sanity check: visualize one batch | 07 |
| 13 | Instantiate `smp.Unet(resnet34, imagenet, in_channels=3, classes=1)` | 03 |
| 14 | Define `BCEDiceLoss` (equal weight, smooth=1.0) | 04 |
| 15 | Configure AdamW with differential LR (encoder 1e-4, decoder 1e-3) | 04 |
| 16 | Implement training loop with AMP, gradient accumulation (4 steps), gradient clipping (1.0) | 04 |
| 17 | Implement threshold-aware early stopping on val Pixel-F1 (patience=10) | 04 |
| 18 | Save checkpoints to `/kaggle/working/checkpoints/` | 04 |
| 19 | Implement metric functions (Pixel-F1, Pixel-IoU, Precision, Recall) | 05 |
| 20 | Run threshold sweep on validation set (0.1–0.9, step 0.02) | 05 |
| 21 | Evaluate on test set (mixed + tampered-only) | 05 |
| 22 | Report forgery-type breakdown | 05 |
| 23 | Plot training curves | 07 |
| 24 | Plot F1-vs-threshold | 07 |
| 25 | Generate prediction grid (4-column) with empty-mask guard | 07 |
| 26 | Generate Grad-CAM heatmaps with safety checks | 07 |
| 27 | Create diagnostic overlays (TP/FP/FN) | 07 |
| 28 | Run failure case analysis | 07 |
| 29 | Save `results_summary.json` to `/kaggle/working/results/` | 08 |

**Decision gate:** Phase 1 is complete when all 29 tasks are done, the notebook runs end-to-end on Kaggle, and test metrics are reported.

---

## Phase 2 — Optimization

Improve beyond MVP. Optional but recommended.

| # | Task | Document |
|---|---|---|
| 1 | Add LR scheduler (CosineAnnealingWarmRestarts) | 04 |
| 2 | Add photometric augmentations beyond the MVP set | 04 |
| 3 | Plot ROC curve | 07 |
| 4 | Actively use W&B tracking (already integrated, just set `USE_WANDB = True`) | 09 |
| 5 | (Optional) Add ELA as 4th input channel (`in_channels=4`) | 03 |
| 6 | (Optional) Add dual-task classification head | 03 |

**Note:** W&B support code is already present in the MVP notebook (guarded behind `USE_WANDB`). Phase 2 refers to actively enabling and using W&B for run comparison, not writing new code.

**Decision gate:** Phase 2 is complete when optimization changes are trained, evaluated, and compared to MVP baseline.

---

## Phase 3 — Bonus Work

Extra credit items. These are individually optional and self-contained. Phase 3 tasks do not require Phase 2 to be complete — they can proceed independently once the MVP pipeline is stable and evaluated.

| # | Task | Document | Requires Phase 2? |
|---|---|---|---|
| 1 | Robustness: JPEG compression (q=70, 50, 30) | 06 | No |
| 2 | Robustness: Gaussian noise (σ=10, 25, 50) | 06 | No |
| 3 | Robustness: Gaussian blur (k=3, 5, 7) | 06 | No |
| 4 | Robustness: Resize degradation (0.5×, 0.25×) | 06 | No |
| 5 | Robustness: Brightness shift (±30, ±60) | 06 | No |
| 6 | Robustness: Contrast scaling (0.5×, 1.5×) | 06 | No |
| 7 | Robustness: Saturation jitter (0.3×, 1.7×) | 06 | No |
| 8 | Robustness: Combined degradation (JPEG 50 + noise σ=15) | 06 | No |
| 9 | Robustness results table and bar chart | 06, 07 | No |
| 10 | Encoder comparison (EfficientNet) | 03 | No |
| 11 | SRM ablation study | 03 | No |
| 12 | Probability heatmap visualization | 07 | No |
| 13 | Feature map inspection | 07 | No |
| 14 | Pixel-level AUC-ROC metric | 05 | No |

**Decision gate:** Phase 3 items are individually optional. Each should be self-contained and not break the core pipeline.

---

## Phase Ordering Rules

1. Complete Phase 1 before starting Phase 2 or Phase 3.
2. Phase 2 and Phase 3 are independent — either can proceed once MVP is stable.
3. Phase 1 alone satisfies the assignment requirements.
4. Each phase must leave the notebook in a runnable state.
5. All phases target the Kaggle T4 environment. No Colab-specific operations.

---

## ELA Channel Convention

If ELA is implemented in Phase 2, it adds a 4th input channel to the model (`in_channels=4` instead of 3). This is consistent across all documentation:
- 01_System_Architecture.md: "ELA map concatenated as 4th channel"
- 03_Model_Architecture.md: `in_channels=4` with ELA
- This timeline: Phase 2 task #5

SRM preprocessing, if explored separately, would use a different channel configuration (e.g., `in_channels=6`). SRM and ELA are independent experimental paths — see 03_Model_Architecture.md for details.
