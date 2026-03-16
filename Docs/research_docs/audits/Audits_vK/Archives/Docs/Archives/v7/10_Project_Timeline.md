# Project Timeline

Three-phase execution plan. Phase 1 must be complete before Phase 2 or Phase 3.

**Reference notebooks:** `tamper_detection_v6.5_kaggle.ipynb`, `tamper_detection_v6.5_colab.ipynb`
**Environment:** Kaggle T4 GPU / Colab T4+

---

## Phase 1 — MVP (Minimum Viable Solution)

Satisfies all core assignment requirements. Must be complete before any other phase.

| # | Task | Document |
|---|---|---|
| 1 | Install dependencies, set seed, verify GPU via `setup_device()` | 08 |
| 2 | Load CASIA dataset (Kaggle mount or Colab API download) | 02 |
| 3 | Discover image–mask pairs with case-insensitive walk, dimension validation | 02 |
| 4 | Validate alignment, log exclusions (unknown type, unreadable, dimension mismatch) | 02 |
| 5 | Binarize masks (threshold > 0) | 02 |
| 6 | Stratified split 70/15/15, persist `split_manifest.json` | 02 |
| 7 | Verify no data leakage (set-intersection assertions) | 02 |
| 8 | Create `TamperingDataset` class | 04 |
| 9 | Define transforms at 384 × 384 (spatial only: flip, rotate90) | 04 |
| 10 | Create config-driven DataLoaders with seeded generator and `worker_init_fn` | 04 |
| 11 | Instantiate model via `setup_model()` with DataParallel support | 03 |
| 12 | Define `BCEDiceLoss` (BCE + Dice, smooth=1.0) | 04 |
| 13 | Configure AdamW with differential LR (encoder 1e-4, decoder 1e-3) | 04 |
| 14 | Implement `train_one_epoch()` with AMP, gradient accumulation (4 steps), clipping (1.0) | 04 |
| 15 | Implement `validate_model()` with threshold-aware early stopping (patience=10) | 04 |
| 16 | Save checkpoints (best, last, periodic) to `/kaggle/working/checkpoints/` | 04 |
| 17 | Implement metric functions (Pixel-F1, Pixel-IoU, Precision, Recall) | 05 |
| 18 | Run threshold sweep on validation set (0.1–0.9, step 0.02) | 05 |
| 19 | Evaluate on test set (mixed + tampered-only + forgery-type breakdown) | 05 |
| 20 | Image-level accuracy and AUC-ROC | 05 |
| 21 | Plot training curves | 07 |
| 22 | Plot F1-vs-threshold sweep | 07 |
| 23 | Generate prediction grid (4-column) with empty-mask guard | 07 |
| 24 | Generate Grad-CAM heatmaps with safety checks | 07 |
| 25 | Create diagnostic overlays (TP/FP/FN) | 07 |
| 26 | Run failure case analysis | 07 |
| 27 | Save `results_summary.json` | 08 |
| 28 | Verify all artifacts exist (final inventory cell) | 08 |

**Decision gate:** Phase 1 is complete when all 28 tasks are done and the notebook runs end-to-end.

---

## Phase 2 — Optimization

Improve beyond MVP. Optional but recommended.

| # | Task | Document |
|---|---|---|
| 1 | Add LR scheduler (CosineAnnealingWarmRestarts) | 04 |
| 2 | Add photometric augmentations beyond the MVP set | 04 |
| 3 | Plot ROC curve | 07 |
| 4 | Enable W&B tracking (`CONFIG['use_wandb'] = True`) | 09 |
| 5 | (Optional) Add ELA as 4th input channel (`in_channels=4`) | 03 |
| 6 | (Optional) Add dual-task classification head | 03 |

**Note:** W&B code is already present in the MVP notebook. Phase 2 refers to actively enabling it, not writing new integration code.

---

## Phase 3 — Bonus Work

Extra credit items. Individually optional and self-contained. Do not require Phase 2.

| # | Task | Document |
|---|---|---|
| 1 | Robustness: JPEG compression (QF 70, 50) | 06 |
| 2 | Robustness: Gaussian noise (light, heavy) | 06 |
| 3 | Robustness: Gaussian blur (5×5) | 06 |
| 4 | Robustness: Resize degradation (0.75×, 0.50×) | 06 |
| 5 | Robustness results table and bar chart | 06, 07 |
| 6 | Mask randomization experiment (shortcut detection) | 13 |
| 7 | Boundary artifact analysis | 13 |
| 8 | Encoder comparison (EfficientNet) | 03 |
| 9 | Probability heatmap visualization | 07 |

---

## Phase Ordering Rules

1. Complete Phase 1 before starting Phase 2 or Phase 3.
2. Phase 2 and Phase 3 are independent of each other.
3. Phase 1 alone satisfies the core assignment requirements.
4. Each phase must leave the notebook in a runnable state.
5. All phases target both Kaggle T4 and Colab T4+ environments.

---

## ELA Channel Convention

If ELA is implemented in Phase 2, it adds a 4th input channel (`in_channels=4`). This is consistent across all documents:
- `01_System_Architecture.md`: ELA as optional preprocessing
- `03_Model_Architecture.md`: `in_channels=4` with ELA
- `11_Research_Alignment.md`: Supported by research paper P7

SRM preprocessing, if explored separately, would use a different channel configuration (e.g., `in_channels=6`). SRM and ELA are independent experimental paths.
