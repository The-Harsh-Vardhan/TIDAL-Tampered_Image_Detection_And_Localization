# Project Timeline

Three-phase execution plan. Phase 1 must be complete before Phase 2. Phase 3 items are independently completable once the MVP is stable.

---

## Phase 1 — MVP (Minimum Viable Solution)

Must be complete before any other phase. Satisfies all core assignment requirements.

| # | Task | Document |
|---|---|---|
| 1 | Install dependencies (incl. `kaggle`) | 08 |
| 2 | Set seed, verify GPU | 08 |
| 3 | Download CASIA v2.0 via Kaggle API | 02 |
| 4 | Discover image–mask pairs dynamically | 02 |
| 5 | Validate alignment and log exclusions | 02 |
| 6 | Binarize masks (threshold > 128) | 02 |
| 7 | Stratified split 85/7.5/7.5, persist manifest | 02 |
| 8 | Create `TamperingDataset` class | 04 |
| 9 | Define MVP transforms (spatial only) | 04 |
| 10 | Create DataLoaders | 04 |
| 11 | Sanity check: visualize one batch | 07 |
| 12 | Instantiate `smp.Unet(resnet34)` | 03 |
| 13 | Define `BCEDiceLoss` | 04 |
| 14 | Configure AdamW with differential LR | 04 |
| 15 | Implement training loop with AMP, accumulation, flush | 04 |
| 16 | Implement early stopping on val Pixel-F1 | 04 |
| 17 | Save checkpoints to Drive | 04 |
| 18 | Implement metric functions | 05 |
| 19 | Run threshold sweep on validation set | 05 |
| 20 | Evaluate on test set (mixed + tampered-only) | 05 |
| 21 | Report forgery-type breakdown | 05 |
| 22 | Plot training curves | 07 |
| 23 | Plot F1-vs-threshold | 07 |
| 24 | Generate prediction grid (4-column) | 07 |
| 25 | Save results summary JSON | 08 |

**Decision gate:** Phase 1 is complete when all 25 tasks are done, the notebook runs end-to-end, and test metrics are reported.

---

## Phase 2 — Optimization

Improve beyond MVP. Optional but recommended.

| # | Task | Document |
|---|---|---|
| 1 | Add LR scheduler (CosineAnnealingWarmRestarts) | 04 |
| 2 | Add photometric augmentations | 04 |
| 3 | Plot ROC curve | 07 |
| 4 | Integrate W&B tracking (guarded behind `USE_WANDB`) | 09 |
| 5 | (Optional) Add ELA as 4th input channel (`in_channels=4`) | 03 |

**Decision gate:** Phase 2 is complete when optimization changes are trained, evaluated, and compared to MVP baseline.

---

## Phase 3 — Bonus Work

Extra credit items. These are individually optional and self-contained. Phase 3 tasks do not require Phase 2 to be complete — they can proceed independently once the MVP pipeline is stable and evaluated.

| # | Task | Document | Requires Phase 2? |
|---|---|---|---|
| 1 | Robustness: JPEG compression | 06 | No |
| 2 | Robustness: Gaussian noise | 06 | No |
| 3 | Robustness: Gaussian blur | 06 | No |
| 4 | Robustness: Resize degradation | 06 | No |
| 5 | Robustness results table and bar chart | 06, 07 | No |
| 6 | Encoder comparison (EfficientNet) | 03 | No |
| 7 | SRM ablation study | 03 | No |
| 8 | Probability heatmap visualization | 07 | No |
| 9 | Feature map inspection | 07 | No |

**Decision gate:** Phase 3 items are individually optional. Each should be self-contained and not break the core pipeline.

---

## Phase Ordering Rules

1. Complete Phase 1 before starting Phase 2 or Phase 3.
2. Phase 2 and Phase 3 are independent — either can proceed once MVP is stable.
3. Phase 1 alone satisfies the assignment requirements.
4. Each phase must leave the notebook in a runnable state.

---

## ELA Channel Convention

If ELA is implemented in Phase 2, it adds a 4th input channel to the model (`in_channels=4` instead of 3). This is consistent across all documentation:
- 01_System_Architecture.md: "ELA map concatenated as 4th channel"
- 03_Model_Architecture.md: `in_channels=4` with ELA
- This timeline: Phase 2 task #5

SRM preprocessing, if explored separately, would use a different channel configuration (e.g., `in_channels=6`). SRM and ELA are independent experimental paths — see 03_Model_Architecture.md for details.
