# Project Timeline

Three-phase execution plan. Complete each phase before starting the next.

---

## Phase 1 — MVP (Minimum Viable Solution)

Must be complete before Phase 2. Satisfies all core assignment requirements.

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
| 4 | Integrate W&B tracking | 09 |
| 5 | (Optional) Add ELA as 4th input channel | 03 |

**Decision gate:** Phase 2 is complete when optimization changes are trained, evaluated, and compared to MVP baseline.

---

## Phase 3 — Bonus Work

Extra credit items. Optional.

| # | Task | Document |
|---|---|---|
| 1 | Robustness: JPEG compression | 06 |
| 2 | Robustness: Gaussian noise | 06 |
| 3 | Robustness: Gaussian blur | 06 |
| 4 | Robustness: Resize degradation | 06 |
| 5 | Robustness results table and bar chart | 06, 07 |
| 6 | Encoder comparison (EfficientNet) | 03 |
| 7 | SRM ablation study | 03 |
| 8 | Probability heatmap visualization | 07 |

**Decision gate:** Phase 3 items are individually optional. Each should be self-contained and not break the core pipeline.

---

## Phase Ordering Rules

1. Complete Phase 1 before starting Phase 2.
2. Complete Phase 2 before starting Phase 3.
3. Phase 1 alone satisfies the assignment requirements.
4. Each phase must leave the notebook in a runnable state.
