# vK.3 Run Audit

**Notebook:** `vk-3-image-detection-and-localisation-run-01.ipynb`
**Environment:** Kaggle (T4 GPU)
**Epochs:** 50 (full run, no early stopping)

---

## Final Test Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | **0.8986** |
| Test Dice | 0.5760 |
| Test IoU | 0.5528 |
| Test F1 | 0.5757 |
| Best Val Accuracy | 0.8836 (epoch 45) |

---

## Training Observations

- Full 50-epoch run to completion (no early stopping implemented)
- Training loss decreased steadily from ~0.88 to ~0.33
- Val accuracy climbed to ~0.88 and plateaued around epoch 30
- Segmentation metrics (Dice/IoU/F1) showed slow improvement then plateau
- CosineAnnealingLR with T_max=10 caused oscillating LR (5 warm restart cycles)
- Best model selected on **val accuracy** (not localization metric) — suboptimal for localization quality

---

## Architecture and Setup

- **Custom UNetWithClassifier** (same as vK.7.5/vK.10.3 lineage)
- Image size: 256x256
- Batch size: 8
- Adam optimizer (lr=1e-4, no weight decay)
- Combined loss: 1.5 * FocalLoss + 1.0 * (0.5*BCE + 0.5*Dice)
- Gradient clipping: max_norm=1.0
- No AMP
- No checkpoint resume capability

---

## Strengths

1. **Highest classification accuracy** of all runs (0.899)
2. **Full 50-epoch training** — actually converged
3. **Reasonable localization** (Dice=0.576, IoU=0.553) — better than v8's tampered-only F1
4. **Clean execution** — ran end-to-end on Kaggle without crashes
5. **Proper stratified splits** (70/15/15)
6. **4-panel visualization** present (Original, GT, Predicted, Overlay)
7. **W&B integration** for experiment tracking
8. **Structured documentation** with docstrings

---

## Weaknesses

1. **No early stopping** — trained full 50 epochs regardless of convergence
2. **No AMP** — slower training, higher memory usage
3. **No checkpoint resume** — if crashed, restart from scratch
4. **Best model selected on accuracy** not localization metrics — misaligned with assignment focus
5. **Broken prior experiment block** (same data leakage as vK.7.5 — trained on test CSV)
6. **No robustness testing**
7. **No explainability (Grad-CAM)**
8. **No threshold optimization** (fixed 0.5)
9. **No forgery-type breakdown**
10. **No mask-size stratification**
11. **No shortcut learning checks**
12. **No data leakage verification** code
13. **Duplicate training blocks** (prior + effective) with same checkpoint path
14. **Scattered hyperparameters** — no centralized CONFIG
15. **No weight decay** — 31M params with no regularization
16. **Oscillating LR** (T_max=10 with 50 epochs = 5 restarts)

---

## Existing Audit Summary (from `Audit-vK.3-run-01.md`)

The existing audit scored vK.3 at **66/100** with these key findings:
- Prior experiment block has miswired CSVs (data leakage)
- Checkpoint selection on accuracy instead of localization quality
- Degenerate segmentation in early epochs (Dice stuck at 0.5949)
- Segmentation metrics eventually improved after epoch ~10
- Classification head learns faster than segmentation head
- Missing robustness testing, explainability, and threshold optimization

---

## Verdict

The best actual training run in terms of classification (0.899 accuracy). Localization is moderate (Dice=0.576). The notebook proves the custom U-Net architecture can learn when given enough epochs, but it lacks all the engineering refinements (AMP, checkpointing, early stopping, CONFIG) and all advanced evaluation features (robustness, explainability, stratified analysis) that v8 has.

**Score: 65/100** — Decent results, poor engineering, no advanced evaluation.
