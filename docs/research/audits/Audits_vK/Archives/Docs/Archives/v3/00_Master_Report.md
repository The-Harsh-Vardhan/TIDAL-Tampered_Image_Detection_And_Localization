# Docs3 — Master Report

**Revision:** 3 (supersedes Docs 2)
**Date:** March 2026
**Overall score:** Addresses all critical and high-priority findings from Audit 2.

---

## 1. Summary of Changes from Docs 2

| Area | Docs 2 state | Docs 3 change |
|---|---|---|
| Image-level score | Unlocked — `max(prob_map)` with alternatives mentioned | **Locked:** `max(prob_map)` for MVP; alternatives documented in Limitations only |
| Pixel / image threshold | Single argument used for both, docs hinted they could differ | **Locked:** One `pixel_threshold` used everywhere; image-level threshold derived from same sweep |
| Resize robustness | Helper function defined, not wired into eval loop | **Wired:** `ResizeDegradationDataset` wrapper feeds resize-degraded images through standard pipeline |
| `kaggle` install | Omitted from default setup | **Added** to default install command |
| `albumentations` version | Unpinned | **Pinned:** `albumentations>=1.3.1,<2.0` |
| Experiment tracking | Mentioned as optional | **Dedicated section** (09) with W&B integration guidance |
| Discovery function | Printed exclusions, did not return them | **Returns** `(pairs, excluded)` tuple |
| Forgery type guard | Silent default to copy-move | **Logs warning** for unrecognized filename patterns |
| ELA feature | Not discussed | **Added** as optional forensic feature (Phase 2) |
| Blur robustness | Not tested | **Added** GaussianBlur degradation to robustness suite |
| GPU verification | Required T4 specifically | **Relaxed:** "CUDA GPU available; T4 is target" |

---

## 2. Architecture Summary

1. Download CASIA v2.0 via Kaggle API in a single Colab notebook.
2. Discover image–mask pairs dynamically; validate alignment; binarize masks; generate zero masks for authentic images; persist the split manifest.
3. Optionally compute Error Level Analysis (ELA) maps and concatenate with RGB (Phase 2).
4. Train `smp.Unet` with `encoder_name="resnet34"`, `encoder_weights="imagenet"` using BCE + Dice loss, AdamW, AMP, gradient accumulation, checkpointing, and early stopping.
5. Select the operating threshold on the validation set only.
6. Evaluate on the test set with mixed-set and tampered-only localization metrics plus image-level accuracy and AUC.
7. Visualize original image, ground-truth mask, binary predicted mask, and overlay.
8. Run robustness testing (JPEG, noise, blur, resize) only after the core pipeline is complete.
9. Log metrics to W&B (optional; notebook must work without it).

---

## 3. Document Index

| File | Purpose |
|---|---|
| [01_System_Architecture.md](01_System_Architecture.md) | End-to-end system overview and data flow |
| [02_Dataset_and_Preprocessing.md](02_Dataset_and_Preprocessing.md) | Dataset, cleaning, splitting, and validation |
| [03_Model_Architecture.md](03_Model_Architecture.md) | U-Net baseline, encoder, output format, optional features |
| [04_Training_Strategy.md](04_Training_Strategy.md) | Loss, optimizer, training loop, checkpointing |
| [05_Evaluation_Methodology.md](05_Evaluation_Methodology.md) | Metrics, threshold protocol, reporting views |
| [06_Robustness_Testing.md](06_Robustness_Testing.md) | Degradation suite with concrete implementation |
| [07_Visualization_and_Explainability.md](07_Visualization_and_Explainability.md) | Required figures, grid layout, curve specs |
| [08_Engineering_Practices.md](08_Engineering_Practices.md) | Environment, dependencies, reproducibility |
| [09_Experiment_Tracking.md](09_Experiment_Tracking.md) | W&B integration, logged metrics, sweep guidance |
| [10_Project_Timeline.md](10_Project_Timeline.md) | Three-phase roadmap with decision gates |

---

## 4. Remaining Known Limitations

1. **Data split integrity** — CASIA v2.0 has no source-image grouping; related images may leak across splits. Mitigated by stratified splitting and manifest persistence.
2. **Image-level fragility** — `max(prob_map)` is sensitive to single false positives. Acknowledged; top-k mean is a Phase 3 alternative.
3. **Dataset size** — ~5,000 images is small by modern standards. Augmentation and pretrained encoder partially compensate.
4. **Colab session limits** — Free tier has idle timeout. Checkpointing to Drive mitigates.

---

## 5. Relationship to Assignment

Every section in `Assignment.md` is covered:

| Assignment section | Docs 3 coverage |
|---|---|
| 1. Dataset Selection & Preparation | 02, 08 |
| 2. Model Architecture & Learning | 03, 04 |
| 3. Testing & Evaluation | 05, 06 |
| 4. Deliverables & Documentation | 07, 08, 09 |
| Bonus: Robustness | 06 |
| Bonus: Subtle tampering detection | 05 (forgery-type breakdown) |
