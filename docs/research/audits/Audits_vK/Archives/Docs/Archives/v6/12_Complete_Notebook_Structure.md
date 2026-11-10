# Complete Notebook Structure

This document describes the authoritative training notebook, `tamper_detection_v5.1_kaggle.ipynb`. It contains **61 cells (18 markdown + 43 code) across 17 sections**.

W&B experiment tracking is integrated throughout the notebook. All W&B calls are guarded behind `USE_WANDB`.

**Environment:** Kaggle T4 GPU, 15 GB VRAM

---

## Section Map

| Section | Cells | Purpose |
|---|---:|---|
| 0. Title / Changelog | 0 | Notebook summary, v5 → v5.1 changes |
| 1. Setup & Environment | 1–7 | Install deps, imports, seed, GPU check, CONFIG, artifact dirs at `/kaggle/working/`, W&B setup via Kaggle Secrets |
| 2. Kaggle Dataset Download | 8–9 | Kaggle API auth, slug-specific cache dir, download/extract, dataset root resolution |
| 3. Dataset Discovery | 10–11 | Case-insensitive `os.walk` for IMAGE/MASK directories, readability checks, dimension checks, dynamic pair discovery, unknown-type exclusion, corruption guard for tampered images |
| 4. Dataset Validation | 12–13 | Dataset counts, sample-load verification, data leakage assertion block |
| 5. Preprocessing & Data Split | 14–16 | Stratified 70/15/15 split, split-manifest reuse, manifest persistence |
| 6. Dataset Class | 17–18 | `TamperingDataset` implementation with mask binarization > 0 |
| 7. DataLoaders | 19–22 | Transforms at 384 × 384, deterministic `DataLoader` setup with seeded generator + `worker_init_fn`, batch sanity check |
| 8. Model Definition | 23–24 | `smp.Unet(resnet34, imagenet, in_channels=3, classes=1)` instantiation and shape check |
| 9. Loss Function & Optimizer | 25–28 | `BCEDiceLoss` (equal weight, smooth=1.0), AdamW with differential LR (encoder 1e-4, decoder 1e-3), metrics, image-score helper (top-k mean) |
| 10. Training Loop | 29–32 | Checkpoints to `/kaggle/working/checkpoints/`, threshold-aware validation, AMP training with gradient accumulation (4 steps, effective batch 16), gradient clipping (max_norm=1.0), early stopping (patience=10) |
| 11. Threshold Selection | 33–35 | Reload best checkpoint, recompute validation sweep (0.1–0.9, step 0.02), confirm best threshold |
| 12. Evaluation on Test Set | 36–39 | Mixed + tampered-only metrics, image-level accuracy/AUC, true-negative consistency (precision=1.0, recall=1.0 for all-zero masks), W&B summary logging |
| 13. Visualization | 40–45 | Training curves, threshold plot, prediction collection, grid with empty-mask guard, W&B images |
| 14. Explainable AI | 46–50 | Grad-CAM with activation safety checks and zero-guard, diagnostic overlays, failure-case analysis |
| 15. Robustness Testing | 51–55 | 8 degradation conditions at 384 resolution, robustness bar chart |
| 16. Experiment Tracking | 56 | Summary of integrated W&B touchpoints |
| 17. Save & Export Results | 57–60 | Save `results_summary.json`, upload model artifact, `wandb.finish()`, final print message |

---

## Key Implementation Details

- **Dataset discovery** uses case-insensitive `os.walk` to locate `IMAGE/` and `MASK/` directories regardless of case or nesting depth.
- Discovery validates tampered image readability, mask readability, image–mask dimensions, and **excludes** `unknown_forgery_type`. Tampered images that fail `cv2.imread` are skipped with a corruption guard.
- **Mask binarization** uses `> 0` (any non-zero pixel is foreground). This is more lenient than the v5 threshold of `> 128` and avoids losing faint mask regions.
- The split manifest is the reproducibility source of truth: the notebook **reloads** `split_manifest.json` on reruns when compatible.
- A **data leakage assertion block** verifies zero overlap between train/val/test image paths before training starts.
- `CONFIG['train_ratio'] = 0.70` drives the 70/15/15 split.
- `DataLoader` construction uses a seeded `torch.Generator` plus `worker_init_fn` for deterministic shuffling and worker RNG state.
- Validation is **threshold-aware during training**: checkpoint selection and early stopping use the best validation Pixel-F1 from the sweep, not a fixed `0.5` threshold.
- Image-level detection uses a **top-k mean tamper score** (top 1% of pixel probabilities) instead of `max(prob_map)`.
- Mixed-set pixel Precision/Recall are reported as **global pixel metrics** accumulated across the split; tampered-only Precision/Recall are reported separately.
- For **true-negative images** (authentic, all-zero ground truth), pixel precision and recall are reported as `(1.0, 1.0)` — not `(0.0, 0.0)` — because a correct all-zero prediction has no false positives or false negatives.
- **Grad-CAM** includes activation safety checks: if the target layer produces zero activations or gradients, the notebook skips the visualization rather than crashing.
- **Prediction grid** includes an empty-mask guard: if a prediction is all zeros after thresholding, the grid annotates it accordingly instead of displaying a blank image.

---

## Public Outputs / Artifacts

All artifacts are written to `/kaggle/working/`:

| Artifact | Path |
|---|---|
| Best checkpoint | `/kaggle/working/checkpoints/best_model.pt` |
| Resume checkpoint | `/kaggle/working/checkpoints/last_checkpoint.pt` |
| Split manifest | `/kaggle/working/results/split_manifest.json` |
| Results summary | `/kaggle/working/results/results_summary.json` |
| Training curves | `/kaggle/working/plots/training_curves.png` |
| Threshold sweep plot | `/kaggle/working/plots/f1_vs_threshold.png` |
| Prediction grid | `/kaggle/working/plots/prediction_grid.png` |
| Grad-CAM analysis | `/kaggle/working/plots/gradcam_analysis.png` |
| Robustness chart | `/kaggle/working/plots/robustness_chart.png` |

---

## Alignment Checklist

| Feature | Docs6 Source | Notebook Section |
|---|---|---|
| CASIA localization dataset pipeline | 02_Dataset_and_Preprocessing.md | 2–7 |
| SMP U-Net baseline | 03_Model_Architecture.md | 8 |
| BCE + Dice and AdamW | 04_Training_Strategy.md | 9–10 |
| Threshold protocol | 05_Evaluation_Methodology.md | 10–12 |
| Robustness testing | 06_Robustness_Testing.md | 15 |
| Visualization / Explainability | 07_Visualization_and_Explainability.md | 13–14 |
| Engineering / reproducibility | 08_Engineering_Practices.md | 1–7, 17 |
| Optional W&B logging | 09_Experiment_Tracking.md | 1, 10, 12, 13, 15, 17 |
