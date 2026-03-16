# Complete Notebook Structure

This document describes the current final training notebook, [tamper_detection_v5.ipynb](/c:/D%20Drive/Projects/BigVision%20Assignment/notebooks/tamper_detection_v5.ipynb). It contains **61 cells across 17 sections**.

W&B experiment tracking is integrated throughout the notebook. All W&B calls are guarded behind `USE_WANDB`.

---

## Section Map

| Section | Cells | Purpose |
|---|---:|---|
| 0. Title / Changelog | 0 | Notebook summary and v4 -> v5 changes |
| 1. Setup & Environment | 1-7 | Install deps, imports, seed, GPU check, config, Drive/local artifact dir, W&B setup |
| 2. Kaggle Dataset Download | 8-9 | Kaggle auth, slug-specific cache dir, download/extract, dataset root resolution |
| 3. Dataset Discovery | 10-11 | Readability checks, dimension checks, dynamic pair discovery, unknown-type exclusion |
| 4. Dataset Validation | 13-14 | Dataset counts, sample-load verification |
| 5. Preprocessing & Data Split | 15-17 | Stratified split, split-manifest reuse, manifest persistence |
| 6. Dataset Class | 18-19 | `TamperingDataset` implementation |
| 7. DataLoaders | 20-23 | Transforms, deterministic `DataLoader` setup, batch sanity check |
| 8. Model Definition | 24-25 | `smp.Unet(resnet34)` instantiation and shape check |
| 9. Loss Function & Optimizer | 26-29 | BCE+Dice loss, AdamW, metrics, image-score helper |
| 10. Training Loop | 30-33 | Checkpoints, threshold-aware validation, AMP training, early stopping |
| 11. Threshold Selection | 34-36 | Reload best checkpoint, recompute validation sweep, confirm best threshold |
| 12. Evaluation on Test Set | 37-40 | Mixed/tampered metrics, image-level accuracy/AUC, W&B summary logging |
| 13. Visualization | 41-46 | Training curves, threshold plot, prediction collection, grid, W&B images |
| 14. Explainable AI | 47-51 | Grad-CAM, diagnostic overlays, failure-case analysis |
| 15. Robustness Testing | 52-56 | JPEG/noise/blur/resize evaluation and robustness chart |
| 16. Experiment Tracking | 57 | Summary of integrated W&B touchpoints |
| 17. Save & Export Results | 58-60 | Save `results_summary_v5.json`, upload model artifact, finish run |

---

## Key Implementation Details

- Dataset download uses a **slug-specific cache directory** instead of scanning all of `/content`.
- Discovery validates tampered image readability, mask readability, image-mask dimensions, and **excludes** `unknown_forgery_type`.
- The split manifest is the reproducibility source of truth: the notebook **reloads** `split_manifest.json` on reruns when compatible.
- `CONFIG['train_ratio']` is used directly for the first split stage.
- `DataLoader` construction uses a seeded `torch.Generator` plus `worker_init_fn` for deterministic shuffling and worker RNG state.
- Validation is **threshold-aware during training**: checkpoint selection and early stopping use the best validation Pixel-F1 from the sweep, not a fixed `0.5` threshold.
- Image-level detection uses a **top-k mean tamper score** instead of `max(prob_map)`.
- Mixed-set pixel Precision/Recall are reported as **global pixel metrics** accumulated across the split; tampered-only Precision/Recall are reported separately.

---

## Public Outputs / Artifacts

| Artifact | Path / Name |
|---|---|
| Best checkpoint | `best_model.pt` |
| Resume checkpoint | `last_checkpoint.pt` |
| Split manifest | `split_manifest.json` |
| Results summary | `results_summary_v5.json` |
| Training curves | `training_curves.png` |
| Threshold sweep plot | `f1_vs_threshold.png` |
| Prediction grid | `prediction_grid.png` |
| Grad-CAM analysis | `gradcam_analysis.png` |
| Robustness chart | `robustness_chart.png` |

---

## Alignment Checklist

| Feature | Docs5 Source | Notebook Section |
|---|---|---|
| CASIA localization dataset pipeline | [02_Dataset_and_Preprocessing.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/02_Dataset_and_Preprocessing.md) | 2-7 |
| SMP U-Net baseline | [03_Model_Architecture.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/03_Model_Architecture.md) | 8 |
| BCE + Dice and AdamW | [04_Training_Strategy.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/04_Training_Strategy.md) | 9-10 |
| Threshold protocol | [05_Evaluation_Methodology.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/05_Evaluation_Methodology.md) | 10-12 |
| Robustness testing | [06_Robustness_Testing.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/06_Robustness_Testing.md) | 15 |
| Visualization / Explainability | [07_Visualization_and_Explainability.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/07_Visualization_and_Explainability.md) | 13-14 |
| Engineering / reproducibility | [08_Engineering_Practices.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/08_Engineering_Practices.md) | 1-7, 17 |
| Optional W&B logging | [09_Experiment_Tracking.md](/c:/D%20Drive/Projects/BigVision%20Assignment/Docs5/09_Experiment_Tracking.md) | 1, 10, 12, 13, 15, 17 |
