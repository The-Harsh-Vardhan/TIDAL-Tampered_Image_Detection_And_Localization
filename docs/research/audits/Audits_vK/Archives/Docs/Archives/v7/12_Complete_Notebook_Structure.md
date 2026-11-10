# Complete Notebook Structure

This document describes the authoritative training notebooks: `tamper_detection_v6.5_kaggle.ipynb` and `tamper_detection_v6.5_colab.ipynb`.

Both contain **56 cells (14 markdown + 42 code) across 13 sections**.

---

## Section Map

| # | Section | Cells | Purpose |
|---|---:|---|---|
| 1 | Project Overview | 1 | Notebook title, version, changelog |
| 2 | Environment Setup | 2–6 | Install deps, imports, seed, `setup_device()`, CONFIG, artifact dirs, optional W&B |
| 3 | Dataset Loading | 7–11 | Kaggle input discovery (Kaggle) / Drive mount + API download (Colab), directory structure |
| 4 | Dataset Validation | 12–15 | Pair discovery, readability + dimension checks, counts, leakage assertions |
| 5 | Preprocessing | 16–20 | Stratified split, manifest persistence, `TamperingDataset`, transforms, DataLoaders |
| 6 | Model Architecture | 21–23 | `setup_model()` → smp.Unet(resnet34), DataParallel, shape verification |
| 7 | Training Pipeline | 24–28 | BCEDiceLoss, AdamW, `train_one_epoch()`, `validate_model()`, training loop with early stopping |
| 8 | Evaluation | 29–34 | Threshold sweep, test metrics (mixed + tampered-only), image-level ACC/AUC, results summary |
| 9 | Visualization | 35–40 | Training curves, threshold plot, prediction collection, grid with empty-mask guard |
| 10 | Explainable AI | 41–45 | Grad-CAM with safety checks, diagnostic overlays, failure case analysis |
| 11 | Robustness Testing | 46–50 | 8 degradation conditions, robustness bar chart |
| 12 | Experiment Tracking | 51 | W&B integration summary (markdown note) |
| 13 | Save Artifacts | 52–56 | Save results JSON, artifact inventory, optional W&B upload, `wandb.finish()` |

---

## Key Engineering Features

| Feature | Implementation |
|---|---|
| **CONFIG dictionary** | Centralized hyperparameters + feature flags at notebook top |
| **Feature flags** | `use_amp`, `use_multi_gpu`, `use_wandb` — 3 boolean controls |
| **`setup_device()`** | Hardware detection, cuDNN benchmark, TF32, device reporting |
| **`setup_model()`** | Model creation, DataParallel wrapping, shape verification |
| **`train_one_epoch()`** | Modular training with AMP, accumulation, partial window flush |
| **`validate_model()`** | Modular validation with AMP and per-image metrics |
| **`GradScaler(enabled=...)`** | AMP becomes no-op when disabled — zero conditional branching |
| **Config-driven DataLoaders** | `loader_kwargs` dict with auto-detected `pin_memory` and `persistent_workers` |
| **Checkpoint resume** | `last_checkpoint.pt` saves full training state for session recovery |
| **Checkpoint portability** | Always saves unwrapped state_dict; load handles prefix mismatch |

---

## Kaggle vs. Colab Differences

| Aspect | Kaggle | Colab |
|---|---|---|
| Dataset access | Pre-mounted at `/kaggle/input/` | Kaggle API download to Drive |
| Auth for data download | Not needed (pre-mounted) | `userdata.get('KAGGLE_USERNAME')` + `KAGGLE_KEY` with getpass fallback |
| Auth for W&B | `UserSecretsClient().get_secret("WANDB_API_KEY")` | `userdata.get('WANDB_API_KEY')` with getpass fallback |
| Kaggle package | Not needed | `kaggle>=1.6,<1.7` (pinned) with `opendatasets` fallback |
| Output root | `/kaggle/working/` | Google Drive path |
| Section 3 content | Directory discovery under `/kaggle/input/` | Drive mount, Kaggle API auth, download/extract |

All other sections (4–13) are functionally identical.

---

## Key Implementation Details

- **Dataset discovery:** Case-insensitive `os.walk` to locate `IMAGE/` and `MASK/` directories regardless of case or nesting.
- **Pair validation:** Checks image readability, mask readability, dimension agreement; excludes unknown forgery types. Corruption guard for `cv2.imread()` failures.
- **Mask binarization:** `> 0` (any non-zero pixel is foreground). Consistent in `TamperingDataset` and `ResizeDegradationDataset`.
- **Split manifest:** `split_manifest.json` is the reproducibility source of truth. Reloaded on subsequent runs when compatible.
- **Data leakage assertions:** Zero overlap between train/val/test paths verified before training.
- **Threshold-aware validation:** Early stopping uses best val Pixel-F1 from the sweep, not a fixed 0.5 threshold.
- **Top-k mean image score:** Image-level detection uses top 1% of pixel probabilities instead of `max(prob_map)`.
- **True-negative convention:** Pixel metrics are `(1.0, 1.0)` for correct all-zero predictions, not `(0.0, 0.0)`.
- **Grad-CAM safety:** `try/except` around hooks; None-check on activations/gradients; visualization skips gracefully on failure.
- **Prediction grid guard:** Empty predictions are annotated instead of displaying blank images.
- **Partial accumulation flush:** `train_one_epoch()` handles final batch window even when not divisible by `accumulation_steps`.

---

## Public Outputs / Artifacts

All artifacts are written to `/kaggle/working/` (Kaggle) or Drive path (Colab):

| Artifact | Path |
|---|---|
| Best checkpoint | `checkpoints/best_model.pt` |
| Resume checkpoint | `checkpoints/last_checkpoint.pt` |
| Periodic checkpoint | `checkpoints/checkpoint_epoch_N.pt` |
| Split manifest | `results/split_manifest.json` |
| Results summary | `results/results_summary.json` |
| Training curves | `plots/training_curves.png` |
| Threshold sweep plot | `plots/f1_vs_threshold.png` |
| Prediction grid | `plots/prediction_grid.png` |
| Grad-CAM analysis | `plots/gradcam_analysis.png` |
| Robustness chart | `plots/robustness_chart.png` |

---

## Alignment Checklist

| Feature | Docs7 Source | Notebook Section |
|---|---|---|
| Hardware abstraction | 01, 08 | 2 (setup_device), 6 (setup_model) |
| CASIA dataset pipeline | 02 | 3–5 |
| SMP U-Net baseline | 03 | 6 |
| BCEDiceLoss + AdamW | 04 | 7 |
| Modular training functions | 04, 08 | 7 |
| Threshold protocol | 05 | 8 |
| Robustness testing | 06 | 11 |
| Visualization / Explainability | 07 | 9–10 |
| Engineering / reproducibility | 08 | 2, 5, 13 |
| Optional W&B logging | 09 | 2, 7, 8, 9, 11, 13 |
