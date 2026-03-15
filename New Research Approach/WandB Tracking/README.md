# W&B Tracking — Ablation Study Execution System

Automates re-running 22 ablation experiments across 5 Kaggle accounts with
centralized Weights & Biases tracking.

## Quick Start

### 1. Create Source Notebooks Dataset (once)

Upload all `vR.P.x Image Detection and Localisation.ipynb` files from
`New Research Approach/` as a public Kaggle dataset named **`vrpx-source-notebooks`**.

### 2. Upload Runner Dataset Packages

Each runner has a pre-packaged dataset in `datasets/kaggle_dataset_runner_X/`.
Upload each as a separate Kaggle dataset:

| Account | Upload folder | Dataset name |
|---------|--------------|--------------|
| 1 | `datasets/kaggle_dataset_runner_1/` | `wandb-runner-1-config` |
| 2 | `datasets/kaggle_dataset_runner_2/` | `wandb-runner-2-config` |
| 3 | `datasets/kaggle_dataset_runner_3/` | `wandb-runner-3-config` |
| 4 | `datasets/kaggle_dataset_runner_4/` | `wandb-runner-4-config` |
| 5 | `datasets/kaggle_dataset_runner_5/` | `wandb-runner-5-config` |

### 3. Set Kaggle Secrets (all accounts)

On each Kaggle account, add these secrets under **Settings > Secrets**:

| Key | Value |
|-----|-------|
| `WANDB_API_KEY` | Your W&B API key from [wandb.ai/authorize](https://wandb.ai/authorize) |
| `WAND_USERNAME` | Your W&B username |

### 4. Run Experiments

On each Kaggle account:

1. **Import notebook**: upload `runners/wandb_runner_X.ipynb`
2. **Attach data sources**:
   - `vrpx-source-notebooks` (all source notebooks)
   - `wandb-runner-X-config` (runner config package)
   - `casia-2-0-dataset-for-image-forgery-detection` (CASIA2 dataset)
3. **Settings**: GPU accelerator ON, Internet ON, Persistence: Files only
4. **Run All** — let it execute overnight

### 5. Monitor Dashboard

Watch results appear at:
```
https://wandb.ai/<YOUR_USERNAME>/Tampered Image Detection & Localization
```

Each completed experiment creates a W&B run named after its version (e.g., `vR.P.3`).

### 6. Generate Leaderboard

After all 22 runs complete, run `leaderboard/wandb_leaderboard.ipynb` to:
- Generate Pixel F1 and Image Accuracy leaderboards
- Compare feature sets across experiments
- Export `leaderboard.csv` and `all_experiment_results.csv`

## Runner Distribution

| Runner | Experiments | Est. Time |
|--------|------------|-----------|
| 1 | P.7, P.0, P.1, P.1.5, P.18 | ~2.75h |
| 2 | P.8, P.2, P.3 (r01), P.3 (r02) | ~2.3h |
| 3 | P.12, P.4, P.5, P.6 | ~2.5h |
| 4 | P.11, P.9, P.16, P.17 | ~2.6h |
| 5 | P.10 (r01), P.10 (r02), P.14 (r01), P.14b (r02), P.15 | ~2.7h |

**Total: 22 experiments, ~13h combined GPU time**

## Metrics Tracked

| Metric | Logged At | Description |
|--------|-----------|-------------|
| `pixel_f1` | End of run | **Primary metric** — Test Pixel F1 |
| `pixel_iou` | End of run | Test Pixel IoU |
| `pixel_auc` | End of run | Test Pixel AUC |
| `image_accuracy` | End of run | Test Image Classification Accuracy |
| `train_loss` | Each epoch | Training loss |
| `val_loss` | Each epoch | Validation loss |
| `val_pixel_f1` | Each epoch | Validation Pixel F1 |
| `val_pixel_iou` | Each epoch | Validation Pixel IoU |
