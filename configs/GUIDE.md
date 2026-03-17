# W&B Sweep — Execution Guide

## Overview

This sweep system runs all 30 ablation experiments (vR.P.0 through vR.P.28) and tracks them in a single Weights & Biases project: **`Tampered Image Detection & Localization`**.

All notebooks already have W&B integration built in. The sweep infrastructure orchestrates their execution.

---

## Option A: Run on Kaggle (Recommended)

Kaggle notebooks run in isolated environments, so `wandb agent` cannot persist across sessions. Instead, run each notebook individually — the built-in W&B integration logs everything to the same project.

### Step 1: Set Up W&B API Key on Kaggle

1. Go to [wandb.ai/authorize](https://wandb.ai/authorize) and copy your API key
2. On Kaggle, go to **Account → Secrets**
3. Add a new secret:
   - **Label:** `WANDB_API_KEY`
   - **Value:** your API key
4. Alternatively, when a notebook prompts for `wandb.login()`, paste the key manually

### Step 2: Upload and Run Notebooks

For each experiment (P.0 through P.28):

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Click **File → Import Notebook** and upload the `.ipynb` file from `New Research Approach/`
3. Verify the CASIA v2.0 dataset is attached (should auto-attach via metadata)
4. Under **Settings**:
   - **Accelerator:** GPU T4 x2 (or P100)
   - **Persistence:** Files only
   - **Internet:** On (required for W&B)
5. Click **Run All**

Each notebook automatically:
- Calls `wandb.init(project='Tampered Image Detection & Localization')`
- Each run is named after its version (e.g., `vR.P.3`, `vR.P.15`)
- Logs per-epoch metrics (train_loss, val_loss, val_pixel_f1, val_pixel_iou, lr)
- Logs final test metrics (pixel_f1, pixel_iou, pixel_auc, image_accuracy, etc.)
- Uploads the best model as a W&B artifact
- Calls `wandb.finish()`

### Step 3: Run Order

Some experiments have dependencies. Follow this order:

```
Phase 0 (baselines):     P.0 → P.1 → P.1.5
Phase 1 (core):          P.2, P.3, P.4, P.5, P.6  (can run in parallel)
Phase 1 (post-P.3):      P.7, P.8, P.9, P.10, P.11, P.12, P.13, P.14, P.15
Phase 1 (DCT):           P.16 → P.17
Phase 1 (robustness):    P.18  (requires P.3 checkpoint — download from W&B artifacts)
Phase 2 (extensions):    P.19 through P.28  (all branch from P.3, can run in parallel)
```

### Step 4: View Results

Go to your W&B dashboard:
```
https://wandb.ai/<YOUR_USERNAME>/Tampered Image Detection & Localization
```

All runs from all notebooks appear in the same project. You can:
- **Sort by `pixel_f1`** to see the leaderboard
- **Group by `feature_set`** to compare input representations
- **Create parallel coordinates plots** across input_type, encoder, in_channels vs pixel_f1
- **Compare training curves** across experiments

---

## Option B: Run Locally with Sweep Agent

If you have a local GPU and the CASIA v2.0 dataset downloaded, you can use the automated sweep agent.

### Prerequisites

```bash
pip install wandb segmentation-models-pytorch albumentations opencv-python-headless torch torchvision
wandb login
```

### Step 1: Set Dataset Path

```bash
# Point to the directory containing Au/ and Tp/ subdirectories
export DATASET_ROOT="/path/to/casia-v2-dataset"
```

### Step 2: Generate Modules (one-time)

```bash
cd "New Research Approach/sweep"
python convert_notebooks.py
```

This converts all 30 notebooks into `.py` scripts in `modules/`.

### Step 3: Create and Run Sweep

```bash
# Create the sweep
wandb sweep sweep.yaml
# Output: Created sweep with ID: XXXXXXXX

# Run all experiments sequentially
wandb agent <YOUR_USERNAME>/Tampered Image Detection & Localization/<SWEEP_ID>
```

The agent picks up each experiment in order and runs it to completion.

### Multi-GPU Parallel Execution

To run multiple experiments simultaneously on different GPUs:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 wandb agent <SWEEP_ID>

# Terminal 2
CUDA_VISIBLE_DEVICES=1 wandb agent <SWEEP_ID>
```

Each agent claims the next unfinished experiment from the grid.

### Step 4: Regenerate Modules After Notebook Changes

If you modify any notebook, re-run the converter:

```bash
python convert_notebooks.py   # regenerates all 30 modules
```

---

## W&B Dashboard — What Gets Logged

### Config (per run)
| Field | Example | Description |
|-------|---------|-------------|
| `experiment` | `vrp3` | Experiment ID |
| `version` | `vR.P.3` | Version string |
| `change` | `ELA as Input` | What changed from parent |
| `feature_set` | `ela` | Input feature type |
| `input_type` | `ELA` | Input representation |
| `encoder` | `resnet34` | Backbone encoder |
| `in_channels` | `3` | Input channels |
| `epochs` | `25` | Max training epochs |
| `tta` | `false` | Test-time augmentation |
| `jpeg_aug` | `false` | JPEG compression augmentation |

### Metrics (per run)
| Metric | Logged At | Description |
|--------|-----------|-------------|
| `train_loss` | Each epoch | Training loss |
| `val_loss` | Each epoch | Validation loss |
| `val_pixel_f1` | Each epoch | Validation Pixel F1 |
| `val_pixel_iou` | Each epoch | Validation Pixel IoU |
| `lr` | Each epoch | Learning rate |
| `pixel_f1` | End of run | **Test Pixel F1 (primary metric)** |
| `pixel_iou` | End of run | Test Pixel IoU |
| `pixel_precision` | End of run | Test Pixel Precision |
| `pixel_recall` | End of run | Test Pixel Recall |
| `pixel_auc` | End of run | Test Pixel AUC |
| `image_accuracy` | End of run | Test Image Classification Accuracy |
| `image_macro_f1` | End of run | Test Image Macro F1 |
| `image_roc_auc` | End of run | Test Image ROC-AUC |

### Artifacts
Each run uploads `best_model.pt` as a W&B artifact for reproducibility.

---

## Recommended W&B Dashboard Views

1. **Leaderboard Table**: Columns = experiment, version, pixel_f1, pixel_iou, pixel_auc, image_accuracy. Sort by pixel_f1 descending.

2. **Parallel Coordinates**: Axes = input_type, encoder, in_channels, epochs → pixel_f1. Shows which configurations drive performance.

3. **Training Curves**: Group by feature_set, plot val_pixel_f1 over epochs. Shows convergence patterns.

4. **Bar Chart**: pixel_f1 by experiment, colored by feature_set. Quick visual comparison.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `wandb.login()` prompts for key on Kaggle | Add `WANDB_API_KEY` as a Kaggle secret |
| Dataset not found | Ensure CASIA v2.0 dataset is attached; check **Add Data** in Kaggle sidebar |
| `ModuleNotFoundError: segmentation_models_pytorch` | The `!pip install` line is in cell 2 of each notebook; ensure internet is enabled |
| P.18 fails (checkpoint not found) | P.18 loads P.3's trained model; run P.3 first and make the checkpoint available |
| W&B runs appear in wrong project | All notebooks use `project='Tampered Image Detection & Localization'`; verify in cell 2 |
| `USE_WANDB` is False after init | W&B init failed (usually missing API key or no internet); check the printed error |
