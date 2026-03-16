# Experiment Tracking

---

## Why Track Experiments

Experiment tracking is used for:

1. Reproducibility of hyperparameters and thresholds
2. Comparison across runs (augmentation changes, encoder changes, threshold behavior)
3. Persistent artifacts and checkpoints
4. Visibility into long training runs

---

## Tool: Weights & Biases

W&B is **optional**. The notebook runs correctly with `USE_WANDB = False`. Every W&B call is guarded behind the `USE_WANDB` flag.

W&B is integrated at the point of action throughout the notebook (not isolated in a separate post-processing section).

---

## Guarded Setup

```python
USE_WANDB = True  # Set True to enable W&B experiment tracking

if USE_WANDB:
    !pip install -q wandb
    import wandb
    from kaggle_secrets import UserSecretsClient
    wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    wandb.init(
        project='v5.1 Tampered Image Detection & Localization',
        config=CONFIG,
        name=f'unet-resnet34-seed{SEED}-kaggle-v5.1',
        tags=['v5.1', 'casia-v2', 'kaggle'],
    )
```

**Authentication:** Uses Kaggle Secrets API to securely retrieve the W&B API key. No interactive login prompt.

**When disabled:**
- No `wandb` install runs
- No import runs
- No login prompt appears

---

## Logged Metrics

### Per-Epoch Training Logs

| Metric | Key |
|---|---|
| Training loss | `train/loss` |
| Validation loss | `val/loss` |
| Validation Pixel-F1 | `val/pixel_f1` |
| Validation Pixel-IoU | `val/pixel_iou` |
| Encoder LR | `train/lr_encoder` |
| Decoder LR | `train/lr_decoder` |
| Epoch | `epoch` |

### End-of-Run Summary Logs

| Metric | Key |
|---|---|
| Best validation F1 | `best/val_f1` |
| Best epoch | `best/epoch` |
| Best threshold | `best/threshold` |
| Test Pixel-F1 (mixed) | `test/pixel_f1_mixed` |
| Test Pixel-F1 (tampered) | `test/pixel_f1_tampered` |
| Test Pixel-IoU (mixed) | `test/pixel_iou_mixed` |
| Test Image Accuracy | `test/image_accuracy` |
| Test Image AUC-ROC | `test/image_auc_roc` |
| Selected threshold | `test/threshold` |

### Image / Artifact Logs

- `prediction_grid.png` — prediction visualization
- `training_curves.png` — loss and metric curves
- `robustness_chart.png` — degradation comparison bar chart
- `best_model.pt` — uploaded as a model artifact

---

## Fallback Artifacts (W&B Disabled)

When `USE_WANDB = False`, all artifacts are saved locally to `/kaggle/working/`:

| Artifact | Path |
|---|---|
| Best checkpoint | `/kaggle/working/checkpoints/best_model.pt` |
| Resume checkpoint | `/kaggle/working/checkpoints/last_checkpoint.pt` |
| Split manifest | `/kaggle/working/results/split_manifest.json` |
| Results summary | `/kaggle/working/results/results_summary.json` |
| Training curves | `/kaggle/working/plots/training_curves.png` |
| Prediction grid | `/kaggle/working/plots/prediction_grid.png` |
| Grad-CAM analysis | `/kaggle/working/plots/gradcam_analysis.png` |
| Robustness chart | `/kaggle/working/plots/robustness_chart.png` |

These artifacts preserve the same core information even when W&B is disabled.

---

## Run Comparison

When enabled, W&B supports comparison of:
- Augmentation changes
- Encoder changes
- Threshold behavior across runs
- Robustness outcomes

---

## Cleanup

```python
if USE_WANDB:
    wandb.finish()
```

Always finish the run inside the guard to flush metrics and artifacts.
