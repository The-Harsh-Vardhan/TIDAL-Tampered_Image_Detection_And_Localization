# Experiment Tracking

---

## Why Track Experiments

Experiment tracking is used for:

1. reproducibility of hyperparameters and thresholds
2. comparison across runs
3. persistent artifacts and checkpoints
4. visibility into long Colab runs

---

## Tool: Weights & Biases

W&B is optional. The notebook runs correctly with `USE_WANDB = False`, and every W&B call is guarded.

In notebook v5, W&B is integrated at the point of action rather than isolated in a separate post-processing section.

---

## Guarded Setup

```python
USE_WANDB = False

if USE_WANDB:
    !pip install -q wandb
    import wandb
    wandb.login()
    wandb.init(
        project='tamper-detection',
        config=CONFIG,
        name=f'unet-resnet34-seed{SEED}',
        tags=['mvp', 'casia-v2'],
    )
```

When disabled:
- no `wandb` install runs
- no import runs
- no login prompt appears

---

## Logged Metrics

### Per-Epoch Training Logs

| Metric | Key |
|---|---|
| Training loss | `train/loss` |
| Validation loss | `val/loss` |
| Validation Pixel-F1 | `val/pixel_f1` |
| Validation Pixel-IoU | `val/pixel_iou` |
| Validation best threshold | `val/best_threshold` |
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
| Test Pixel Precision / Recall (mixed) | `test/pixel_precision_mixed`, `test/pixel_recall_mixed` |
| Test Pixel Precision / Recall (tampered) | `test/pixel_precision_tampered`, `test/pixel_recall_tampered` |
| Test Image Accuracy | `test/image_accuracy` |
| Test Image AUC-ROC | `test/image_auc_roc` |
| Selected threshold | `test/threshold` |

### Image / Artifact Logs

- `prediction_grid.png`
- `training_curves.png`
- `robustness_chart.png`
- `best_model.pt` as a model artifact

---

## Fallback Artifacts (W&B Disabled)

When `USE_WANDB = False`, the notebook writes artifacts to `CHECKPOINT_DIR`:
- Google Drive in Colab if enabled
- otherwise the local fallback directory

| Artifact | File |
|---|---|
| Best checkpoint | `best_model.pt` |
| Resume checkpoint | `last_checkpoint.pt` |
| Split manifest | `split_manifest.json` |
| Results summary | `results_summary_v5.json` |
| Training curves | `training_curves.png` |
| Prediction grid | `prediction_grid.png` |
| Grad-CAM analysis | `gradcam_analysis.png` |
| Robustness chart | `robustness_chart.png` |

These artifacts preserve the same core information even when W&B is disabled.

---

## Run Comparison

When enabled, W&B supports comparison of:
- augmentation changes
- encoder changes
- threshold behavior across runs
- image-score settings
- robustness outcomes

---

## Cleanup

```python
if USE_WANDB:
    wandb.finish()
```

Always finish the run inside the guard to flush metrics and artifacts.
