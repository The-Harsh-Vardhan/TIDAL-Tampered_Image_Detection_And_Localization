# Experiment Tracking

---

## Overview

W&B (Weights & Biases) is **optional**. The notebook runs correctly with `CONFIG['use_wandb'] = False`. Every W&B call is guarded behind the flag.

W&B is integrated at the point of action throughout the notebook — not isolated in a separate section. This means training metrics are logged in the training loop, evaluation metrics in the evaluation section, and artifact uploads in the save section.

---

## Guarded Setup

```python
if CONFIG['use_wandb']:
    !pip install -q wandb
    import wandb
    from kaggle_secrets import UserSecretsClient
    wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
    wandb.init(
        project='Tampered Image Detection & Localization',
        config=CONFIG,
        name=f'unet-resnet34-seed{SEED}-v6.5',
        tags=['v6.5', 'casia-v2', 'kaggle'],
    )
```

**Kaggle authentication:** Uses Kaggle Secrets API (`UserSecretsClient`) for secure W&B API key retrieval. No interactive login prompt.

**Colab authentication:** Uses `google.colab.userdata.get('WANDB_API_KEY')` with a `getpass` fallback if the secret is not configured.

**When disabled:**
- No `wandb` pip install runs
- No import executes
- No login prompt appears
- All logging calls are skipped
- No external network requests

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

### Image / Artifact Logs

| What | When |
|---|---|
| `prediction_grid.png` | After visualization section |
| `training_curves.png` | After visualization section |
| `robustness_chart.png` | After robustness testing |
| `best_model.pt` | Uploaded as model artifact at end |

---

## W&B Integration Points in Notebook

| Section | What Is Logged |
|---|---|
| 2. Environment Setup | `wandb.init()` with full CONFIG |
| 7. Training Pipeline | Per-epoch train/val metrics, learning rates |
| 8. Evaluation | `wandb.summary.update()` with test metrics |
| 9. Visualization | Plot images via `wandb.Image()` |
| 11. Robustness Testing | Per-degradation F1 values |
| 13. Save Artifacts | Model artifact upload, `wandb.finish()` |

---

## Fallback Artifacts (W&B Disabled)

When `CONFIG['use_wandb'] = False`, all artifacts are saved locally:

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

The core information is preserved regardless of W&B status. W&B adds remote storage, versioning, and interactive dashboards — not essential data.

---

## Cleanup

```python
if CONFIG['use_wandb']:
    wandb.finish()
```

Always called inside the guard to flush pending metrics and artifact uploads.
