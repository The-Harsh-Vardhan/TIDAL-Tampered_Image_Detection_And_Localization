# Experiment Tracking

---

## Why Experiment Tracking

Manual tracking of hyperparameters, metrics, and model versions across training runs is error-prone and does not scale. Experiment tracking provides:

1. **Reproducibility** — Every run's configuration is logged automatically.
2. **Comparison** — Side-by-side comparison of metrics across runs.
3. **Visibility** — Real-time training curves during long Colab sessions.
4. **Accountability** — Persistent record of what was tried and what worked.

---

## Tool: Weights & Biases (W&B)

W&B is the recommended experiment tracker. It is free for personal/academic use, integrates with PyTorch, and works in Colab.

**Status:** Optional. The notebook must run correctly without W&B installed. All W&B calls are guarded:

```python
USE_WANDB = False  # Set True to enable
try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
```

---

## Setup

```python
!pip install -q wandb

import wandb
wandb.login()  # Uses API key from Colab Secrets or interactive prompt

wandb.init(
    project="tamper-detection",
    config=CONFIG,       # Logs all hyperparameters
    name=f"unet-resnet34-seed{SEED}",
    tags=["mvp", "casia-v2"],
)
```

---

## Logged Metrics

### Per-Epoch (during training)

| Metric | Key | Source |
|---|---|---|
| Training loss | `train/loss` | Mean batch loss per epoch |
| Validation loss | `val/loss` | Validation set |
| Validation Pixel-F1 | `val/pixel_f1` | Primary checkpoint metric |
| Validation Pixel-IoU | `val/pixel_iou` | Secondary metric |
| Learning rate | `train/lr_encoder`, `train/lr_decoder` | Optimizer param groups |
| Epoch | `epoch` | Current epoch number |

```python
if USE_WANDB:
    wandb.log({
        'epoch': epoch + 1,
        'train/loss': avg_train_loss,
        'val/loss': val_loss,
        'val/pixel_f1': val_f1,
        'val/pixel_iou': val_iou,
        'train/lr_encoder': optimizer.param_groups[0]['lr'],
        'train/lr_decoder': optimizer.param_groups[1]['lr'],
    })
```

### End of Training

| Metric | Key |
|---|---|
| Best validation F1 | `best/val_f1` |
| Best epoch | `best/epoch` |
| Test Pixel-F1 (mixed) | `test/pixel_f1_mixed` |
| Test Pixel-F1 (tampered) | `test/pixel_f1_tampered` |
| Test Pixel-IoU (mixed) | `test/pixel_iou_mixed` |
| Test Image Accuracy | `test/image_accuracy` |
| Test Image AUC-ROC | `test/image_auc_roc` |
| Selected threshold | `test/threshold` |

```python
if USE_WANDB:
    wandb.summary.update({
        'best/val_f1': best_f1,
        'best/epoch': best_epoch + 1,
        'test/pixel_f1_mixed': test_results['pixel_f1_mean'],
        'test/pixel_f1_tampered': test_results['tampered_f1_mean'],
        'test/pixel_iou_mixed': test_results['pixel_iou_mean'],
        'test/image_accuracy': test_results['image_accuracy'],
        'test/image_auc_roc': test_results['image_auc_roc'],
        'test/threshold': best_threshold,
    })
```

### Artifacts

```python
if USE_WANDB:
    # Log prediction samples
    wandb.log({"predictions": wandb.Image("prediction_grid.png")})
    wandb.log({"training_curves": wandb.Image("training_curves.png")})

    # Log best model
    artifact = wandb.Artifact('best-model', type='model')
    artifact.add_file(os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    wandb.log_artifact(artifact)
```

---

## Run Comparison

W&B automatically provides:
- **Table view** — Compare metrics across runs sorted by any column.
- **Parallel coordinates** — Visualize hyperparameter–metric relationships.
- **Run diff** — See what changed between two runs.

Use the W&B dashboard to compare:
- MVP vs. Phase 2 augmentations
- Different encoder backbones
- With/without ELA features
- Different learning rates

---

## Hyperparameter Sweeps (Future Work)

W&B Sweeps can automate hyperparameter search. Not implemented in MVP but documented for future use.

```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/pixel_f1', 'goal': 'maximize'},
    'parameters': {
        'encoder_lr': {'min': 1e-5, 'max': 1e-3},
        'decoder_lr': {'min': 1e-4, 'max': 1e-2},
        'batch_size': {'values': [2, 4, 8]},
        'accumulation_steps': {'values': [2, 4, 8]},
    },
}
```

**Status:** Future work. Documented for reference only.

---

## Cleanup

```python
if USE_WANDB:
    wandb.finish()
```

Always call `wandb.finish()` at the end of the notebook to flush logs and close the run.
