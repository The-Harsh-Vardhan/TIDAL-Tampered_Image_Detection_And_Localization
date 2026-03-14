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

**Status:** Optional. The notebook runs correctly without W&B. All W&B calls are guarded behind `USE_WANDB`.

---

## Guarded Setup

W&B is controlled by a single flag in the configuration cell. The notebook never installs, imports, or initializes W&B unless the flag is explicitly set to `True`.

```python
# ── Configuration cell ──
USE_WANDB = False  # Set True to enable W&B experiment tracking

# ── W&B setup (runs only when enabled) ──
if USE_WANDB:
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

When `USE_WANDB = False`:
- No `pip install wandb` runs.
- No `import wandb` runs.
- No login prompt appears.
- The notebook relies on local artifacts only (see Fallback Artifacts below).

---

## Logged Metrics (when enabled)

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

    # Log best model as versioned artifact
    artifact = wandb.Artifact('best-model', type='model')
    artifact.add_file(os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    wandb.log_artifact(artifact)
```

---

## Fallback Artifacts (when W&B is disabled)

When `USE_WANDB = False`, the notebook still produces these local artifacts on Google Drive:

| Artifact | Path | Content |
|---|---|---|
| Best model checkpoint | `CHECKPOINT_DIR/best_model.pt` | Model weights at best val Pixel-F1 |
| Last checkpoint | `CHECKPOINT_DIR/last_checkpoint.pt` | Model + optimizer + epoch state |
| Results summary | `CHECKPOINT_DIR/results_summary.json` | All test metrics, threshold, config |
| Split manifest | `CHECKPOINT_DIR/split_manifest.json` | Train/val/test file lists for reproducibility |
| Training curves | Notebook inline | matplotlib plots saved in cell output |
| Prediction grid | Notebook inline | matplotlib grid saved in cell output |

These artifacts ensure full reproducibility even without W&B. The `results_summary.json` contains the same metrics that would otherwise be logged to W&B.

---

## Run Comparison (when W&B is enabled)

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

W&B Sweeps can automate hyperparameter search. Not implemented in MVP. Documented for future reference only.

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

**Status:** Future work. Not part of the MVP or Phase 2 path.

---

## Cleanup

```python
if USE_WANDB:
    wandb.finish()
```

Always call `wandb.finish()` inside the guard at the end of the notebook to flush logs and close the run.
