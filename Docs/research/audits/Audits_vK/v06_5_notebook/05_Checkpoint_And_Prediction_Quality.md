# Audit 6.5 — Part 5: Checkpoint & Prediction Quality

## Checkpoint Strategy

### Saved Checkpoints

| Checkpoint | Location | Trigger |
|---|---|---|
| `best_model.pt` | `/kaggle/working/checkpoints/` | Val F1 improvement |
| `last_checkpoint.pt` | `/kaggle/working/checkpoints/` | Every epoch |
| `checkpoint_epoch_10.pt` | `/kaggle/working/checkpoints/` | Every 10th epoch |
| `checkpoint_epoch_20.pt` | `/kaggle/working/checkpoints/` | Every 10th epoch |

### Checkpoint Contents

Each checkpoint saves:
```python
state = {
    'epoch': epoch,
    'model_state_dict': model_state,       # Unwrapped from DataParallel
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_f1': best_f1,
    'best_epoch': best_epoch,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_f1': val_f1,
}
```

### Assessment

| Criterion | Status | Notes |
|---|---|---|
| Best model saved on val metric | ✅ | Saved on val Pixel-F1 improvement |
| Periodic checkpoints exist | ✅ | Every 10 epochs |
| Training can be resumed | ✅ | `last_checkpoint.pt` with optimizer + scaler state |
| DataParallel handled correctly | ✅ | State dict unwrapped before saving |
| Model artifact uploaded | ✅ | W&B artifact for best model |

**Checkpoint strategy is robust.** The `load_checkpoint()` function handles `module.` prefix mismatches (DataParallel ↔ single GPU), making checkpoints portable. The last checkpoint enables resumption from any interruption.

### Minor Gaps

1. **No checkpoint_epoch_25.pt** — Training stopped at epoch 25 via early stopping, but the periodic checkpoint fires on `(epoch + 1) % 10 == 0`, so epoch 25 isn't a multiple of 10. The `last_checkpoint.pt` covers this, but a "final" checkpoint at early stopping would be more explicit.

2. **No training history saved in checkpoint** — The `history` dict (all epoch metrics) is not included in checkpoint state. If training resumes, the history would be empty. This is a minor gap since the results summary saves final values.

---

## Prediction Quality Analysis

### Best Model: Epoch 15 (F1=0.7289)

The model was loaded from the best checkpoint for evaluation:
```
Loaded best model from epoch 15 (F1=0.7289)
```

### Threshold Selection

```
Best threshold: 0.1327
Best val F1 at threshold: 0.7344
```

The threshold sweep improved F1 from 0.7289 (at default 0.5) to 0.7344 (at 0.1327), a modest improvement that depends heavily on the threshold. This indicates the model's probability calibration is poor.

### Prediction Grid Analysis

The notebook generates a prediction grid with:
- **Best 2** tampered predictions (highest F1)
- **Median 2** tampered predictions
- **Worst 2** tampered predictions
- **2 authentic** samples

This is a well-designed qualitative evaluation that shows the full range of model performance.

### Diagnostic Overlays

The TP/FP/FN color-coded overlays (green=correct, red=false positive, blue=false negative) provide genuine insight into model behavior. The Grad-CAM analysis targets `encoder.layer4`, which is the correct layer for understanding high-level feature attention.

### Failure Case Analysis

```
Failure Case Analysis (worst 10 predictions):
  Mean Pixel-F1:     0.0000
  Mean GT mask area: 0.0961
  Forgery types:     {'splicing': 2, 'copy-move': 8}
  Patterns detected:
    - Fails on small tampered regions (<2% area): 6/10
    - Disproportionately fails on copy-move: 8/10
```

**Key findings:**

1. **Complete failure (F1=0.0)** on the 10 worst cases — the model produces zero overlap with ground truth
2. **Copy-move dominates failures** (8/10) — consistent with the overall copy-move F1=0.31
3. **Small regions fail** (6/10 have <2% area) — expected given the class imbalance and lack of BCE weighting

### Do Predictions Align with Tampered Regions?

Based on the aggregate metrics:

- **Splicing predictions:** Moderate alignment (F1=0.59). The model detects splicing artifacts but with imprecise boundaries.
- **Copy-move predictions:** Poor alignment (F1=0.31). The model largely fails to distinguish copy-move regions from background.
- **False positive rate:** Precision=0.7455 suggests ~25% of predicted tampered pixels are incorrect.
- **False negative rate:** Recall=0.7634 suggests ~24% of actual tampered pixels are missed.

### Boundary Quality

Without direct visual inspection, we can infer from IoU < F1:
- IoU penalizes boundary errors more than F1
- The gap (F1=0.72, IoU=0.70 on mixed-set) is small, suggesting boundary quality is reasonable when the model detects a region at all

The larger gap on tampered-only (F1=0.41, IoU=0.36) suggests the model produces coarse, blobby predictions rather than precise boundary delineation.

---

## Verdict

**Checkpoint strategy:** Robust, well-engineered, handles DataParallel portability.  
**Prediction quality:** The model shows genuine localization ability for splicing but largely fails on copy-move. Complete failures (F1=0.0) on some samples indicate the model is inconsistent. The qualitative evaluation framework (grid, diagnostics, Grad-CAM, failure analysis) is thorough and honest.
