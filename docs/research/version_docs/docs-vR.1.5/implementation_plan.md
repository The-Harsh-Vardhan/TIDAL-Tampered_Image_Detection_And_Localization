# Implementation Plan — vR.1.5

| Field | Value |
|-------|-------|
| **Version** | vR.1.5 |
| **Source Notebook** | `vR.1.4 Image Detection and Localisation.ipynb` |
| **Output Notebook** | `vR.1.5 ETASR Run-01 Image Detection and Localisation.ipynb` |
| **Build Method** | Python script modifying parent notebook JSON |
| **Single Change** | Add `ReduceLROnPlateau` callback |

---

## 1. Cell-by-Cell Changes

### Code Cells Modified

| Cell | Section | Change |
|------|---------|--------|
| 2 | Config/Imports | Add `ReduceLROnPlateau` import, update `VERSION`, `CHANGE`, version print |
| 17 | Training | Add `lr_scheduler` callback, include in `model.fit()` callbacks list |

### Markdown Cells Modified

| Cell | Section | Change |
|------|---------|--------|
| 0 | Title | Update version to vR.1.5, update change description |
| 1 | Changelog | Add vR.1.5 entry, show diff from vR.1.4 |
| 15 | Training pipeline | Add LR scheduler to pipeline description |
| 28 | Tracking table | Add vR.1.4 results (88.75%, NEUTRAL), vR.1.5 as live row |
| 29 | Discussion | Update next steps to reference vR.1.6 |

### Cells NOT Modified (Frozen)

All other cells remain identical to vR.1.4:
- Dataset loading (cells 4-5)
- ELA preprocessing (cells 7-9)
- Data splitting + class weights (cells 11-12)
- Model architecture with BN (cell 14)
- Callbacks — early stopping definition (cell 16)
- Evaluation suite (cells 19-23)
- Paper comparison (cell 25-26)
- Model save (cell 28, 30)

---

## 2. Key Implementation Details

### 2.1 Import Addition (Cell 2)

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

### 2.2 Version Constants (Cell 2)

```python
VERSION = 'vR.1.5'
CHANGE = 'LR Scheduler: add ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=3, min_lr=1e-6)'
```

### 2.3 LR Scheduler Definition (Cell 17 — before model.fit)

```python
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)
```

**Important:** NO `verbose` parameter. Removed in newer Keras versions and causes errors on Kaggle.

### 2.4 model.fit Callbacks (Cell 17)

```python
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=MAX_EPOCHS,
    validation_data=(X_val, y_val),
    class_weight=CLASS_WEIGHT_DICT,
    callbacks=[early_stopping, lr_scheduler],  # ← lr_scheduler added
    verbose=1
)
```

### 2.5 Model Save Name

```python
model.save(f'{VERSION}_ela_cnn_model.keras')
# → vR.1.5_ela_cnn_model.keras
```

---

## 3. Verification Checklist

After notebook generation, verify:

| # | Check | Expected |
|---|-------|----------|
| 1 | `ReduceLROnPlateau` in imports | ✅ Present |
| 2 | `VERSION = 'vR.1.5'` | ✅ Exact match |
| 3 | `lr_scheduler = ReduceLROnPlateau(` present | ✅ Present |
| 4 | NO `verbose` on ReduceLROnPlateau | ✅ Not present |
| 5 | `callbacks=[early_stopping, lr_scheduler]` in model.fit | ✅ Both callbacks |
| 6 | `class_weight=CLASS_WEIGHT_DICT` preserved | ✅ From vR.1.3 |
| 7 | `BatchNormalization()` appears 2× | ✅ From vR.1.4 |
| 8 | Model saves as `vR.1.5_ela_cnn_model.keras` | ✅ Version prefix |
| 9 | ELA quality = 90 | ✅ Frozen |
| 10 | Image size = (128, 128) | ✅ Frozen |
| 11 | Seed = 42 | ✅ Frozen |
| 12 | Batch size = 32 | ✅ Frozen |
| 13 | Learning rate = 0.0001 (initial) | ✅ Frozen |
| 14 | Early stopping patience = 5 | ✅ Frozen |
| 15 | 70/15/15 split | ✅ Frozen |

---

## 4. ReduceLROnPlateau Configuration Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `monitor` | `'val_loss'` | Loss is smoother than accuracy; better signal for LR decisions |
| `factor` | `0.5` | Standard halving; aggressive enough to matter, gentle enough to not kill learning |
| `patience` | `3` | 3 epochs = shorter than early stopping patience (5). LR reduces before ES triggers. |
| `min_lr` | `1e-6` | Floor to prevent LR from reaching zero. 100× below initial LR (1e-4). |

### Interaction with Early Stopping

```
Epoch flow:
  Epochs 1-N:  LR = 1e-4 (initial)
  If val_loss plateaus for 3 epochs → LR = 5e-5
  If val_loss plateaus for 3 more → LR = 2.5e-5
  ...
  If val_accuracy doesn't improve for 5 epochs → early stopping triggers
```

The scheduler's patience (3) is deliberately shorter than early stopping's patience (5), so the model gets at least one LR reduction before training stops.

---

## 5. Runtime Estimate

| Phase | Time |
|-------|------|
| Dataset loading | ~30s |
| ELA preprocessing | ~5 min |
| Model build | ~1s |
| Training (est. 15-30 epochs) | ~3-8 min |
| Evaluation | ~30s |
| **Total** | **~10-15 min** |

Training is expected to be longer than vR.1.4 (8 epochs) because the LR scheduler gives the model more room to converge gradually.
