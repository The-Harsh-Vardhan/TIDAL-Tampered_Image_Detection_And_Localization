# Implementation Plan: vR.1.2 — Data Augmentation

---

## Cells to Modify

Only **3 cells** change from vR.1.1. All others are frozen.

### Cell 2 (Imports and Configuration)

**Add version info:**
```python
VERSION = 'vR.1.2'
CHANGE = 'Data augmentation: horizontal flip, vertical flip, rotation ±15°'
```

**Add import (already available from keras):**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

### Cell 17 (Training)

**Replace direct `model.fit(X_train, Y_train, ...)` with:**
```python
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    fill_mode='nearest'
)

train_generator = train_datagen.flow(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    seed=SEED
)

steps_per_epoch = len(X_train) // BATCH_SIZE

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=(X_val, Y_val),  # No augmentation on val
    callbacks=[early_stopping],
    verbose=1
)
```

### Cell 28 (Ablation Tracking Table)

**Add vR.1.1 results as a historical row (hardcoded):**
```
vR.1.0  Baseline (val metrics)    89.89%*  0.8279*  0.9483*  ...
vR.1.1  Eval fix                  88.38%   0.8393   0.8830   ...
vR.1.2  Augmentation (this run)   {live}   {live}   {live}   ...
```

---

## Cells That DO NOT Change

All other cells remain byte-identical to vR.1.1:

- Cell 0: Title and introduction (update version reference only)
- Cell 1: Version change log table (update to show vR.1.2 changes)
- Cells 4–12: Dataset pipeline, ELA, splitting — unchanged
- Cell 14: Model architecture — unchanged
- Cell 16: Model compile — unchanged
- Cells 19–22: Evaluation — unchanged
- Cells 24–26: Visualization — unchanged
- Cell 30: Model save — update filename to vR.1.2

---

## Key Implementation Details

### Why ImageDataGenerator Instead of tf.data

1. **Simplicity** — ImageDataGenerator works directly with numpy arrays already in memory
2. **Minimal code change** — Only the training cell changes
3. **Proven** — Standard Keras approach for in-memory augmentation
4. **No RAM increase** — Augmentation is on-the-fly per batch, not pre-computed

### Why These Specific Augmentations

| Transform | Why included | Why this parameter |
|-----------|-------------|-------------------|
| horizontal_flip | Tampering is not directional. ELA patterns are symmetric. | Binary on/off |
| vertical_flip | Same logic. Doubles effective diversity. | Binary on/off |
| rotation_range=15 | Mild rotation preserves ELA structure. >15° risks interpolation artifacts. | 15° is conservative |

### What is NOT Included (and Why)

| Transform | Why excluded |
|-----------|-------------|
| zoom_range | Could crop out tampered regions, destroying the forensic signal |
| width/height_shift | Could shift tampered regions out of frame |
| shear | Introduces interpolation artifacts that could confuse ELA analysis |
| brightness/contrast | ELA brightness is already normalized — additional changes would interfere |
| channel_shift | ELA channels carry forensic information — shifting them destroys the signal |

---

## Validation Plan

After running on Kaggle, verify:

- [ ] Augmentation only applied to training data (val/test unchanged)
- [ ] `steps_per_epoch` matches `len(X_train) // BATCH_SIZE`
- [ ] Training takes longer per epoch (augmentation overhead)
- [ ] Best epoch likely shifts later (slower convergence = more exploration)
- [ ] Training accuracy is lower than vR.1.1 at same epoch (augmentation makes training harder)
- [ ] Val accuracy plateau is higher (better generalization)
- [ ] Val collapse at epochs 12–13 is reduced or absent
- [ ] Test accuracy ≥ 88.38% (vR.1.1 baseline)
- [ ] FN rate < 11.7% (vR.1.1 baseline)
- [ ] All evaluation metrics computed on unchanged test set
- [ ] Model saved as vR.1.2_ela_cnn_model.keras
