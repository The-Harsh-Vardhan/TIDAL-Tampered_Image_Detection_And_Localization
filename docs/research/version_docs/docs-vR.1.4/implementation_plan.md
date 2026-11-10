# Implementation Plan: vR.1.4 — BatchNormalization

---

## Notebook Structure

The notebook follows the same 12-section structure as vR.1.3. Only Section 7 (Model Architecture) changes.

---

## Changes Required

### 1. Imports (Section 2)

Add `BatchNormalization` to the Keras imports:

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
```

### 2. Version Info (Section 2)

```python
VERSION = 'vR.1.4'
CHANGE = 'BatchNormalization: add BN layer after each Conv2D to stabilize training'
```

### 3. Model Architecture (Section 7) — THE ONLY FUNCTIONAL CHANGE

```python
def build_model(input_shape=(128, 128, 3)):
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', padding='valid', input_shape=input_shape),
        BatchNormalization(),  # NEW in vR.1.4
        Conv2D(32, (5, 5), activation='relu', padding='valid'),
        BatchNormalization(),  # NEW in vR.1.4
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    return model
```

### 4. Training (Section 8)

**No change**. `class_weight=CLASS_WEIGHT_DICT` remains from vR.1.3.

### 5. Markdown Documentation Updates

- Title: vR.1.4
- Section 1: Version change log updated
- Section 7: Architecture table updated with BN rows
- Section 12: Discussion updated to explain BatchNorm

---

## What Must NOT Change

| Component | Value | Reason |
|-----------|-------|--------|
| ELA preprocessing | Q=90, 128×128, /255.0 | Frozen |
| Data split | 70/15/15, seed=42, stratified | Frozen |
| Class weights | compute_class_weight('balanced') | Inherited from vR.1.3 |
| Optimizer | Adam(lr=0.0001) | Frozen |
| Loss | categorical_crossentropy | Frozen |
| Batch size | 32 | Frozen |
| Early stopping | val_accuracy, patience=5 | Frozen |
| Evaluation metrics | Per-class, macro, ROC-AUC on test | Frozen |
| Dropout rates | 0.25 (conv), 0.5 (dense) | Frozen |

---

## Verification Checklist

- [ ] Only `build_model()` function changes
- [ ] BatchNormalization appears exactly twice (after each Conv2D)
- [ ] No other layers added, removed, or reordered
- [ ] Class weights still applied in model.fit()
- [ ] Model summary shows ~29,520,290 total params (+256 from BN)
- [ ] Ablation tracking table includes vR.1.4 row
- [ ] Model saved as `vR.1.4_ela_cnn_model.keras`
