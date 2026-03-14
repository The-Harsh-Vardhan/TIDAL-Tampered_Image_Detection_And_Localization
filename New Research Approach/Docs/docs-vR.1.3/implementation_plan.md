# Implementation Plan: vR.1.3 — Class Weights

---

## Cell-by-Cell Changes from vR.1.1

vR.1.3 branches from **vR.1.1** (not vR.1.2). The following cells are modified:

### Cell 0 (Markdown): Title and Introduction

- Title: `# vR.1.3 — ETASR Ablation Study: Class Weights`
- Parent version: vR.1.1
- Change description: Add inverse-frequency class weights
- Pipeline diagram: unchanged (class weights don't change the pipeline, only the loss computation)
- Note: Mention that vR.1.2 (augmentation) was rejected

### Cell 1 (Markdown): Version Change Log

- Table: vR.1.1 → vR.1.3 diff
- Note: vR.1.2 listed as rejected
- Cumulative changes: vR.1.1 (eval fix) + vR.1.3 (class weights)

### Cell 2 (Code): Imports and Configuration

- Add import: `from sklearn.utils.class_weight import compute_class_weight`
- VERSION = 'vR.1.3'
- CHANGE = 'Class weights: inverse-frequency balanced weighting'
- Remove AUGMENTATION dict (not used — branching from vR.1.1)

### Cell 12 (Code): Data Splitting

- After split, compute and print class weights:
```python
Y_train_int = np.argmax(Y_train, axis=1)
class_weights_array = compute_class_weight('balanced', classes=np.array([0, 1]), y=Y_train_int)
CLASS_WEIGHT_DICT = {0: class_weights_array[0], 1: class_weights_array[1]}
print(f'Class weights: {CLASS_WEIGHT_DICT}')
```

### Cell 15 (Markdown): Training Pipeline Description

- Update to mention class weights
- Explain the formula: `w_c = n_samples / (n_classes × n_c)`

### Cell 17 (Code): Training

- Revert to `model.fit(X_train, Y_train, ...)` (no generator, no augmentation)
- Add `class_weight=CLASS_WEIGHT_DICT` parameter
- Print class weights before training starts

### Cell 28 (Code): Ablation Tracking Table

- Add vR.1.1 and vR.1.2 as historical rows
- vR.1.2 row annotated with "REJECTED"

### Cell 29 (Markdown): Discussion

- Explain class weights change
- Explain why vR.1.2 was rejected and why we branch from vR.1.1
- Preview next version (vR.1.4: BatchNormalization)

### Cell 30 (Code): Model Save

- Filename: `vR.1.3_ela_cnn_model.keras`

---

## Key Implementation Details

### Class Weight Computation

```python
from sklearn.utils.class_weight import compute_class_weight

Y_train_int = np.argmax(Y_train, axis=1)
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.array([0, 1]),
    y=Y_train_int
)
CLASS_WEIGHT_DICT = {0: class_weights_array[0], 1: class_weights_array[1]}
```

The `'balanced'` mode computes: `w_c = n_samples / (n_classes × n_c)`

For the vR.1.1 training set (5,243 Au + 3,586 Tp):
- Authentic: 8829 / (2 × 5243) ≈ 0.842
- Tampered: 8829 / (2 × 3586) ≈ 1.231

### model.fit() Call

```python
history = model.fit(
    X_train, Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping],
    class_weight=CLASS_WEIGHT_DICT,  # NEW in vR.1.3
    verbose=1
)
```

---

## Validation Checklist

Before running on Kaggle, verify:

- [ ] VERSION = 'vR.1.3'
- [ ] No augmentation code (branching from vR.1.1)
- [ ] `compute_class_weight` imported
- [ ] CLASS_WEIGHT_DICT computed after split
- [ ] `class_weight=CLASS_WEIGHT_DICT` in model.fit()
- [ ] No `ImageDataGenerator` (removed)
- [ ] Training uses `model.fit(X_train, Y_train, ...)` (not generator)
- [ ] Ablation table includes vR.1.1 and vR.1.2 (rejected) rows
- [ ] Model save filename: vR.1.3_ela_cnn_model.keras
- [ ] All other parameters frozen from vR.1.1
