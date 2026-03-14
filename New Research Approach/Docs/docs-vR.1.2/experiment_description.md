# Experiment Description: vR.1.2 — Data Augmentation

| Field | Value |
|-------|-------|
| **Version** | vR.1.2 |
| **Parent** | vR.1.1 (evaluation fix — honest baseline) |
| **Date** | 2026-03-14 |
| **Category** | Data pipeline |
| **Weakness Fixed** | W6 — No data augmentation |
| **Status** | Ready for execution |

---

## Change Introduced

Add **real-time data augmentation** during training to combat overfitting and compensate for the reduced training set (8,829 images in 70/15/15 vs 10,091 in 80/20).

### Augmentation Transforms

| Transform | Parameter | Rationale |
|-----------|-----------|-----------|
| Horizontal flip | 50% probability | Tampering artifacts are not directional — flipping preserves forensic signal |
| Vertical flip | 50% probability | Same rationale as horizontal |
| Random rotation | ±15 degrees | Mild rotation preserves ELA structure while adding pose diversity |

### Implementation Method

Use `tf.keras.preprocessing.image.ImageDataGenerator` for **on-the-fly augmentation** (no RAM increase). Augmentation is applied ONLY to the training set. Validation and test sets are NOT augmented.

```python
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    fill_mode='nearest'
)
# Note: Data is already normalized to [0,1], so no rescaling needed.
# flow() is used with the existing numpy arrays.
```

---

## Motivation

### From vR.1.1 Audit

The vR.1.1 audit identified:

1. **Severe overfitting:** Train acc 92.89% vs Val acc 88.64% at best epoch (4.25pp gap), expanding to 13pp by epoch 13.
2. **Val collapse at epochs 12–13:** Val_loss explodes from 0.31 to 0.64, indicating the model is memorizing training data.
3. **FN rate doubled:** From 5.4% (vR.1.0) to 11.7% (vR.1.1), caused by 10% fewer training images.
4. **29.5M parameters vs 8,829 training images** = massive overfitting capacity. Augmentation is the standard remedy.

### From Ablation Master Plan

> "vR.1.2 (augmentation) is the single most likely change to close the 6.3% accuracy gap. Augmentation is explicitly required by the assignment and is the standard fix for overfitting in small datasets."

---

## What DID NOT Change (Frozen from vR.1.1)

Everything not listed above remains frozen:

- ELA quality: 90
- Image size: 128×128
- CNN architecture: 2×Conv2D(32) + Dense(256) + Dense(2,Softmax)
- Optimizer: Adam(lr=0.0001)
- Loss: categorical_crossentropy
- Batch size: 32
- Early stopping: patience=5 on val_accuracy, restore_best_weights
- Seed: 42
- Data split: 70/15/15 train/val/test (stratified)
- Evaluation: per-class + macro metrics, ROC-AUC on test set
- Model save: active

---

## Expected Impact

| Metric | vR.1.1 | Expected vR.1.2 | Reasoning |
|--------|--------|------------------|-----------|
| Test Accuracy | 88.38% | **89–91%** | Augmentation regularizes, reducing overfitting gap |
| Tampered Recall | 0.8830 | **0.90–0.94** | More diverse training examples improve minority class sensitivity |
| FN rate | 11.7% | **6–10%** | Augmentation compensates for smaller training set |
| Train-val gap | 4.25pp | **2–3pp** | Augmentation reduces memorization |
| Val collapse | Epochs 12–13 | **Later or absent** | Augmentation smooths the loss landscape |
| ROC-AUC | 0.9601 | **0.96–0.97** | Better generalization improves discrimination |
| Best epoch | 8 | **8–15** | More diversity means slower convergence but better final model |
