# Experiment Description: vR.1.3 — Class Weights

---

## Version Info

| Field | Value |
|-------|-------|
| Version | vR.1.3 |
| Parent | **vR.1.1** (NOT vR.1.2 — augmentation was rejected) |
| Change | Add inverse-frequency class weights to training |
| Category | Data pipeline / Training |
| Weakness fixed | W7 — Class imbalance (1.46:1) not addressed |

---

## Change Description

Add `class_weight` parameter to `model.fit()` using inverse-frequency weighting:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights_array = compute_class_weight('balanced', classes=np.array([0, 1]), y=Y_train_int)
class_weight_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
```

This makes the loss function penalize misclassification of the minority class (Tampered) more heavily, proportional to its under-representation.

### Expected Weights

Given the training set distribution (5,243 Au + 3,586 Tp = 8,829 total):
- Authentic weight: `8829 / (2 × 5243)` ≈ 0.842
- Tampered weight: `8829 / (2 × 3586)` ≈ 1.231

The tampered class will receive ~46% more penalty per misclassification than the authentic class.

---

## Motivation

### Why This Change

The vR.1.1 confusion matrix shows:
- FP rate: 11.6% (130 authentic → tampered)
- FN rate: 11.7% (90 tampered → authentic)

While these rates appear balanced percentage-wise, the absolute counts differ (130 FP vs 90 FN) because there are more authentic images in the test set. The model's loss function treats every misclassification equally regardless of class frequency, which biases training toward the majority class.

### Why Now (After Rejected vR.1.2)

The master ablation plan specified class weights as the next item after augmentation. Since augmentation was rejected (NEGATIVE verdict: -2.85pp accuracy), we branch from vR.1.1 and proceed to class weights. This is a low-risk change that:
1. Doesn't modify the data pipeline or architecture
2. Only changes the loss weighting
3. Has well-understood theoretical behavior
4. Is unlikely to cause training failure

### Why Not Re-order the Roadmap?

We follow the plan unless 3 consecutive versions are NEGATIVE/NEUTRAL. vR.1.2 was the first NEGATIVE result. We proceed to vR.1.3 as planned, branching from the last known good version (vR.1.1).

---

## What Changes

| Component | vR.1.1 (Parent) | vR.1.3 (This Version) |
|-----------|-----------------|----------------------|
| Class weights | None | `compute_class_weight('balanced')` |
| `model.fit()` arg | No class_weight | `class_weight=class_weight_dict` |
| Training method | `model.fit(X_train, Y_train, ...)` | `model.fit(X_train, Y_train, ..., class_weight=class_weight_dict)` |

### What Does NOT Change (Frozen)

- ELA quality: 90
- Image size: 128×128
- CNN architecture: 2×Conv2D(32) + Dense(256) + Dense(2)
- Optimizer: Adam(lr=0.0001)
- Loss: categorical_crossentropy
- Batch size: 32
- Early stopping: patience=5 on val_accuracy
- Seed: 42
- Data split: 70/15/15 train/val/test (stratified)
- Evaluation: per-class + macro metrics, ROC-AUC on test set
- **No data augmentation** (rejected in vR.1.2)
