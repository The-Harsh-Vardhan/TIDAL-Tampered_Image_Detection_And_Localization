# Experiment Description: vR.1.4 — BatchNormalization

---

## Version Info

| Field | Value |
|-------|-------|
| Version | vR.1.4 |
| Parent | **vR.1.3** (class weights) |
| Change | Add BatchNormalization after each Conv2D layer |
| Category | Architecture |
| Weakness fixed | W9 — No BatchNormalization contributes to training instability |

---

## Change Description

Add `BatchNormalization()` layers after each Conv2D activation:

```python
# vR.1.3 (Parent)
Conv2D(32, (5,5), activation='relu', ...)
Conv2D(32, (5,5), activation='relu', ...)

# vR.1.4 (This Version)
Conv2D(32, (5,5), activation='relu', ...)
BatchNormalization()                        # NEW
Conv2D(32, (5,5), activation='relu', ...)
BatchNormalization()                        # NEW
```

BatchNormalization normalizes activations to zero mean and unit variance within each mini-batch, then applies learned scale (gamma) and shift (beta) parameters.

---

## Motivation

### Why This Change

The vR.1.0 baseline showed training instability:
- Epoch 11 spike: val_accuracy dropped 4% in one epoch
- Train-val gap opened after epoch 8 (overfitting)
- Without normalization, internal covariate shift forces the network to constantly adapt to shifting input distributions across layers

BatchNorm addresses this by:
1. **Stabilizing training dynamics** — reduces sensitivity to learning rate and initialization
2. **Acting as a mild regularizer** — batch statistics add noise that reduces overfitting
3. **Enabling faster convergence** — normalized activations allow higher effective learning rates

### Why Now (After vR.1.3)

The ablation roadmap specifies BatchNorm after class weights because:
1. Class weights (vR.1.3) changes only the loss weighting — a non-architectural change
2. BatchNorm changes the architecture — it modifies the forward pass and adds parameters
3. BatchNorm must come before the LR scheduler (vR.1.5) because it changes the loss landscape that the scheduler operates on

### Relationship to Dropout

The model already uses Dropout(0.25) after MaxPooling and Dropout(0.5) after Dense(256). BatchNorm and Dropout serve different purposes:
- **Dropout** — regularization via random neuron zeroing (test-time: no change)
- **BatchNorm** — normalization via batch statistics (test-time: uses running statistics)

Some literature suggests tension between BN and Dropout, but at these Dropout rates the effect is well-studied and generally positive.

---

## What Changes

| Component | vR.1.3 (Parent) | vR.1.4 (This Version) |
|-----------|-----------------|----------------------|
| After Conv2D(32, 5×5) #1 | Nothing | **BatchNormalization()** |
| After Conv2D(32, 5×5) #2 | Nothing | **BatchNormalization()** |
| Total new parameters | 0 | +256 (128 from each BN layer: 64 gamma + 64 beta) |
| Total parameters | 29,520,034 | ~29,520,290 |

### Architecture Comparison

```
vR.1.3 (Parent):                       vR.1.4 (This Version):
───────────────                        ───────────────────────
Input(128,128,3)                       Input(128,128,3)
Conv2D(32, 5×5, ReLU)                  Conv2D(32, 5×5, ReLU)
                                       BatchNormalization()      ← NEW
Conv2D(32, 5×5, ReLU)                  Conv2D(32, 5×5, ReLU)
                                       BatchNormalization()      ← NEW
MaxPooling2D(2×2)                      MaxPooling2D(2×2)
Dropout(0.25)                          Dropout(0.25)
Flatten                                Flatten
Dense(256, ReLU)                       Dense(256, ReLU)
Dropout(0.5)                           Dropout(0.5)
Dense(2, Softmax)                      Dense(2, Softmax)
```

### What Does NOT Change (Frozen)

- ELA quality: 90
- Image size: 128×128
- Conv2D filters: 32, kernel: 5×5, activation: ReLU
- Dense layers: 256 + 2
- Dropout rates: 0.25 and 0.5
- Optimizer: Adam(lr=0.0001)
- Loss: categorical_crossentropy
- Class weights: inverse-frequency balanced (from vR.1.3)
- Batch size: 32
- Early stopping: patience=5 on val_accuracy
- Seed: 42
- Data split: 70/15/15 train/val/test (stratified)
- Evaluation: per-class + macro metrics, ROC-AUC on test set
- No data augmentation

---

## Implementation Details

### BatchNorm Layer Configuration

Using Keras defaults:
- `momentum=0.99` — running mean/variance update rate
- `epsilon=0.001` — numerical stability constant
- `center=True` — learnable beta (shift)
- `scale=True` — learnable gamma (scale)

### Placement Decision: After Activation

BatchNorm is placed **after** the ReLU activation (post-activation BN). While the original BN paper proposed pre-activation placement, post-activation is more common in practice and is the default when using `activation='relu'` in the Conv2D layer.

### Impact on Inference

During training, BN uses per-batch statistics. During inference (`model.predict()`), it uses accumulated running statistics. This makes predictions deterministic — they don't depend on batch composition.

---

## Cumulative Changes from Baseline (vR.1.0)

1. **vR.1.1:** 70/15/15 split, per-class metrics, ROC-AUC, ELA viz, model save
2. ~~**vR.1.2:** Data augmentation~~ (REJECTED)
3. **vR.1.3:** Class weights (inverse-frequency balanced)
4. **vR.1.4:** BatchNormalization after each Conv2D (this version)
