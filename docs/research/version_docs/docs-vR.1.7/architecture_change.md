# Architecture Change — vR.1.7: Global Average Pooling

## 1. The Problem (W10, continued)

vR.1.6 reduced the Flatten→Dense bottleneck from 29.5M to 13.8M parameters by adding a deeper convolutional layer. While this was a clear improvement (+1.85pp accuracy over vR.1.1), the Flatten→Dense(256) connection still accounts for **99.7% of all trainable parameters** (13,779,200 out of 13,826,530).

This causes:
- **Overfitting** — vR.1.6 showed a ~5pp train-val accuracy gap by epoch 18, with val_loss rising from 0.2552 (epoch 9) to 0.3149 (epoch 18)
- **Spatial memorization** — Flatten encodes pixel positions into the weight matrix, making the model sensitive to exact spatial layout
- **Parameter inefficiency** — 99.7% of weights are in a single Dense layer, not learning feature extraction

## 2. The Solution

Replace `Flatten()` with `GlobalAveragePooling2D()`. GAP takes the average of each filter's 29×29 spatial activation map, producing a single value per filter:

```
Before (Flatten):           After (GAP):
29×29×64 feature maps       29×29×64 feature maps
     │                           │
     ▼                           ▼
Flatten: 53,824 values      GAP: avg each 29×29 map → 64 values
     │                           │
     ▼                           ▼
Dense(256): 13,779,200 W    Dense(256): 16,640 W
```

Each of the 64 filters produces one number — the spatial average of its activation. This forces each filter to detect the **presence** of a feature across the image, not its **exact position**.

## 3. Layer-by-Layer Data Flow

```
INPUT: (128, 128, 3) — ELA image, normalized [0,1]
       │
       ▼
Conv2D(32, 5×5, ReLU, valid)     → (124, 124, 32)    2,432 params
BatchNormalization                → (124, 124, 32)      128 params
       │
       ▼
Conv2D(32, 5×5, ReLU, valid)     → (120, 120, 32)   25,632 params
BatchNormalization                → (120, 120, 32)      128 params
       │
       ▼
MaxPooling2D(2×2)                 → (60, 60, 32)          0 params
       │
       ▼
Conv2D(64, 3×3, ReLU, valid)     → (58, 58, 64)    18,496 params
       │
       ▼
MaxPooling2D(2×2)                 → (29, 29, 64)          0 params
       │
       ▼
Dropout(0.25)                     → (29, 29, 64)          0 params
       │
       ▼
GlobalAveragePooling2D            → 64                     0 params  ← CHANGED
       │
       ▼
Dense(256, ReLU)                  → 256               16,640 params  ← was 13,779,200
Dropout(0.5)                      → 256                    0 params
Dense(2, Softmax)                 → 2                    514 params
       │
       ▼
OUTPUT: [P(Authentic), P(Tampered)]

TOTAL: ~63,970 params (down from 13,826,530)
```

## 4. Why GlobalAveragePooling2D?

| Benefit | Explanation |
|---------|-------------|
| **Massive regularization** | 99.5% fewer parameters → dramatically less overfitting capacity |
| **Spatial invariance** | Each filter reports presence, not position. Tampered regions detected regardless of where they appear |
| **No new hyperparameters** | GAP is deterministic — no weights, no tuning needed |
| **Standard practice** | Used in ResNet, GoogLeNet, MobileNet, EfficientNet — proven in modern CNNs |
| **Forces discriminative filters** | Each of 64 filters must be a useful global feature detector, since the classifier only sees 64 numbers |

## 5. The Key Risk

GAP discards all spatial information within each filter. In ELA-based detection, the **location** of compression artifacts may matter:
- Spliced regions have different ELA intensity than surrounding areas
- The spatial pattern of high-ELA pixels relative to low-ELA pixels carries information
- Flatten preserves this spatial structure; GAP averages it away

If ELA detection requires fine-grained spatial reasoning, GAP may hurt performance. However, the Conv layers have already extracted spatial features into 64 filter channels. GAP should preserve the "which patterns are present" information while discarding "where exactly they are."

## 6. What This Does NOT Change

- All three Conv2D layers and their parameters are identical
- BatchNormalization layers are unchanged
- Dropout rates (0.25 and 0.5) are unchanged
- Dense(256) + Dense(2) classifier head structure is the same (just with fewer input connections)
- All training configuration (optimizer, LR, scheduler, callbacks, class weights) is frozen
