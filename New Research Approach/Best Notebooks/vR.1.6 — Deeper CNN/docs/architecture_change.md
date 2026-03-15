# Architecture Change — vR.1.6: Deeper CNN

## 1. The Problem (W10)

The ETASR CNN has only **2 convolutional layers** extracting features from 128×128×3 ELA images. After MaxPooling, the feature maps are 60×60×32 = **115,200 values**. These are flattened and fed directly into Dense(256), creating a single layer with **29,491,456 parameters** — 99.9% of the entire model.

This is the root cause of:
- **Training instability** (val collapse at epochs 12-14 in every run)
- **Augmentation incompatibility** (vR.1.2 REJECTED — Flatten memorizes pixel positions)
- **Overfitting** (train_acc=0.95 vs val_acc=0.89 gap)
- **Stalled accuracy** (~88-89% ceiling despite 4 ablations)

## 2. The Solution

Add a **3rd Conv2D(64, 3×3, ReLU) + MaxPooling2D(2×2)** between the existing MaxPool and Dropout. This:

1. **Deepens feature extraction** — 3 conv layers instead of 2, with the 3rd using 64 filters (2× the first two layers) and a smaller 3×3 kernel for fine-grained pattern detection
2. **Reduces the Flatten size by 53%** — from 115,200 to 53,824 (29×29×64), cutting the Dense(256) layer from 29.5M to 13.8M parameters
3. **Adds hierarchical feature processing** — low-level edges (layer 1) → mid-level textures (layer 2) → higher-level ELA patterns (layer 3)

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
Conv2D(64, 3×3, ReLU, valid)     → (58, 58, 64)    18,496 params  ← NEW
       │
       ▼
MaxPooling2D(2×2)                 → (29, 29, 64)          0 params  ← NEW
       │
       ▼
Dropout(0.25)                     → (29, 29, 64)          0 params
       │
       ▼
Flatten                           → 53,824                 0 params
       │
       ▼
Dense(256, ReLU)                  → 256           13,779,200 params
Dropout(0.5)                      → 256                    0 params
Dense(2, Softmax)                 → 2                    514 params
       │
       ▼
OUTPUT: [P(Authentic), P(Tampered)]

TOTAL: 13,826,530 params (down from 29,520,290)
```

## 4. Why Conv2D(64, 3×3)?

| Design Choice | Rationale |
|--------------|-----------|
| **64 filters** (vs 32) | Deeper layers should learn more diverse features. Doubling filter count at each stage is standard (VGG, ResNet). |
| **3×3 kernel** (vs 5×5) | Smaller kernel captures finer patterns. The receptive field at layer 3 already covers a large area through two prior conv+pool stages. 3×3 is the standard modern choice. |
| **ReLU activation** | Consistent with existing layers. |
| **valid padding** | Consistent with existing layers. |
| **No BN on layer 3** | BN is only on the original two Conv2D layers (from vR.1.4). Adding BN to the new layer would violate single-variable ablation — we are adding depth, not more normalization. |
| **MaxPool after layer 3** | Reduces spatial dimensions before Flatten, which is the primary goal. |

## 5. What This Does NOT Change

- The first two Conv2D layers and their BN are identical
- The Dense layers are identical (256 + 2)
- All training configuration is frozen
- The Dropout positions and rates are unchanged
- The only structural change is inserting Conv2D(64,3×3) + MaxPool between the first MaxPool and Dropout
