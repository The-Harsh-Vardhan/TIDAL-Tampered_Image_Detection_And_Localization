# Version Notes — vR.1.7

| Field | Value |
|-------|-------|
| **Version** | vR.1.7 |
| **Parent** | vR.1.6 (Deeper CNN — POSITIVE, 90.23% test acc) |
| **Change** | Replace Flatten with GlobalAveragePooling2D |
| **Category** | Architecture |
| **Weakness Addressed** | W10 — Excessive parameters in Flatten→Dense connection (13.8M params) |

---

## 1. What Changed

The Flatten layer is replaced with GlobalAveragePooling2D. This is the **only** modification.

### vR.1.6 Architecture (Before)

```
Conv2D(32, 5×5, ReLU)        → (124, 124, 32)
BatchNormalization            → (124, 124, 32)
Conv2D(32, 5×5, ReLU)        → (120, 120, 32)
BatchNormalization            → (120, 120, 32)
MaxPooling2D(2×2)             → (60, 60, 32)
Conv2D(64, 3×3, ReLU)        → (58, 58, 64)
MaxPooling2D(2×2)             → (29, 29, 64)
Dropout(0.25)                 → (29, 29, 64)
Flatten                       → 53,824
Dense(256, ReLU)              → 256       ← 13,779,200 params (99.7%)
Dropout(0.5)
Dense(2, Softmax)
```

### vR.1.7 Architecture (After)

```
Conv2D(32, 5×5, ReLU)        → (124, 124, 32)
BatchNormalization            → (124, 124, 32)
Conv2D(32, 5×5, ReLU)        → (120, 120, 32)
BatchNormalization            → (120, 120, 32)
MaxPooling2D(2×2)             → (60, 60, 32)
Conv2D(64, 3×3, ReLU)        → (58, 58, 64)
MaxPooling2D(2×2)             → (29, 29, 64)
Dropout(0.25)                 → (29, 29, 64)
GlobalAveragePooling2D        → 64                   ← CHANGED
Dense(256, ReLU)              → 256       ← 16,640 params (26.0%)
Dropout(0.5)
Dense(2, Softmax)
```

### Parameter Impact

| Component | vR.1.6 | vR.1.7 | Change |
|-----------|--------|--------|--------|
| Conv layers | 46,560 | 46,560 | Unchanged |
| BN layers | 256 | 256 | Unchanged |
| Pooling→Dense | 13,779,200 | 16,640 | **−99.9%** |
| Dense(2) | 514 | 514 | Unchanged |
| **Total** | **13,826,530** | **~63,970** | **−99.5%** |

The Flatten layer converts 29×29×64 = 53,824 spatial values into a flat vector, requiring 53,824×256 + 256 = 13,779,200 parameters in the Dense(256) layer. GlobalAveragePooling2D reduces each of the 64 filters to a single average value, requiring only 64×256 + 256 = 16,640 parameters. The model drops from ~13.8M to ~64K trainable parameters.

---

## 2. What DID NOT Change (Frozen)

- ELA quality: 90
- Image size: 128×128
- Optimizer: Adam(lr=0.0001)
- Loss: categorical_crossentropy
- Batch size: 32
- Early stopping: patience=5 on val_accuracy
- ReduceLROnPlateau: monitor=val_loss, factor=0.5, patience=3, min_lr=1e-6
- Seed: 42
- Data split: 70/15/15 train/val/test (stratified)
- Class weights: inverse-frequency balanced (from vR.1.3)
- BatchNormalization after Conv2D layers 1 and 2 (from vR.1.4)
- Conv2D(64, 3×3) + MaxPool layer (from vR.1.6)
- No data augmentation

---

## 3. Cumulative Changes from Baseline (vR.1.0)

1. **vR.1.1:** 70/15/15 split, per-class metrics, ROC-AUC, ELA viz, model save
2. ~~**vR.1.2:** Data augmentation~~ (REJECTED)
3. **vR.1.3:** Class weights (inverse-frequency balanced)
4. **vR.1.4:** BatchNormalization after each Conv2D
5. **vR.1.5:** ReduceLROnPlateau learning rate scheduler
6. **vR.1.6:** 3rd Conv2D(64, 3×3) + MaxPool (deeper feature extraction)
7. **vR.1.7:** GlobalAveragePooling2D replaces Flatten (this version)
