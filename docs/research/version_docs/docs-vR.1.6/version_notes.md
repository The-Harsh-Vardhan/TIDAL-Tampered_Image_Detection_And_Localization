# Version Notes — vR.1.6

| Field | Value |
|-------|-------|
| **Version** | vR.1.6 |
| **Parent** | vR.1.5 (LR Scheduler — NEUTRAL) |
| **Change** | Add 3rd Conv2D(64, 3×3, ReLU) + MaxPooling2D(2×2) before Flatten |
| **Category** | Architecture |
| **Weakness Addressed** | W10 — Excessive reliance on Flatten→Dense with insufficient convolutional feature extraction |

---

## 1. What Changed

A third convolutional layer is inserted between the existing MaxPooling2D and Dropout:

### vR.1.5 Architecture (Before)

```
Conv2D(32, 5×5, ReLU)        → (124, 124, 32)
BatchNormalization            → (124, 124, 32)
Conv2D(32, 5×5, ReLU)        → (120, 120, 32)
BatchNormalization            → (120, 120, 32)
MaxPooling2D(2×2)             → (60, 60, 32)
Dropout(0.25)                 → (60, 60, 32)
Flatten                       → 115,200
Dense(256, ReLU)              → 256       ← 29,491,456 params (99.9%)
Dropout(0.5)
Dense(2, Softmax)
```

### vR.1.6 Architecture (After)

```
Conv2D(32, 5×5, ReLU)        → (124, 124, 32)
BatchNormalization            → (124, 124, 32)
Conv2D(32, 5×5, ReLU)        → (120, 120, 32)
BatchNormalization            → (120, 120, 32)
MaxPooling2D(2×2)             → (60, 60, 32)
Conv2D(64, 3×3, ReLU)        → (58, 58, 64)    ← NEW
MaxPooling2D(2×2)             → (29, 29, 64)    ← NEW
Dropout(0.25)                 → (29, 29, 64)
Flatten                       → 53,824
Dense(256, ReLU)              → 256       ← 13,779,200 params (now 97.5%)
Dropout(0.5)
Dense(2, Softmax)
```

### Parameter Impact

| Component | vR.1.5 | vR.1.6 | Change |
|-----------|--------|--------|--------|
| Conv layers | 28,064 | 46,464 (+18,400) | +65.6% |
| BN layers | 256 | 256 | Unchanged |
| Flatten→Dense | 29,491,456 | 13,779,200 | **−53.3%** |
| Dense(2) | 514 | 514 | Unchanged |
| **Total** | **29,520,290** | **13,826,434** | **−53.2%** |

The 3rd Conv2D adds 18,400 parameters but the reduced Flatten size (115,200→53,824) removes **15.7M parameters** from the Dense layer. Net reduction: ~15.7M params (53%).

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
- No data augmentation

---

## 3. Cumulative Changes from Baseline (vR.1.0)

1. **vR.1.1:** 70/15/15 split, per-class metrics, ROC-AUC, ELA viz, model save
2. ~~**vR.1.2:** Data augmentation~~ (REJECTED)
3. **vR.1.3:** Class weights (inverse-frequency balanced)
4. **vR.1.4:** BatchNormalization after each Conv2D
5. **vR.1.5:** ReduceLROnPlateau learning rate scheduler
6. **vR.1.6:** 3rd Conv2D(64, 3×3) + MaxPool (this version)
