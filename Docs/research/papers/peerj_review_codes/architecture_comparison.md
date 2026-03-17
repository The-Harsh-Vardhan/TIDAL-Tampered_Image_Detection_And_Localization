# Architecture Comparison — Paper vs Both Notebooks

**Paper:** "Enhanced Image Tampering Detection using Error Level Analysis and a CNN" (Nagm et al., PeerJ CS 2024)
**NB1:** `CASIA2 ELA CNN Image Tampering Detection.ipynb`
**NB2:** `ELA CNN Image Forgery Detection.ipynb`

---

## TL;DR

| Notebook | Matches Paper? | Closest Paper Model |
|----------|---------------|-------------------|
| **NB1** | **Yes** (faithful reproduction + improvements) | Proposed model (94.14% test acc) |
| **NB2** | **No** (completely different architecture) | Resembles ablation Model 2 (88.9% test acc) |

---

## Layer-by-Layer Architecture Comparison

### Paper's Proposed Model vs NB1

| # | Paper (Proposed) | NB1 (CASIA2 ELA CNN) | Match? |
|---|-----------------|---------------------|--------|
| 1 | Conv2D(32, 5×5, valid, relu) | Conv2D(32, 5×5, valid, relu) | ✅ Exact |
| 2 | Conv2D(32, 5×5, valid, relu) | Conv2D(32, 5×5, valid, relu) | ✅ Exact |
| 3 | MaxPool2D(2×2) | MaxPool2D(2×2) | ✅ Exact |
| 4 | Dropout(0.25) | Dropout(0.25) | ✅ Exact |
| 5 | Flatten | Flatten | ✅ Exact |
| 6 | Dense(150, relu) | Dense(150, relu) | ✅ Exact |
| 7 | Dropout(0.5) | Dropout(0.5) | ✅ Exact |
| 8 | Dense(2, sigmoid) | Dense(2, sigmoid) | ✅ Exact |

**Verdict:** NB1 is a **100% faithful reproduction** of the paper's proposed architecture.

### Paper's Proposed Model vs NB2

| # | Paper (Proposed) | NB2 (ELA CNN Forgery) | Match? |
|---|-----------------|----------------------|--------|
| 1 | Conv2D(32, 5×5, valid, relu) | Conv2D(**64**, **3×3**, valid, relu) | ❌ Different filters & kernel |
| 2 | — | **BatchNormalization()** | ❌ Not in paper |
| 3 | — | **MaxPool2D(2×2)** | ❌ Extra pooling layer |
| 4 | Conv2D(32, 5×5, valid, relu) | Conv2D(**128**, **3×3**, valid, relu) | ❌ Different filters & kernel |
| 5 | — | **BatchNormalization()** | ❌ Not in paper |
| 6 | MaxPool2D(2×2) | **MaxPool2D(2×2)** | ❌ Different position |
| 7 | — | Conv2D(**256**, **3×3**, valid, relu) | ❌ Extra conv block |
| 8 | — | **BatchNormalization()** | ❌ Not in paper |
| 9 | — | **MaxPool2D(2×2)** | ❌ Extra pooling |
| 10 | Dropout(0.25) | Dropout(**0.5**) | ❌ Different rate |
| 11 | Flatten | Flatten | ✅ |
| 12 | Dense(150, relu) | Dense(**512**, relu) | ❌ Different units |
| 13 | Dropout(0.5) | Dropout(0.5) | ✅ |
| 14 | Dense(2, sigmoid) | Dense(2, sigmoid) | ✅ |

**Verdict:** NB2's architecture has **11 out of 14 layers different** from the paper. It is a 3-block CNN with BatchNormalization, resembling the paper's **ablation Model 2** which achieved only 88.9% test accuracy.

---

## Full Parameter Comparison

| Parameter | Paper | NB1 | NB2 |
|-----------|-------|-----|-----|
| **Image size** | 150×150 | 150×150 ✅ | 150×150 ✅ |
| **Input channels** | 3 (RGB ELA) | 3 ✅ | 3 ✅ |
| **Conv layers** | 2 | 2 ✅ | 3 ❌ |
| **Conv filters** | 32, 32 | 32, 32 ✅ | 64, 128, 256 ❌ |
| **Conv kernel** | 5×5 | 5×5 ✅ | 3×3 ❌ |
| **Conv padding** | valid | valid ✅ | valid ✅ |
| **BatchNorm** | None | None ✅ | After each conv ❌ |
| **MaxPool layers** | 1 | 1 ✅ | 3 ❌ |
| **Dropout rates** | 0.25, 0.5 | 0.25, 0.5 ✅ | 0.5, 0.5 ❌ |
| **Dense hidden** | 150 | 150 ✅ | 512 ❌ |
| **Output** | Dense(2, sigmoid) | Dense(2, sigmoid) ✅ | Dense(2, sigmoid) ✅ |
| **~Parameters** | 24.2M | 24.2M ✅ | Much less ❌ |
| **Optimizer** | Adam(lr=1e-4) | Adam(lr=1e-4) ✅ | Adam(lr=1e-4) ✅ |
| **Loss** | binary_crossentropy | binary_crossentropy ✅ | binary_crossentropy ✅ |
| **Batch size** | 8 | 8 ✅ | 8 ✅ |
| **Epochs** | 40 | 40 ✅ | 40 ✅ |
| **ELA quality** | 90 | 90 ✅ | 90 ✅ |
| **ELA method** | Temp file on disk | BytesIO (in-memory) ⬆️ | Temp file on disk ✅ |
| **Data split** | 80/16/4 | 80/10/10 ⚠️ | 70/15/15 ❌ |
| **Stratified split** | No | Yes ⬆️ | No ✅ |
| **Callbacks** | None (commented out) | None ✅ | EarlyStopping(patience=2) ❌ |
| **Augmentation** | None | None ✅ | None ✅ |
| **Metrics (Keras)** | acc, precision, recall | acc, precision, recall ✅ | acc, precision, recall ✅ |
| **Metrics (sklearn)** | None | F1, classification_report ⬆️ | F1, classification_report ⬆️ |
| **Seeds set** | No | Yes (42) ⬆️ | No ✅ |

Legend: ✅ = matches paper, ❌ = deviates from paper, ⬆️ = improvement over paper, ⚠️ = minor difference

---

## Why NB2 Resembles Ablation Model 2

The paper's ablation Table 4 describes **Model 2** as:

> "4× (Conv → BatchNorm → MaxPool), Dropout, Flatten, Dense, Dropout, Dense"

NB2 implements:

> 3× (Conv → BatchNorm → MaxPool), Dropout, Flatten, Dense(512), Dropout, Dense

This is structurally the same pattern — multiple conv blocks each followed by BatchNormalization and MaxPooling — with one fewer block and a larger dense layer. Model 2 achieved:
- Train accuracy: 96.66%
- **Test accuracy: 88.9%** (vs proposed model's 94.14%)

NB2 is therefore expected to underperform the paper's proposed model by approximately 5 percentage points on test accuracy.

---

## Visual Architecture Diagrams

### NB1 (matches paper)
```
Input(150×150×3)
  ↓
Conv2D(32, 5×5, relu)  →  146×146×32
  ↓
Conv2D(32, 5×5, relu)  →  142×142×32
  ↓
MaxPool2D(2×2)          →  71×71×32
  ↓
Dropout(0.25)
  ↓
Flatten                 →  161,312
  ↓
Dense(150, relu)
  ↓
Dropout(0.5)
  ↓
Dense(2, sigmoid)       →  [tampered, authentic]
```

### NB2 (different architecture)
```
Input(150×150×3)
  ↓
Conv2D(64, 3×3, relu)  →  148×148×64
BatchNorm               →  148×148×64
MaxPool2D(2×2)          →  74×74×64
  ↓
Conv2D(128, 3×3, relu) →  72×72×128
BatchNorm               →  72×72×128
MaxPool2D(2×2)          →  36×36×128
  ↓
Conv2D(256, 3×3, relu) →  34×34×256
BatchNorm               →  34×34×256
MaxPool2D(2×2)          →  17×17×256
  ↓
Dropout(0.5)
  ↓
Flatten                 →  73,984
  ↓
Dense(512, relu)
  ↓
Dropout(0.5)
  ↓
Dense(2, sigmoid)       →  [tampered, authentic]
```

---

## Conclusion

**NB1 is the correct implementation of the paper.** It faithfully reproduces every architectural choice and adds meaningful improvements (BytesIO ELA, stratified splitting, reproducibility seeds, sklearn metrics).

**NB2 implements a different model entirely** — a deeper, BatchNorm-heavy CNN that resembles the paper's ablation Model 2. While this architecture is valid, it is NOT the paper's proposed model and should be expected to yield lower test accuracy (~88.9% vs 94.14%).
