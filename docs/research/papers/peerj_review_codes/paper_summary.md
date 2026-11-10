# Paper Reference Card — PeerJ CS 10-2205

**Title:** Enhanced Image Tampering Detection using Error Level Analysis and a CNN
**Authors:** Ahmad M. Nagm et al.
**Published in:** PeerJ Computer Science, 2024
**Local PDF:** `Docs/Papers/peerj-cs-10-2205.pdf`

---

## Proposed Architecture

| # | Layer | Config | Output Shape |
|---|-------|--------|-------------|
| 1 | Input | — | 150 × 150 × 3 |
| 2 | Conv2D | 32 filters, 5×5 kernel, padding='valid', activation='relu' | 146 × 146 × 32 |
| 3 | Conv2D | 32 filters, 5×5 kernel, padding='valid', activation='relu' | 142 × 142 × 32 |
| 4 | MaxPool2D | pool_size=2×2 | 71 × 71 × 32 |
| 5 | Dropout | rate=0.25 | 71 × 71 × 32 |
| 6 | Flatten | — | 161,312 |
| 7 | Dense | 150 units, activation='relu' | 150 |
| 8 | Dropout | rate=0.5 | 150 |
| 9 | Dense | 2 units, activation='sigmoid' | 2 |

**Total parameters:** ~24.2 million
**No BatchNormalization** in the proposed model.

---

## ELA Configuration

| Parameter | Value |
|-----------|-------|
| JPEG recompression quality | 90 |
| Scaling method | Dynamic brightness: `scale = 255.0 / max_pixel_diff` |
| Implementation | Temp file on disk (`resaved.jpg`) |
| Resize | PIL `.resize()` to 150×150 |
| Normalization | Pixel values / 255.0 |

**Note:** The paper never specifies Q=90 in the text — only recoverable from the reference code.

---

## Dataset

| Detail | Value |
|--------|-------|
| Dataset | CASIA 2.0 |
| Total images | 12,614 |
| Authentic (Au) | 7,491 |
| Tampered (Tp) | 5,123 (3,274 copy-move + 1,849 splicing) |
| **Used (JPEG only)** | **9,501** |
| Original sizes | 320×240 and 800×600 |
| File formats excluded | TIFF (no explanation given) |

---

## Data Split

**Paper claims:** "80% training and 20% validation and testing"

**Actual split in reference code:**

```python
X_train, X_rest = train_test_split(X, Y, test_size=0.2)   # 80% train
X_test, X_val   = train_test_split(X_rest, test_size=0.2)  # 16% test, 4% val
```

| Set | Percentage | ~Count |
|-----|-----------|--------|
| Train | 80% | 7,601 |
| Test | 16% | 1,520 |
| Validation | **4%** | **380** |

The 4% validation set is extremely small and not clearly disclosed.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Batch size | 8 |
| Epochs | 40 |
| Loss | Binary cross-entropy |
| Metrics | Accuracy, Precision, Recall |
| Early stopping | Defined in code but **commented out** (not used) |
| Hardware | Kaggle — 4 CPU cores, 30 GB RAM, **no GPU** |
| Labels | One-hot via `to_categorical(Y, 2)` |
| Shuffling | 10 rounds with `random_state=0..9` before split |

---

## Reported Results

| Metric | Value |
|--------|-------|
| Training accuracy | 99.05% |
| **Test accuracy** | **94.14%** |
| Precision | 94.1% |
| Recall | 94.07% |
| Inference time (test set) | 52 seconds |
| Model file size (claimed) | 277 MB |

### Confusion Matrix (percentages)

|  | Pred. Forged | Pred. Authentic |
|---|---|---|
| **True Forged** | 90% (TP) | 10% (FN) |
| **True Authentic** | 4.5% (FP) | 95.5% (TN) |

---

## Ablation Study (Table 4)

| Model | Architecture | Train Acc | Test Acc |
|-------|-------------|-----------|----------|
| Model 1 | 4× (Conv → MaxPool), Flatten, Dense, Dropout, Dense | 98.55% | 80.37% |
| **Model 2** | **4× (Conv → BatchNorm → MaxPool), Dropout, Flatten, Dense, Dropout, Dense** | **96.66%** | **88.9%** |
| Model 3 | 4× (Conv → MaxPool → BatchNorm), Dropout, Flatten, Dense, Dropout, Dense | 90.18% | 86.8% |
| Model 4 | Mixed Conv/BatchNorm blocks, Flatten, Dense, Dropout, Dense | 96.76% | 85% |
| **Proposed** | **2× Conv, MaxPool, Dropout, Flatten, Dense, Dropout, Dense** | **99.05%** | **94.14%** |

**Key finding:** The simplest, shallowest model (no BatchNorm) outperformed all deeper alternatives by 5–14% on test accuracy.

---

## Paper Inconsistencies & Omissions

### Inconsistencies

1. **Dropout rate contradiction** — Prose says "dropout probability of 0.25" globally; Algorithm 1 and code use 0.25 *then* 0.5 (two different rates)
2. **Model size: 277 MB is implausible** — 24.2M params × 4 bytes ≈ 92 MB; 277 MB is 3× expected (likely includes optimizer state or is an error)
3. **Sigmoid on 2-output + BCE** — Anti-pattern: should be softmax + categorical_crossentropy OR single sigmoid + binary_crossentropy. Current setup trains outputs independently (multi-label formulation)
4. **Val/test split not clearly disclosed** — Paper obscures the 80/16/4 actual split behind vague "80/20" language
5. **Confusion matrix vs metrics mismatch** — With class imbalance (more Au than Tp), the reported 94.14% accuracy doesn't obviously reconcile with 90% TP rate and 95.5% TN rate without exact class counts

### Omissions

| Missing | Impact |
|---------|--------|
| F1 score | Critical for imbalanced data |
| AUC / ROC curve | Standard classification metric |
| Data augmentation | None used or discussed |
| Cross-validation | Single split only |
| ELA quality factor in text | Only in code |
| Per-class precision/recall | Only aggregate reported |
| Class balancing/weighting | Not discussed despite ~5,600 Au vs ~3,900 Tp |
| Training/validation loss curves | Not shown |
| Statistical significance | No repeated experiments |
| Per-forgery-type results | Copy-move vs splicing not evaluated separately |
| Why TIFF excluded | Not explained |
| Parameter count | Not reported |

### Critical Bug in Reference Code

**Line 79 of `CASIA2code.py`:**
```python
random.shuffle(X)  # Shuffles X WITHOUT shuffling Y
```

This shuffles the image list independently of the label list, **destroying the image-label alignment**. The subsequent `sklearn.utils.shuffle(X, Y)` loops (lines 100–101) re-pair them randomly, but the damage is already done — the model trains on randomly assigned labels. If the paper's 94.14% accuracy was achieved with this bug present, the result is suspect.
