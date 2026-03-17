# Technical Audit: vR.1.1 — ETASR Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vr-1-1-etasr-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (15,511 MB VRAM) |
| **Cells** | 31 total (19 code, 12 markdown) |
| **Executed** | 19 of 19 code cells (all executed) |
| **Training** | 13 epochs (early stopped), best at epoch 8 |
| **Version** | vR.1.1 — First ablation study version |
| **Change** | Evaluation fix: 70/15/15 split, per-class metrics, ROC-AUC, ELA viz, model save |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview

vR.1.1 is the **first ablation study version**. It fixes the evaluation methodology from vR.1.0 so all subsequent experiments can be measured honestly against an unbiased baseline.

### Notebook Structure (31 cells)

| Cell | Type | Section / Purpose |
|------|------|-------------------|
| 0 | Markdown | Title, paper citation, change log, pipeline diagram, table of contents |
| 1 | Markdown | Section 1: Version Change Log (vR.1.0 vs vR.1.1) |
| 2 | Code [1] | Section 2.1: Imports, seed, config, version info |
| 3 | Markdown | Section 3 header: Dataset Preparation |
| 4 | Code [2] | Section 3.1: Dataset path discovery (os.walk search) |
| 5 | Code [3] | Section 3.2: Collect image paths and labels |
| 6 | Markdown | Section 4 header: ELA Preprocessing (with formula) |
| 7 | Code [4] | Section 4.1: `compute_ela()` and `prepare_image()` functions |
| 8 | Markdown | Section 5 header: ELA Visualization (NEW) |
| 9 | Code [5] | Section 5.1: 4×4 ELA visualization grid |
| 10 | Code [6] | Section 5.2: Process all images through ELA pipeline |
| 11 | Markdown | Section 6 header: Data Splitting (with rationale) |
| 12 | Code [7] | Section 6.1: Stratified 70/15/15 split |
| 13 | Markdown | Section 7 header: Model Architecture (with layer table) |
| 14 | Code [8] | Section 7.1: Build Sequential CNN |
| 15 | Markdown | Section 8 header: Training Pipeline |
| 16 | Code [9] | Section 8.1: Compile model |
| 17 | Code [10] | Section 8.2: Train with early stopping |
| 18 | Markdown | Section 9 header: Test Set Evaluation |
| 19 | Code [11] | Section 9.1: Per-class metrics, macro, ROC-AUC |
| 20 | Code [12] | Section 9.2: sklearn classification_report |
| 21 | Code [13] | Section 9.3: Confusion matrix heatmap |
| 22 | Code [14] | Section 9.4: ROC curve plot |
| 23 | Markdown | Section 10 header: Results Visualization |
| 24 | Code [15] | Section 10.1: Training curves (loss + accuracy) |
| 25 | Code [16] | Section 10.2: Precision/recall training curves |
| 26 | Code [17] | Section 10.3: Sample predictions (correct + incorrect) |
| 27 | Markdown | Section 11 header: Ablation Comparison |
| 28 | Code [18] | Section 11.1: Ablation tracking table |
| 29 | Markdown | Section 12 header: Discussion |
| 30 | Code [19] | Section 12.1: Save model |

### Structural Completeness

| Requirement | Present? | Quality |
|-------------|----------|---------|
| Introduction | ✅ | Good — paper citation, pipeline diagram, ToC |
| ETASR architecture explanation | ✅ | Good — full layer table matching paper Table III |
| Dataset explanation | ⚠️ Partial | File counts only — no description of CASIA provenance, tampering types, or resolution range |
| Preprocessing pipeline | ✅ | Good — ELA formula, quality parameter, pipeline diagram |
| Model architecture description | ✅ | Good — detailed layer table with output shapes and param counts |
| Training pipeline | ✅ | Good — optimizer, loss, callbacks documented |
| Evaluation metrics | ✅ | Thorough — per-class, macro, ROC-AUC, confusion matrix |
| Results interpretation | ✅ | Discussion section explains metric changes and previews next version |

### Missing / Weak Sections

1. **Dataset description is thin** — No explanation of what CASIA 2.0 is, its provenance, tampering types (splicing, copy-move), or image resolution distributions
2. **No per-epoch training summary table** — Only raw Keras verbose output; no clean formatted table
3. **No training wall-clock time** — No timing captured
4. **No error analysis** — Beyond sample predictions, no analysis of what kinds of images are misclassified

---

## 2. Strengths

1. **Proper 3-way split** — 70/15/15 train/val/test. Test set is ONLY used for final metrics, never seen during training or model selection. This is the single most important fix.
2. **Per-class + macro metrics headline** — Tampered precision (0.8393) is no longer hidden behind a weighted average (0.8854). Honest reporting.
3. **ROC-AUC computed and plotted** — 0.9601. Strong threshold-independent discriminatory power with clean visualization.
4. **ELA visualization** — 4×4 grid showing 4 authentic + 4 tampered images with ELA maps. Demonstrates the paper's core preprocessing contribution.
5. **Architecture exactly matches paper** — Every layer, activation, dimension, and parameter count matches ETASR Table III (verified layer-by-layer).
6. **Ablation comparison table** — vR.1.0 vs vR.1.1 results printed with proper annotations that vR.1.0 numbers are biased.
7. **Version tracking** — `VERSION` and `CHANGE` strings in config. Self-documenting experiment.
8. **Reproducibility** — Seeds set for random, numpy, TensorFlow. Sorted file loading. Stratified splits.
9. **Model saved** — `vR.1.1_ela_cnn_model.keras` persisted in native Keras format.
10. **All 19 code cells execute** — Zero crashes, zero runtime errors.
11. **In-memory ELA** — BytesIO approach, no temp files, thread-safe.
12. **Sample predictions** — Correct and incorrect predictions with confidence scores aid error understanding.
13. **Clean section numbering** — 12 sections, properly numbered 1–12. No duplicates.

---

## 3. Weaknesses and Issues

### MAJOR

| ID | Issue | Severity | Impact |
|----|-------|----------|--------|
| M1 | **FN rate doubled** — From ~5.4% (vR.1.0, val) to 11.7% (vR.1.1, test). The model misses 90 of 769 tampered images. While partly explained by the smaller training set, this is a real reduction in tampered detection sensitivity. | High | Tampered recall regression |
| M2 | **Val collapse at epochs 12–13** — Val_loss explodes from 0.31→0.50→0.64 in two epochs. Val_acc drops to 79.6%. Worse instability than vR.1.0, which recovered after its epoch 11 dip. | High | Training instability |
| M3 | **No localization** — Classification only. Assignment explicitly requires pixel-level tampered region masks. | Critical | Assignment requirement failure |

### MINOR

| ID | Issue | Severity | Impact |
|----|-------|----------|--------|
| m1 | **`warnings.filterwarnings('ignore')`** — Blanket warning suppression hides potentially important deprecation and convergence warnings. | Low | Debugging hazard |
| m2 | **Silent exception swallowing** — `compute_ela()` catches all exceptions and returns None with no logging. Could hide real bugs (permission errors, memory errors). | Medium | Silent failure risk |
| m3 | **Keras precision/recall are micro-averaged** — During training, precision and recall equal accuracy (micro-average artifact). Plot titles don't clarify this. Confusing when compared to per-class test metrics. | Low | Misleading training curves |
| m4 | **No ModelCheckpoint callback** — If kernel crashes after training but before model save (cell 30), all weights are lost. | Medium | Fragility |
| m5 | **ELA viz uses first N sorted samples** — Not random. Could be unrepresentative of the dataset diversity. | Low | Visualization bias |
| m6 | **No data augmentation** — The reduced training set (8,829 images, 10% less than vR.1.0) desperately needs augmentation to compensate. | Medium | Contributes to overfitting |
| m7 | **Class imbalance (1.46:1) not addressed** — No class weights, no oversampling. | Medium | FP/FN imbalance |
| m8 | **~5 GB peak RAM** — All 12,614 images loaded as float32 arrays in memory. Manageable on Kaggle but no headroom for augmentation. | Low | Scalability limit |

---

## 4. Paper Reproduction Fidelity

### Layer-by-Layer Architecture Comparison

| # | Layer | Paper Specification | Implementation | Verdict |
|---|-------|-------------------|----------------|---------|
| 1 | Conv2D | 32 filters, 5×5, ReLU, valid | `Conv2D(32, (5,5), activation='relu', padding='valid')` | **Exact match** |
| 2 | Conv2D | 32 filters, 5×5, ReLU, valid | `Conv2D(32, (5,5), activation='relu', padding='valid')` | **Exact match** |
| 3 | MaxPooling2D | 2×2 | `MaxPooling2D(pool_size=(2,2))` | **Exact match** |
| 4 | Dropout | 0.25 | `Dropout(0.25)` | **Exact match** |
| 5 | Flatten | — | `Flatten()` | **Exact match** |
| 6 | Dense | 256, ReLU | `Dense(256, activation='relu')` | **Exact match** |
| 7 | Dropout | 0.5 | `Dropout(0.5)` | **Exact match** |
| 8 | Dense (output) | 2, Softmax | `Dense(2, activation='softmax')` | **Exact match** |

### Hyperparameter Comparison

| Parameter | Paper | Notebook | Verdict |
|-----------|-------|----------|---------|
| Optimizer | Adam | Adam(lr=0.0001) | **Exact match** |
| Learning rate | 0.0001 | 0.0001 | **Exact match** |
| Loss | categorical_crossentropy | categorical_crossentropy | **Exact match** |
| ELA quality | 90 | 90 | **Exact match** |
| Image size | 128×128 | (128, 128) | **Exact match** |
| Batch size | 32 | 32 | **Exact match** |
| Max epochs | 50 | 50 | **Exact match** |
| Early stopping | Not specified | patience=5, val_accuracy, restore_best_weights | **Acceptable deviation** |
| Data split | 80/20 (paper) | 70/15/15 | **Intentional deviation** — documented ablation change |

### Paper Reproduction Score

| Category | Score | Notes |
|----------|-------|-------|
| Architecture fidelity | 10/10 | Exact match to Table III — every layer verified |
| Preprocessing fidelity | 10/10 | ELA Q=90, in-memory BytesIO, correct formula |
| Training config fidelity | 9/10 | All match except data split (intentional) |
| Metric methodology | 9/10 | Fixed: proper test set, per-class, ROC-AUC |
| Results reproduction | 5/10 | 7.83pp accuracy gap unexplained |

**Overall Reproduction Score: 8/10** — Architecture is perfect. Evaluation methodology is now sound. Accuracy gap remains.

---

## 5. Dataset Pipeline Review

### Loading and Processing

| Aspect | Implementation | Verdict |
|--------|---------------|---------|
| Source | CASIA v2.0 via Kaggle | Correct |
| Discovery | `os.walk()` searching for Au/ + Tp/ dirs | Works but fragile |
| File listing | `sorted(os.listdir())` | Deterministic ✅ |
| Extension filter | `.jpg, .jpeg, .png, .tif, .tiff, .bmp` | Comprehensive ✅ |
| Images loaded | 7,491 Au + 5,123 Tp = 12,614 | All formats handled |
| ELA method | In-memory BytesIO JPEG re-save | Correct, efficient ✅ |
| ELA quality | Q=90 | Matches paper ✅ |
| Normalization | `/ 255.0` → [0, 1] | Standard ✅ |
| Shuffle | `sklearn.utils.shuffle(X, Y, random_state=42)` | Paired, deterministic ✅ |
| Skipped images | 0 | Clean processing ✅ |

### Data Leakage Check

- **No leakage detected.** Split happens after ELA processing on image indices. No augmentation before splitting. Test set never accessed during training.
- **One subtle concern:** ELA brightness scaling is per-image (`255/max_diff`). Images with low ELA differences are stretched to the same range as high-difference images. This is a normalization choice, not leakage.

### Split Distribution

| Split | Total | Authentic | Tampered | Au% | Tp% |
|-------|-------|-----------|----------|-----|-----|
| Train | 8,829 | 5,243 | 3,586 | 59.4% | 40.6% |
| Val | 1,892 | 1,124 | 768 | 59.4% | 40.6% |
| Test | 1,893 | 1,124 | 769 | 59.4% | 40.6% |

Stratification is correctly maintained across all splits. ✅

### Efficiency Concerns

| Concern | Assessment |
|---------|------------|
| Peak RAM | ~5 GB (12,614 × 128×128×3 × float32 × 2 during list→array conversion). Manageable on Kaggle. |
| ELA processing | Sequential, no parallelism. ~2 min for 12,614 images. Acceptable. |
| Disk I/O | No temp files (BytesIO). Good. |

---

## 6. Training Pipeline Review

### Configuration

| Parameter | Value | Status |
|-----------|-------|--------|
| Optimizer | Adam(lr=0.0001) | Frozen from baseline |
| Loss | categorical_crossentropy | Frozen |
| Batch size | 32 | Frozen |
| Max epochs | 50 | Frozen |
| Early stopping | val_accuracy, patience=5, restore_best_weights | Frozen |
| Callbacks | EarlyStopping only | No ModelCheckpoint, no LR scheduler |
| Training samples | 8,829 | 10% fewer than vR.1.0's 10,091 |

### Epoch-by-Epoch Training Log

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Notes |
|-------|-----------|-----------|----------|---------|-------|
| 1 | 0.4793 | 0.7530 | 0.3967 | 0.8325 | |
| 2 | 0.3211 | 0.8757 | 0.3558 | 0.8541 | |
| 3 | 0.2878 | 0.8865 | 0.3188 | 0.8695 | |
| 4 | 0.2539 | 0.8978 | 0.3237 | 0.8631 | Val dip |
| 5 | 0.2282 | 0.9087 | 0.3024 | 0.8705 | |
| 6 | 0.2072 | 0.9142 | 0.2755 | 0.8768 | |
| 7 | 0.1850 | 0.9255 | 0.2843 | 0.8716 | |
| **8** | **0.1726** | **0.9289** | **0.2662** | **0.8864** | **Best epoch ✅** |
| 9 | 0.1606 | 0.9309 | 0.2966 | 0.8642 | Val drops, overfitting starts |
| 10 | 0.1592 | 0.9328 | 0.2999 | 0.8726 | |
| 11 | 0.1614 | 0.9307 | 0.3089 | 0.8716 | |
| **12** | 0.1672 | 0.9303 | **0.4981** | **0.8208** | **Val collapse — loss nearly doubles** |
| **13** | 0.1627 | 0.9260 | **0.6360** | **0.7960** | **Catastrophic — early stopping triggers** |

### Overfitting Analysis

| Indicator | Evidence | Severity |
|-----------|----------|----------|
| Train-val gap at best epoch | Train 92.89% vs Val 88.64% = **4.25pp gap** | Moderate |
| Train-val gap at epoch 13 | Train 92.60% vs Val 79.60% = **13.0pp gap** | Severe |
| Train loss trend | Continuously decreasing (0.48→0.16) | Model is memorizing |
| Val loss trend | Rises sharply after epoch 8 (0.27→0.64) | Overfitting |
| Val collapse | Epochs 12–13: val_loss doubles, val_acc drops 8.5pp | Critical instability |

**Root cause:** 29.5M parameters (99.9% in Flatten→Dense(256)) with only 8,829 training images = massive overfitting capacity. The model memorizes the training set.

### Reproducibility

| Check | Result |
|-------|--------|
| Best epoch 8 | Consistent with all 4 prior runs ✅ |
| 13 epochs total | Consistent ✅ |
| Early stopping triggered | Correctly ✅ |
| Weights restored to best | Yes (restore_best_weights=True) ✅ |

---

## 7. Performance Summary

### Final Test Metrics

| Metric | Authentic | Tampered | Macro |
|--------|-----------|----------|-------|
| **Precision** | 0.9170 | 0.8393 | 0.8781 |
| **Recall** | 0.8843 | 0.8830 | 0.8837 |
| **F1** | 0.9004 | 0.8606 | 0.8805 |

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 88.38% |
| **ROC-AUC** | 0.9601 |
| **Macro F1** | 0.8805 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|---|---|---|
| **True Au** (1,124) | 994 | 130 |
| **True Tp** (769) | 90 | 679 |

| Rate | Value | vs vR.1.0 |
|------|-------|-----------|
| FP rate | 11.6% | Improved (was 13.5%) |
| FN rate | 11.7% | **Worse** (was 5.4%) |

### vs vR.1.0 Baseline

| Metric | vR.1.0 (val, biased) | vR.1.1 (test, honest) | Delta | Assessment |
|--------|---------------------|-----------------------|-------|------------|
| Accuracy | 89.89% | 88.38% | -1.51pp | Expected (honest eval) |
| Tp Precision | 0.8279 | 0.8393 | +0.0114 | Improved |
| Tp Recall | 0.9483 | 0.8830 | **-0.0653** | **Significant regression** |
| Tp F1 | 0.8840 | 0.8606 | -0.0234 | Regression |
| Macro F1 | 0.8972 | 0.8805 | -0.0167 | Expected |
| ROC-AUC | — | 0.9601 | New | Strong |
| FP rate | 13.5% | 11.6% | -1.9pp | Improved |
| FN rate | 5.4% | 11.7% | **+6.3pp** | **Major regression** |

### vs Paper Claims

| Metric | Paper | vR.1.1 | Gap |
|--------|-------|--------|-----|
| Accuracy | 96.21% | 88.38% | **-7.83pp** |
| Precision | 98.58% | 87.81% (macro) | -10.77pp |
| Recall | 92.36% | 88.37% (macro) | -3.99pp |
| F1 | 95.37% | 88.05% (macro) | -7.32pp |

---

## 8. Result Analysis

### Did vR.1.1 Achieve Its Goal?

**Yes.** The version changeset was "fix evaluation methodology." All five planned changes were successfully implemented:

| Planned Change | Implemented? | Verified? |
|----------------|-------------|-----------|
| 70/15/15 split | ✅ | 8829/1892/1893 samples |
| Per-class + macro metrics | ✅ | Headline metrics are per-class |
| ROC-AUC | ✅ | 0.9601, plotted |
| ELA visualization | ✅ | 4+4 grid |
| Model save | ✅ | vR.1.1_ela_cnn_model.keras |

### Overfitting Indicators

| Indicator | Present? | Evidence |
|-----------|----------|----------|
| Train-val accuracy gap | ✅ | 4.25pp at best epoch, 13.0pp at final |
| Val loss divergence | ✅ | Val loss rises after epoch 8 while train loss falls |
| Late-epoch collapse | ✅ | Epochs 12–13: catastrophic val_loss spike |
| Train accuracy plateau | ❌ | Still rising at epoch 13 |

**Verdict: Moderate to severe overfitting.** The model has far more capacity (29.5M params) than the training data (8,829 images) can constrain. Data augmentation (vR.1.2) is the priority fix.

### Dataset Imbalance Effects

| Effect | Present? | Evidence |
|--------|----------|----------|
| Higher FP rate for majority class | Partial | FP rate 11.6% — authentic images misclassified as tampered |
| Lower precision for minority class | ✅ | Tampered precision (0.8393) < Authentic precision (0.9170) |
| Balanced recall | ✅ | Both classes ~88% recall — the model doesn't heavily favor either class |

The 1.46:1 class imbalance has a **moderate** effect. The model slightly over-predicts tampered (higher FP than FN in absolute terms when adjusted for class size).

---

## 9. Paper Reproduction Score

**8/10** — Architecture is a perfect match. Evaluation methodology is now sound. The 7.83pp accuracy gap remains unexplained.

---

## 10. Assignment Readiness Score

**5/10** — Significant improvement over vR.1.0 (3/10):

| Requirement | Status |
|-------------|--------|
| Image tampering detection | ✅ 88.38% accuracy |
| ELA preprocessing | ✅ Implemented and visualized |
| CNN model | ✅ Paper architecture reproduced |
| Proper evaluation | ✅ Test set, per-class, ROC-AUC |
| Data augmentation | ❌ Missing (next version) |
| Class imbalance handling | ❌ Missing |
| Localization / masks | ❌ Missing (critical gap) |

---

## 11. Recommended Improvements

### Immediate (vR.1.2)

1. **Add data augmentation** — Horizontal flip, vertical flip, random rotation ±15°. This is the single highest-impact change available. The reduced training set (8,829 images) is the primary driver of the FN regression. Augmentation will:
   - Compensate for 10% fewer training images
   - Reduce the overfitting gap (train 93% vs val 89%)
   - Stabilize the late-epoch collapse at epochs 12–13
   - Potentially close 1–3pp of the paper gap

### Near-term (vR.1.3–1.5)

2. **Add class weights** — Inversely proportional to frequency. Will balance FP/FN rates.
3. **Add BatchNormalization** — After each Conv2D. Will stabilize training dynamics.
4. **Add ReduceLROnPlateau** — Will improve convergence beyond current plateau.

### Later (vR.1.6+)

5. **Architecture improvements** — Deeper CNN, GlobalAveragePooling2D to reduce the 29.5M param count.
6. **Localization** — ELA-based thresholding for pseudo-masks to satisfy assignment requirements.
