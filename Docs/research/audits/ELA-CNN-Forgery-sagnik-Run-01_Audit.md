# ELA CNN Image Forgery Detection (Sagnik Dataset) --- Run-01 Audit

## Overview

| Field | Value |
|-------|-------|
| **Name** | ELA CNN Image Forgery Detection on Sagnik Data |
| **Run** | Run-01 |
| **Track** | Standalone Research Paper Architecture |
| **Architecture** | 3-block CNN (Conv64+BN+Pool -> Conv128+BN+Pool -> Conv256+BN+Pool -> Dense512 -> Dense2) |
| **Parent** | None (standalone, deeper variant of paper architecture) |
| **GPU** | Tesla P100-PCIE-16GB |
| **Framework** | TensorFlow 2.19.0 / Keras |

---

## Experiment Goal

Apply the deeper 3-block CNN architecture (with BatchNormalization, resembling the research paper's ablation Model 2) to the Sagnik-hosted CASIA v2.0 dataset for binary classification of ELA images as authentic or tampered.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| IMAGE_SIZE | 150x150 |
| ELA_QUALITY | 90 |
| BATCH_SIZE | 8 |
| EPOCHS | 40 (stopped at 6) |
| Optimizer | Adam (lr=0.0001) |
| Loss | Binary Crossentropy |
| Early Stopping | patience=2 on val_accuracy, restore_best_weights=True |
| Split | 70/15/15 |
| Total Params | 38,253,954 |

---

## Training Summary

| Metric | Value |
|--------|-------|
| Epochs completed | 6 / 40 |
| Best epoch | 4 |
| Best val loss | 0.0023 |
| Best val accuracy | 100.00% |
| Early stopping | Yes (epoch 6) |

Training converged to near-perfect accuracy by epoch 1 (94.97% train, 99.79% val). By epoch 4, validation accuracy reached 100%.

---

## Test Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.95%** |
| Test Loss | 0.0071 |
| Precision (weighted) | 0.9995 |
| Recall (weighted) | 0.9995 |
| F1 (weighted) | 0.9995 |

### Confusion Matrix

| | Pred Tampered | Pred Authentic |
|---|---|---|
| **Tampered** | 776 | 1 |
| **Authentic** | 0 | 1116 |

**Only 1 misclassification** out of 1,893 test images.

---

## Strengths

1. **Near-perfect accuracy** (99.95%) with only 6 epochs of training
2. **Clean execution** --- no errors, all cells completed
3. **Comprehensive evaluation** including confusion matrix and per-class metrics
4. **Uses find_dataset() auto-discovery** for Kaggle path compatibility

---

## Weaknesses

1. **No localization** --- classification only, cannot produce pixel-level masks
2. **No AUC/ROC curve** reported
3. **Architecture does NOT match the paper** --- implements ablation Model 2, not the proposed model
4. **Low resolution** (150x150) discards spatial detail needed for forensics
5. **Suspiciously high accuracy** suggests dataset contamination or data leakage

---

## Major Issues

### 1. DATA LEAKAGE (CRITICAL)

99.95% accuracy on a tampering detection task is scientifically implausible. The research paper itself only achieved 94.14% on the same dataset with the proposed simpler architecture. Possible causes:

- **Mask images loaded as input:** The dataset path references a `MASK` directory. If mask images (which visually encode the ground truth) were loaded instead of photographs, the model is learning to distinguish blank images from masks --- not detecting tampering.
- **EXIF/filename leakage:** Different naming patterns for Au/ and Tp/ folders could allow the model to solve the task from metadata rather than image content.
- **Duplicate images across splits:** Without deduplication, train/test overlap would inflate accuracy.

The other Sagnik-dataset notebook (`casia2-ela-cnn-sagnik`) achieved 100% accuracy with the simpler architecture, reinforcing the data leak hypothesis.

### 2. Architecture Mismatch

This notebook claims to implement the paper's architecture but actually implements a deeper 3-block CNN with BatchNorm (Conv64/128/256 + BN + Dense512). The paper's proposed model is 2xConv32(5x5) + Dense150. This is the paper's ablation Model 2, which scored only 88.9% in the paper --- not 99.95%.

---

## Minor Issues

1. Early stopping patience=2 is extremely aggressive
2. No reproducibility seeds set (no `np.random.seed` or `tf.random.set_seed`)
3. Keras HDF5 format deprecation warning
4. No training curves plotted

---

## Roast

**As a strict conference reviewer:**

This experiment achieves 99.95% accuracy on a task where the state-of-the-art (including the reference paper's own result) is 94.14%. This is not a breakthrough --- it is a bug.

The Sagnik-hosted CASIA v2.0 dataset on Kaggle appears to be structurally compromised. The `MASK` directory path in the dataset loader strongly suggests mask images are being included in the training data, making the classification task trivially solvable. The fact that a completely different architecture on the same dataset achieves 100% accuracy confirms this: it is not the model that is special --- it is the dataset that is leaking.

This result is **scientifically invalid** and must not be cited as model performance. If this were submitted to a conference, it would receive an immediate desk rejection for insufficient data integrity validation.

The architecture mismatch (claiming to implement the paper while actually implementing the ablation model) is an additional credibility issue, though secondary to the data leak.

**Verdict:** Invalid result due to probable data leakage. Cannot be used for any scientific conclusion about model capability.

---

## Assignment Alignment

| Deliverable | Status |
|-------------|--------|
| Pixel-level prediction | **MISSING** (classification only) |
| GT mask comparison | **MISSING** |
| Standard metrics | Partial (accuracy, P/R/F1; no AUC/ROC) |
| Visual results | **MISSING** |
| Model weights saved | Yes (.h5 format) |
| Single notebook | Yes |
| Localization masks | **MISSING** |

---

## Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture Implementation | 10 | /20 | Mismatched from paper; functional but wrong claim |
| Dataset Handling | 3 | /15 | Probable data leak; no dedup check; no JPEG-only filter |
| Experimental Methodology | 5 | /20 | Invalid results; no data integrity validation |
| Evaluation Quality | 8 | /20 | Basic metrics only; no AUC/ROC; no leak investigation |
| Documentation Quality | 6 | /15 | Minimal documentation; architecture mismatch not noted |
| Assignment Alignment | 2 | /10 | No localization; classification only |

### **Final Score: 34/100**
