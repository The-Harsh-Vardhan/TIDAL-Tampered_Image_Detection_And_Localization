# Technical Audit: CASIA2-ELA-CNN (sagnik) — Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `casia2-ela-cnn-with-sagnik-dataset-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 40 epochs (no early stopping) |
| **Version** | Paper Architecture on Sagnik Dataset |
| **Parent** | None (standalone — research paper reproduction) |
| **Change** | Run paper architecture on alternative dataset (sagnikkayalcse52) |
| **Status** | **FULLY EXECUTED — CRITICAL DATA INTEGRITY ISSUE** |

---

## 1. Notebook Overview & Experiment Goal

This notebook runs the paper's CNN architecture on the Sagnik dataset (sagnikkayalcse52) instead of the standard divg07 dataset. The goal was to test whether the alternative dataset produces different results.

**Result:** Test accuracy reaches **100.00%** — a perfect score that is **SCIENTIFICALLY INVALID**. This is a **DATA LEAK**, not a breakthrough.

**THIS RUN MUST NOT BE CITED, USED, OR REFERENCED AS A VALID RESULT.**

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 2× Conv2D(32, 5×5) + MaxPool2D(2×2) + Flatten + Dense(150) + Dense(2, sigmoid) |
| Input | ELA 150×150, scaled to [0,1] |
| Loss | binary_crossentropy |
| Optimizer | Adam (lr=0.0001) |
| Batch size | 8 |
| Epochs | 40 (fixed — NO early stopping) |
| Dataset | CASIA v2.0 (**sagnikkayalcse52** — DIFFERENT from divg07) |
| Train/Val/Test split | 70/15/15 |

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **Exposed a critical dataset integrity issue** | 100% accuracy revealed the data leak |

There are no other meaningful strengths. The entire run is compromised.

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **CRITICAL** | 100% test accuracy = DATA LEAK — scientifically invalid |
| W2 | **CRITICAL** | Dataset likely contains mask images as inputs — trivial classification |
| W3 | **CRITICAL** | X data range [0.0, 0.76] (vs [0.0, 1.0] for divg07) — confirms different data |
| W4 | **MAJOR** | No data validation — mask images should have been detected and excluded |
| W5 | **MAJOR** | No early stopping (same issue as divg07 run) |
| W6 | MODERATE | Results could mislead if cited without the leak disclosure |

---

## 5. Major Issues

### 5.1 CRITICAL: Data Leak — 100% Accuracy (W1, W2, W3)

A 100% test accuracy on a real-world image forgery detection task is physically impossible. The achievable ceiling for CASIA 2.0 with ELA+CNN is approximately 90-95%.

**Evidence of data leak:**

1. **X data range:** The Sagnik dataset produces X values in [0.0, 0.76], while divg07 produces [0.0, 1.0]. The images are fundamentally different.

2. **Dataset path structure:** The Sagnik dataset path contains references to "MASK" directories. The data loader may be loading ground truth mask images instead of original photographs.

3. **Perfect accuracy from early epochs:** The model reaches near-100% accuracy within the first few epochs — consistent with classifying masks vs photographs.

4. **Training loss near zero:** Both training and test losses converge to effectively zero.

**Mechanism:** If the "tampered" class contains ground truth masks (binary images showing tampered regions) instead of tampered photographs, the CNN can trivially distinguish between them based on image statistics alone.

### 5.2 MAJOR: No Data Validation (W4)

The notebook contains no data validation step to verify that input images are actual photographs. A simple check of image statistics would have revealed the anomaly.

---

## 6. Minor Issues

None — all issues are major or critical.

---

## 7. Training Summary

| Epoch | Train Acc | Test Acc | Train Loss | Test Loss |
|-------|----------|---------|-----------|----------|
| 1 | ~99.5% | ~99.8% | ~0.02 | ~0.01 |
| 10 | 100.0% | 100.0% | ~0.001 | ~0.001 |
| 20 | 100.0% | 100.0% | ~0.0001 | ~0.0001 |
| 40 (final) | 100.0% | 100.0% | ~0.0000 | ~0.0000 |

**This training curve is a smoking gun for data leakage.**

---

## 8. Test Results

### Image-Level Classification

| Metric | Value | Assessment |
|--------|-------|------------|
| Test Accuracy | 100.00% | **INVALID — DATA LEAK** |
| Precision | 1.0000 | **INVALID** |
| Recall | 1.0000 | **INVALID** |
| F1 Score | 1.0000 | **INVALID** |

### Pixel-Level Localization

**NOT AVAILABLE** — classification-only model.

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"100% accuracy. Perfect precision. Perfect recall. Perfect F1. The authors have done what generations of computer vision researchers could not — created a perfect image forgery detector.

Or they loaded the answer key as input data.

The X-data range of [0.0, 0.76] is the smoking gun. Normal photographs produce ELA values across the full [0, 1] range. A restricted range indicates the input images are not photographs — they're likely ground truth masks. A dataset path containing 'MASK' really should have been the first clue.

I have three recommendations: (1) retract this result immediately, (2) add a data validation cell to all future notebooks, and (3) frame this as a cautionary tale in the methods section. 'We accidentally trained on the answer key' is a more common ML mistake than anyone admits."

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction | **FAIL** | Classification only — no masks |
| Valid evaluation metrics | **FAIL** | All metrics invalid due to data leak |
| Train/val/test split | **FAIL** | Valid split, but on compromised data |
| Scientific validity | **FAIL** | Data leak invalidates all conclusions |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 10 | 15 | Same paper arch — not the problem here |
| Dataset | 3 | 15 | DATA LEAK — wrong images used as input (-12) |
| Methodology | 3 | 20 | No data validation (-7), no early stopping (-5), leak not caught (-5) |
| Evaluation | 5 | 20 | Metrics computed but scientifically meaningless |
| Documentation | 5 | 15 | Notebook runs but doesn't flag the impossibility of 100% |
| Assignment Alignment | 2 | 15 | No localization, no valid metrics |
| **Total** | **28** | **100** | |

---

## 12. Final Verdict: **INVALID — DATA LEAK** — Score: 28/100

**This run is scientifically invalid.** The 100% test accuracy is caused by a data leak in the Sagnik dataset. All metrics must be discarded.

### Lessons

1. **Always validate input data** — check X statistics, visualize samples, verify image sources
2. **Question impossible results** — 100% accuracy on a real-world task is always suspicious
3. **Dataset source matters** — the same "CASIA 2.0" label can refer to very different data
4. **Never cite leaked results** — even as a "best case" comparison
