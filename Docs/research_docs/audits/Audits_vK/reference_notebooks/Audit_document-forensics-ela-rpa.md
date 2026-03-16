# Audit: Document Forensics Using ELA and RPA

**Auditor:** Principal AI Engineer
**Date:** 2026-03-14
**File:** `document-forensics-using-ela-and-rpa.ipynb` (16 KB)

---

## Notebook Overview

This notebook implements a **non-neural-network** approach to image tampering detection using Error Level Analysis (ELA) and Residual Pixel Analysis (RPA). It is purely heuristic-based: compute the standard deviation of an ELA image and compare it against a threshold. No model is trained — classification is a simple `if std > threshold` check.

| Attribute | Value |
|---|---|
| Cell Count | 21 (10 code, 10 markdown, 1 empty) |
| Model | None — threshold-based heuristic |
| Trainable Parameters | 0 |
| Dataset | CASIA 2.0 (`Au/` + `Tp/`) |
| Task | Binary classification (authentic vs tampered) |

---

## Dataset Pipeline Review

| Property | Value |
|---|---|
| Dataset | CASIA 2.0 |
| Authentic | `Au/` directory |
| Tampered | `Tp/` directory |
| Split | 80/20 train-test with `stratify=y` |
| Train evaluation | First 1,000 samples only |
| Test evaluation | First 200 samples only |

**Critical:** The notebook only evaluates on a subset of the data (1,000 train, 200 test) rather than the full dataset. Results are therefore not representative.

---

## Model Architecture Review

There is no neural network. The "model" is a hand-crafted pipeline:

1. **ELA Computation:** Resize image to a tiny scale (10×10, 20×20, or 30×30), save as JPEG at a given quality, compute `cv2.absdiff` between original and recompressed, convert to grayscale.
2. **RPA Classification:** Compute `np.std()` of the ELA image. If standard deviation exceeds a threshold, classify as "Tampered".

**Grid search parameters:**
- Scale: {10, 20, 30}
- JPEG Quality: {50, 70, 90}
- Threshold: {1, 3, 5, 7, 9, 11}

Best configuration: `scale=30, quality=90, threshold=9`

---

## Training Pipeline Review

No training occurs. The grid search is an exhaustive evaluation of all parameter combinations on the training subset. The "optimal" parameters are selected by maximum F1 score.

---

## Evaluation Metrics Review

| Set | Metric | Value |
|---|---|---|
| Train (1,000 samples) | F1 | 0.7508 |
| Train (1,000 samples) | Accuracy | 0.603 |
| Test (200 samples) | F1 | 0.7082 |
| Test (200 samples) | Accuracy | **0.555** |

The accuracy of 55.5% is barely above random chance (50%). The F1 score of 0.71 looks better only because of the class imbalance: the model over-predicts "tampered" (high recall, low precision).

---

## Visualization Assessment

No visualizations. The notebook outputs only text metrics from grid search evaluation.

---

## Engineering Quality Assessment

| Criterion | Rating | Notes |
|---|---|---|
| Code Quality | **Poor** | Writes temp files to disk, `except: pass` error handling |
| Reproducibility | **Fair** | Fixed random_state=42 for split |
| Documentation | **Minimal** | Markdown cells explain ELA concept but not RPA |
| Error Handling | **Terrible** | `except: pass` silently swallows all errors |
| Evaluation | **Flawed** | Only subsets evaluated, not full data |
| Scalability | **Poor** | Writes `temp.jpg` to disk on every image — not parallelizable |

---

## Strengths

1. **Educational value:** Demonstrates ELA concept without requiring deep learning infrastructure
2. **Simple and interpretable:** The threshold-based approach is fully explainable
3. **Grid search:** Systematically explores the parameter space rather than picking arbitrary values

---

## Weaknesses

1. **55.5% test accuracy** — essentially random guessing
2. **Tiny-scale ELA is inaccurate:** Resizing to 10×10/20×20/30×30 destroys virtually all spatial information before ELA computation
3. **Subset evaluation:** Only 1,000/200 samples evaluated, not the full dataset
4. **No localization:** Only binary classification, no pixel-level detection
5. **Silent error handling:** `except: pass` hides file I/O failures, corrupt images, etc.
6. **Temp file disk I/O:** Writes a temporary JPEG file for every single image evaluation

---

## Critical Issues

1. **ELA on 10×10 images is meaningless.** Error Level Analysis works by detecting JPEG compression inconsistencies at the pixel level. Resizing to 10×10 collapses all spatial structure — the ELA of a 10×10 thumbnail tells you nothing about tampering in the original image.

2. **Confusion matrix logic is inverted.** The code counts `cnt == 0 and yTrain == 0` as "True Positive" (genuine correctly predicted as genuine), but standard convention defines TP as "tampered correctly detected as tampered." True negatives are never computed.

3. **Partial evaluation.** Grid search on 1,000 samples and testing on 200 samples is statistically unreliable for a dataset of thousands of images.

---

## Suggested Improvements

1. Compute ELA at full resolution (or at least 128×128) instead of 10×10
2. Use proper ELA: resave at quality=90, compute absolute difference at full resolution
3. Evaluate on the FULL dataset, not subsets
4. Replace the threshold heuristic with a learned classifier (even logistic regression on ELA features would be better)
5. Fix the confusion matrix TP/TN/FP/FN definitions
6. Replace `except: pass` with proper error handling

---

## Roast Section

This notebook asks the age-old question: "What if we squished every image down to 10 pixels wide, saved it as a JPEG, and then classified it based on whether the pixel variance was above 9?" The answer is: you get 55.5% accuracy, which is what you'd get by flipping a coin but with extra steps.

The ELA computation is the forensic equivalent of examining a crime scene through a telescope held backwards. Error Level Analysis requires comparing pixel-level compression artifacts — information that is completely obliterated when you resize the image to 10×10. At that scale, you have fewer pixels than a QR code, and the "analysis" boils down to checking if the thumbnail's compression looks noisy.

The `except: pass` pattern ensures that when things go wrong — and they will, with disk temp files on every evaluation — nobody will ever know. Files might fail to open, JPEGs might fail to save, images might be corrupt — all silently ignored, all contributing to the 55.5% accuracy that the notebook proudly presents.

The F1 score of 0.71 is a statistical mirage. When your model labels everything as "tampered" to get high recall, F1 looks respectable even though accuracy is at chance level. It's the same trick as a fire alarm that goes off every 5 minutes — great recall, terrible precision.

**Bottom line:** This notebook demonstrates what ELA is conceptually, but the implementation is so degraded (tiny-scale processing, partial evaluation, broken metrics) that it has negative practical value. Use it as a "what not to do" reference.
