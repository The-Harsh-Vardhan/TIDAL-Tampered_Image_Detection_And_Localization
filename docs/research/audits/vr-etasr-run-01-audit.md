# Technical Audit: vR.ETASR (Run 01)

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-14 |
| **Notebook** | `vr-etasr-image-detection-and-localisation-run-01.ipynb` |
| **Paper** | ETASR_9593 — "Enhanced Image Tampering Detection using ELA and a CNN" (Gorle & Guttavelli, 2025) |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB (16 GB VRAM) |
| **Cells** | 29 total (15 code, 14 markdown) |
| **Executed** | 14 of 15 code cells (1 save cell commented out) |
| **Training** | 13 epochs (early stopped at patience=5), best at epoch 8 |
| **Status** | **FULLY EXECUTED — MODEL TRAINS AND CONVERGES** |

---

## 1. Notebook Overview

This notebook implements the ETASR_9593 paper: a binary classification pipeline that applies Error Level Analysis (ELA) preprocessing to CASIA v2.0 images, then feeds the resulting 128×128×3 ELA maps into a compact 2-layer CNN for authentic/tampered classification.

### Key Results

| Metric | This Run | Paper Claims |
|--------|----------|-------------|
| Accuracy | **89.89%** | 96.21% |
| Precision | 0.9068 (weighted) | 98.58% |
| Recall | 0.8989 (weighted) | 92.36% |
| F1 Score | 0.8997 (weighted) | 95.37% |
| Epochs | 13 (best: 8) | Not specified |

**Bottom line:** The model trains, converges, and achieves 89.89% accuracy. This is a functional implementation. However, it falls 6.3 percentage points short of the paper's claimed 96.21%, and it fundamentally does not satisfy the assignment's localization requirement.

---

## 2. Notebook Structure Audit

### Section Coverage

| Section | Present | Quality |
|---------|---------|---------|
| Introduction | Yes | Good — clear, concise, states key idea |
| Research Paper Summary | Yes | Good — architecture table, training config, reported results |
| Reference Code Audit | Yes (markdown only in this run) | Good in source notebook; **missing from this executed run** (Section 3 heading exists but no audit cell) |
| Dataset Preparation | Yes | Good — auto-discovery, format support, class counts |
| ELA Preprocessing | Yes | Good — formula, explanation, implementation |
| Model Architecture | Yes | Good — architecture table, parameter counts |
| Training Pipeline | Yes | Adequate — compile, callbacks, train |
| Evaluation Metrics | Yes | Adequate — accuracy, precision, recall, F1, confusion matrix |
| Results Visualization | Yes | Adequate — training curves, precision/recall curves, sample predictions |
| Discussion & Conclusion | Yes | Adequate — limitations acknowledged |

### Missing Sections

1. **Section 3 (Reference Code Audit)** — The table of contents lists it, but the notebook jumps from Section 2 directly to Section 4. The `cell-3` markdown says "## 4. Dataset Preparation" — the numbering is broken. Section 3 content exists in the source template but was stripped from this run.
2. **No ELA visualization** — The source notebook (vR.1) has a `visualize_ela_samples()` cell showing authentic vs tampered ELA maps side-by-side. This run notebook skips it entirely. For a paper reproduction claiming ELA is the key contribution, not showing what ELA actually produces is a serious presentation gap.
3. **No hold-out test set** — Only train/val split (80/20). No separate test set. Evaluation is done on the validation set.
4. **No ROC-AUC metric** — The notebook computes accuracy, precision, recall, F1 but no ROC curve or AUC.
5. **No model save** — The save cell is commented out.

### Structure Score: 6/10

The structure is clean and readable, but missing Section 3, missing ELA visualization, and missing a proper test set are significant gaps for a submission.

---

## 3. Paper Reproduction Audit

### Preprocessing Pipeline

| Paper Specification | Implementation | Match |
|---|---|---|
| ELA at JPEG quality 90 | `ELA_QUALITY = 90` | **MATCH** |
| Pixel-wise absolute difference | `ImageChops.difference(original, resaved)` | **MATCH** |
| Brightness scaling to [0, 255] | `ImageEnhance.Brightness(ela_image).enhance(scale)` | **MATCH** |
| Resize to 128×128 | `ela.resize(target_size, Image.BILINEAR)` | **MATCH** |
| Normalize to [0, 1] | `np.array(ela_resized, dtype=np.float32) / 255.0` | **MATCH** |

**Verdict: ELA preprocessing faithfully reproduces the paper.**

### CNN Architecture

| Paper (Table III) | Implementation | Match |
|---|---|---|
| Conv2D(32, 5×5, valid, ReLU) | `Conv2D(32, (5,5), activation='relu', padding='valid')` | **MATCH** |
| Conv2D(32, 5×5, valid, ReLU) | Same | **MATCH** |
| MaxPooling2D(2×2) | `MaxPooling2D(pool_size=(2,2))` | **MATCH** |
| Dropout(0.25) | `Dropout(0.25)` | **MATCH** |
| Flatten | `Flatten()` | **MATCH** |
| Dense(256, ReLU) | `Dense(256, activation='relu')` | **MATCH** |
| Dropout(0.5) | `Dropout(0.5)` | **MATCH** |
| Dense(2, Softmax) | `Dense(2, activation='softmax')` | **MATCH** |

**Verified parameter count:** 29,520,034 (matches expected ~29.5M).

**Verdict: Architecture exactly matches the paper.**

### Training Configuration

| Paper Specification | Implementation | Match |
|---|---|---|
| Adam optimizer | `Adam(learning_rate=0.0001)` | **MATCH** |
| Learning rate 0.0001 | Same | **MATCH** |
| Categorical cross-entropy | `loss='categorical_crossentropy'` | **MATCH** |
| Batch size 32 | `BATCH_SIZE = 32` | **MATCH** |
| 80/20 train/val split | `test_size=0.2` | **MATCH** |
| Early stopping | `patience=5, monitor='val_accuracy'` | **MATCH** (paper implies but doesn't specify patience) |

### Deviations from Paper

| Deviation | Classification | Impact |
|---|---|---|
| Max epochs 50 (paper unspecified) | Acceptable | Early stopping handles this; paper doesn't specify |
| No batch normalization discussed | Minor issue | Paper doesn't mention BN either; consistent |
| `average='weighted'` for precision/recall/F1 | **MAJOR ISSUE** | Paper reports per-class or macro metrics. Weighted average inflates scores toward the majority class. The 0.9068 precision is not comparable to the paper's 98.58% per-class precision. See Section 6. |
| In-memory BytesIO for ELA instead of temp file | Acceptable | Better implementation; same result |
| Early stopping patience=5 | Acceptable | Paper doesn't specify patience value |

### Paper Reproduction Score: 7/10

Architecture and preprocessing are exact. Training config matches. The critical deviation is the metric computation method (`weighted` vs per-class), which makes the reported numbers non-comparable to the paper. The 6.3% accuracy gap is also concerning but may be partially explained by implementation differences.

---

## 4. Data Pipeline Audit

### Dataset Loading

| Check | Status | Notes |
|---|---|---|
| Dataset discovered automatically | **PASS** | `find_dataset()` walks `/kaggle/input/` |
| All formats supported | **PASS** | JPG, JPEG, PNG, TIF, TIFF, BMP |
| Sorted file listing | **PASS** | `sorted(os.listdir(directory))` ensures determinism |
| Correct image counts | **PASS** | 7,491 Au + 5,123 Tp = 12,614 total |
| Zero images skipped | **PASS** | "Skipped: 0" |

### Data Leakage Check

| Check | Status | Notes |
|---|---|---|
| Stratified split | **PASS** | `stratify=Y` in `train_test_split` |
| Paired shuffle | **PASS** | `shuffle(X, Y, random_state=SEED)` — fixes fatal bug in reference code |
| No test set | **FAIL** | Only 80/20 train/val. Validation set used for both early stopping AND final evaluation. This is methodologically incorrect. |

### ELA Processing

| Check | Status | Notes |
|---|---|---|
| In-memory JPEG resave | **PASS** | BytesIO avoids temp file issues |
| Quality = 90 | **PASS** | Matches paper |
| Brightness scaling | **PASS** | Correct implementation |
| RGB conversion | **PASS** | `Image.open(path).convert('RGB')` |
| Normalization | **PASS** | `/255.0` to [0, 1] range |

### Class Imbalance

| Metric | Value |
|---|---|
| Authentic | 7,491 (59.4%) |
| Tampered | 5,123 (40.6%) |
| Ratio | 1.46:1 |

The imbalance is moderate (not severe). However, the notebook **does not address it at all** — no class weights, no oversampling, no undersampling. The paper also doesn't mention addressing it, so this is consistent with the reproduction, but it's a missed opportunity.

### Critical Issue: No Test Set

The notebook uses the validation set for both:
1. Early stopping (selecting the best epoch)
2. Final evaluation (reporting accuracy/F1/etc.)

This means the reported metrics are **optimistically biased**. The model was indirectly optimized for this exact set via early stopping. A proper evaluation requires a separate held-out test set that the model never sees, even indirectly.

**Severity: MAJOR** — This is a fundamental evaluation methodology error.

---

## 5. Model Architecture Audit

### Layer Verification

```
Layer 1: Conv2D      (124, 124, 32)    2,432 params    ✓
Layer 2: Conv2D      (120, 120, 32)   25,632 params    ✓
Layer 3: MaxPool2D   (60, 60, 32)          0 params    ✓
Layer 4: Dropout     (60, 60, 32)          0 params    ✓
Layer 5: Flatten     (115,200)             0 params    ✓
Layer 6: Dense       (256)        29,491,456 params    ✓
Layer 7: Dropout     (256)                 0 params    ✓
Layer 8: Dense       (2)                 514 params    ✓
                              ─────────────────────
Total:                          29,520,034 params    ✓
```

**All layers, shapes, activations, and parameter counts match the paper exactly.**

### Architectural Notes

- The 29.5M parameter count is dominated by the Flatten→Dense(256) connection. This is not per se wrong — the paper specifies it — but it means 99.9% of the model's parameters are in a single fully-connected layer. This is architecturally inefficient and would not generalize well to higher resolutions.
- No BatchNormalization is used. The paper doesn't specify it either, so this is consistent.
- Softmax output with 2 classes is correct (not sigmoid, not Dense(1)). This fixes a critical bug in the reference code.

### Architecture Score: 10/10

Perfect reproduction of the paper's architecture.

---

## 6. Training Pipeline Audit

### Training Dynamics

| Metric | Value |
|---|---|
| Total epochs | 13 (early stopped) |
| Best epoch | 8 (val_accuracy = 0.8989) |
| Final train accuracy | 0.9323 (epoch 13) |
| Final train loss | 0.1600 |
| Best val accuracy | 0.8989 (epoch 8) |
| Best val loss | 0.2473 |

### Convergence Analysis

- **Epoch 1**: Train acc 0.7320, val acc 0.8720 — fast initial learning from ELA features
- **Epochs 2-8**: Steady improvement, val acc plateau around 0.89-0.90
- **Epochs 9-13**: Train acc continues to 0.93 but val acc stagnates/declines — **classic overfitting**
- **Epoch 11**: Val acc drops to 0.8565 (spike), suggesting unstable optimization landscape
- **Epoch 13**: Early stopping triggers (patience=5 from best at epoch 8)

### Overfitting Assessment

| Epoch | Train Acc | Val Acc | Gap |
|---|---|---|---|
| 1 | 0.7320 | 0.8720 | -0.14 (val > train, normal for early epochs) |
| 8 | ~0.92 | 0.8989 | ~0.02 (acceptable) |
| 13 | 0.9323 | 0.8831 | **0.05 (overfitting beginning)** |

The gap is growing but early stopping caught it at epoch 8. The epoch 11 spike (val acc 0.8565) is concerning — it suggests the loss landscape has instability, possibly due to the massive Flatten→Dense layer.

### Weaknesses

1. **No learning rate scheduler** — A ReduceLROnPlateau or cosine annealing schedule would help the val accuracy plateau.
2. **No data augmentation** — The paper doesn't mention augmentation, but for a reproduction with a 6% accuracy gap, augmentation (flips, rotations) would likely close the gap.
3. **Epoch 11 instability** — Val accuracy drops 4% in one epoch, suggesting fragile optimization. The 29.5M-parameter Dense layer likely contributes to this.

---

## 7. Evaluation Metrics Audit

### Metrics Computed

| Metric | Present | Correct |
|---|---|---|
| Accuracy | Yes | **Yes** — `accuracy_score(y_true, y_pred)` = 0.8989 |
| Precision | Yes | **PROBLEMATIC** — `average='weighted'` |
| Recall | Yes | **PROBLEMATIC** — `average='weighted'` |
| F1 Score | Yes | **PROBLEMATIC** — `average='weighted'` |
| Confusion Matrix | Yes | **Yes** — correct values (1296/202/53/972) |
| ROC Curve / AUC | **No** | **Missing** |
| Per-class Precision/Recall | Yes (in classification_report) | **Yes** — correct |

### The `weighted` Average Problem

The paper reports:
- Precision: 98.58%
- Recall: 92.36%
- F1: 95.37%

These are almost certainly **per-class metrics for the tampered class** (or macro average). The notebook reports `average='weighted'` metrics, which weight by class support:

```
Weighted Precision = (0.9607 × 1498 + 0.8279 × 1025) / 2523 = 0.9068
```

This is **not comparable** to the paper's numbers. The weighted average inflates the tampered class precision (actual: 0.8279) by blending it with the authentic class precision (0.9607).

**From the classification report, the actual per-class metrics are:**

| Metric | Authentic | Tampered | Paper (likely tampered) |
|---|---|---|---|
| Precision | 0.9607 | **0.8279** | 98.58% |
| Recall | 0.8652 | **0.9483** | 92.36% |
| F1 | 0.9104 | **0.8840** | 95.37% |

The tampered class precision (82.79%) is **15.8 percentage points below** the paper's claim of 98.58%. This is a very large gap.

### Missing Metrics

1. **ROC-AUC** — Standard for binary classification, especially with class imbalance. Should absolutely be included.
2. **Per-class accuracy** — The `classification_report` partially covers this, but it should be called out explicitly.
3. **No comparison table** — The final summary table shows "This Run" metrics but doesn't show paper targets for comparison. The reader has to scroll back to Section 2 to compare.

### Confusion Matrix Analysis

```
                 Predicted
              Au      Tp
Actual Au   1296     202    (86.5% correct)
       Tp     53     972    (94.8% correct)
```

- **False Positive Rate** (Au predicted as Tp): 202/1498 = **13.5%** — This is high. The model is too aggressive in labeling authentic images as tampered.
- **False Negative Rate** (Tp predicted as Au): 53/1025 = **5.2%** — This is good.
- The model has a **bias toward predicting tampered**, which inflates recall for the tampered class (0.9483) but tanks precision (0.8279).

---

## 8. Results Analysis

### Accuracy Gap: 89.89% vs 96.21%

The 6.32% gap between this run and the paper's claim is significant. Possible explanations:

1. **Metric methodology difference** — The paper may have used a different split, different subset of CASIA, or different evaluation protocol. The paper's 80/20 split without stratification specification makes exact reproduction impossible.
2. **Image subset** — The paper may have filtered to JPEG-only images. CASIA v2.0 contains TIF and BMP files where ELA would be less effective (no prior JPEG compression). This run processes all 12,614 images including non-JPEG formats.
3. **Missing augmentation** — The paper may have used augmentation that isn't fully described.
4. **Random seed sensitivity** — The 80/20 split could produce substantially different results depending on the random seed, especially without stratification specified in the paper.
5. **One-hot vs integer labels** — The reference code has bugs that might have accidentally filtered or resampled the dataset.

### The Paper's Numbers May Be Unreliable

96.21% accuracy from a 2-layer CNN on CASIA v2.0 is exceptionally high. For reference:
- VGG16 reportedly achieves 90.32% on the same dataset (per the paper itself)
- ResNet101 reportedly achieves 74.75% (per the paper)

If a 2-layer CNN with 2 Conv layers outperforms VGG16 by 6 percentage points, it raises questions about whether the paper's evaluation was on a cherry-picked subset, a different dataset version, or benefited from data leakage.

### Training Curves Interpretation

The training curves show textbook early overfitting:
- Training accuracy climbs steadily to 93.2%
- Validation accuracy plateaus at ~89-90% and begins declining
- The gap opens from epoch 8 onward

This is expected behavior for a model with 29.5M parameters trained on ~10K images without augmentation. The model has more than enough capacity to memorize the training set.

### Is the Result Reasonable?

**Yes, 89.89% is a reasonable result** for this architecture on the full CASIA v2.0 dataset. It is arguably more honest than the paper's 96.21%. The model clearly learned ELA-based forensic features — the jump from 73% at epoch 1 to 89% by epoch 8 demonstrates genuine feature extraction, not random chance.

---

## 9. Alignment with Assignment Deliverables

### Assignment Requirements Check

| Requirement | Status | Notes |
|---|---|---|
| **Detect tampered images** | **PARTIAL** | Classification only, no pixel-level detection |
| **Generate pixel-level mask** | **FAIL** | **Not implemented. The assignment explicitly requires localization masks.** |
| **Dataset with ground truth masks** | **FAIL** | CASIA v2.0 has masks for tampered images, but this notebook ignores them entirely |
| **Train/validation/test split** | **FAIL** | Only train/val. No test set. |
| **Data augmentation** | **FAIL** | None implemented |
| **Performance metrics** | **PARTIAL** | Classification metrics present. No localization metrics (Dice, IoU, pixel-F1) |
| **Visual results** | **PARTIAL** | Sample predictions shown, but no Original/GT/Predicted/Overlay comparison |
| **Clear visualizations** | **PARTIAL** | Training curves and confusion matrix present, but no ELA visualization |
| **Model weights** | **FAIL** | Save cell commented out |
| **Robustness testing** | **FAIL** | Not implemented (bonus but expected) |

### Critical Assignment Gap: NO LOCALIZATION

The assignment states:

> "The model should not only classify whether an image is tampered, **but also generate a pixel-level mask** highlighting altered regions."

This notebook does **classification only**. There is no mask prediction, no decoder, no segmentation output, no Ground Truth mask comparison, no pixel-level metric. This is the single most critical deficiency.

The assignment also requires:

> "Provide clear visual results comparing the **Original Image, Ground Truth, Predicted output, and an Overlay Visualization**."

This notebook shows ELA map predictions with class labels. It does not show original images, ground truth masks, predicted masks, or overlay visualizations.

### Assignment Readiness Score: 3/10

The notebook is a technically competent paper reproduction, but it addresses less than half of the assignment requirements. The localization requirement — the core deliverable — is entirely missing.

---

## 10. Code Quality Audit

### Strengths

- **Clean structure** — Numbered sections, clear cell headers, consistent formatting
- **Centralized configuration** — All hyperparameters in one place at the top
- **Bug fixes documented** — Reference code bugs (11 total) identified and fixed
- **Deterministic** — Seeds set for Python, NumPy, TensorFlow
- **Memory management** — `del X, Y_onehot` to free arrays after split
- **Graceful error handling** — `compute_ela()` returns None on failure, caller counts skips
- **In-memory ELA** — BytesIO avoids temp file I/O and thread-safety issues

### Weaknesses

- **No modular functions for evaluation** — All metric computation is inline
- **Section numbering skips** — Section 3 (Reference Code Audit) missing, Section 7 appears twice (Training Pipeline and Evaluation Metrics both labeled "## 7")
- **Commented-out code** — Model save is commented out rather than being active
- **No logging** — Metrics not saved to CSV or JSON for later comparison
- **All data loaded into RAM** — The entire dataset (12,614 × 128 × 128 × 3 × 4 bytes ≈ 2.5 GB) is loaded into a single NumPy array. This works on P100 with 16 GB but is fragile for larger datasets.

### Code Quality Score: 7/10

Clean, readable, well-organized. The main issues are structural (missing sections, duplicate numbering) and the RAM-loading approach.

---

## 11. Critical Roast

### What a Strict Professor Would Say

**"You implemented the paper correctly, but you didn't do the assignment."**

1. **The assignment asks for localization. You built a classifier.** This is not a minor gap — it's a fundamental mismatch. The assignment says "generate a pixel-level mask highlighting altered regions." Your model outputs "{Authentic, Tampered}" — a single label for the entire image. Where is the mask?

2. **You evaluated on your validation set.** That's the set your early stopping used to select the best epoch. You don't get to then report those numbers as "test results." This is a statistics 101 error.

3. **Your accuracy is 6 points below the paper, and you didn't investigate why.** The Discussion section acknowledges limitations but doesn't analyze the accuracy gap. A proper reproduction would test hypotheses: Is it the non-JPEG images? Is it the split randomness? Is it the lack of augmentation? You just report the number and move on.

4. **You don't show what ELA looks like.** The entire paper is about ELA preprocessing making classification easier. You explain ELA in markdown, but you don't include a single visualization showing ELA maps for authentic vs. tampered images. How is the reader supposed to understand the key contribution?

5. **You have no augmentation, no robustness testing, and no model weights saved.** The assignment explicitly asks for augmentation, the bonus section asks for robustness testing, and the deliverables section asks for model weights. All three are missing.

6. **Your metrics are calculated with `weighted` average and presented as comparable to the paper's per-class metrics.** They aren't. The paper says 98.58% precision; your tampered-class precision is 82.79%. That's a 16-point gap that gets hidden by the `weighted` average.

### What Looks Rushed

- Section 3 (Reference Code Audit) is listed in the ToC but absent from the notebook
- Section numbering breaks (two Section 7s)
- Model save commented out instead of being executed
- No analysis of why accuracy doesn't match the paper
- No ELA visualization despite this being the paper's key contribution
- Discussion/Conclusion are generic templates, not analysis of the actual results

### What Would Cause Mark Deductions

| Issue | Severity |
|---|---|
| No localization / mask prediction | **-30 to -40%** of total grade |
| No test set (evaluation on validation) | **-10%** |
| No data augmentation | **-5%** |
| No robustness testing | **-5%** (bonus) |
| No ELA visualization | **-3%** |
| No ROC-AUC | **-2%** |
| No model weights saved | **-2%** |
| No Original/GT/Predicted/Overlay visual | **-5%** |
| Accuracy 6% below paper with no analysis | **-3%** |

---

## 12. Summary Scores

| Category | Score | Notes |
|---|---|---|
| **Paper Reproduction** | **7/10** | Architecture exact. Metrics methodology wrong (`weighted` vs per-class). 6% accuracy gap unexplained. |
| **Assignment Readiness** | **3/10** | Classification works but localization — the core requirement — is entirely absent. |
| **Code Quality** | **7/10** | Clean, readable, well-organized. Missing sections and structural issues. |
| **Evaluation Rigor** | **4/10** | No test set, no ROC-AUC, wrong metric averaging, no gap analysis. |
| **Presentation** | **5/10** | Good structure but missing ELA visualization, no comparison table in results, section numbering errors. |

---

## 13. Strengths

1. **Architecture is an exact reproduction** — Every layer matches the paper's Table III
2. **ELA implementation is correct** — In-memory BytesIO, proper scaling, matches paper formula
3. **Reference code bugs identified and fixed** — 11 bugs documented, all critical ones addressed
4. **Clean notebook structure** — Clear sections, centralized config, good markdown documentation
5. **The model actually works** — 89.89% accuracy is reasonable and honest
6. **Fast execution** — Full pipeline completes in well under 30 minutes on P100
7. **Deterministic** — Proper seeding for reproducibility

---

## 14. Weaknesses

1. **No localization** — Fails the assignment's core requirement
2. **No test set** — Validation set used for both model selection and final evaluation
3. **Metric averaging is wrong** — `weighted` average is not comparable to paper's per-class metrics
4. **6.3% accuracy gap unexplained** — No analysis of why the reproduction falls short
5. **No ELA visualization** — Key contribution not visually demonstrated
6. **No data augmentation** — Assignment explicitly requires it
7. **No robustness testing** — Even basic JPEG compression testing is absent
8. **No ROC-AUC** — Standard metric for binary classification missing
9. **No model weights saved** — Assignment requires model assets
10. **No Ground Truth / Predicted / Overlay** visualization
11. **Section numbering broken** — Section 3 missing, two Section 7s
12. **Class imbalance not addressed** — 59/41 split ignored

---

## 15. Major Issues

### MAJOR-1: No Localization (Assignment Failure)

The assignment requires pixel-level mask prediction. This notebook does image-level classification only. This is not an incremental improvement — it requires a fundamentally different model architecture (encoder-decoder, not classifier).

**Impact:** Fails the primary assignment deliverable.

### MAJOR-2: No Test Set (Evaluation Bias)

Reporting metrics on the validation set that was used for early stopping produces optimistically biased numbers. The model was indirectly optimized for this set.

**Impact:** Reported accuracy (89.89%) may be 2-5% higher than true generalization performance.

### MAJOR-3: Metric Averaging Mismatch

The `weighted` average metrics (P=0.9068, R=0.8989, F1=0.8997) are presented alongside paper metrics that are per-class. This makes the reproduction appear closer to the paper than it actually is. The actual tampered-class precision is 82.79%, not 90.68%.

**Impact:** Misleading comparison to paper results.

---

## 16. Minor Issues

1. **Section 3 missing from executed notebook** — ToC references it but content is absent
2. **Two sections numbered "7"** — Training Pipeline and Evaluation Metrics both say "## 7"
3. **Model save commented out** — Should be active and executed
4. **No learning rate scheduler** — Would likely improve the epoch 11 instability
5. **Sample predictions show ELA images, not originals** — Less intuitive for the reader
6. **Final summary table doesn't compare to paper targets** — Reader has to scroll back
7. **`warnings.filterwarnings('ignore')` suppresses all warnings** — Too aggressive
8. **Precision/Recall from `tf.keras.metrics`** — These track binary class 1 only by default, not matching the sklearn `weighted` metrics in the evaluation cell. The training log metrics and evaluation metrics use different definitions.

---

## 17. Actionable Fixes

### Priority 0 — Must Fix Before Submission

| # | Fix | Effort | Impact |
|---|---|---|---|
| 1 | **Add localization** — Either add a U-Net decoder for mask prediction, or reframe the submission to explicitly acknowledge classification-only scope and discuss why localization was not pursued | High | Addresses core assignment requirement |
| 2 | **Add a proper test set** — Split 70/15/15 (train/val/test). Use val for early stopping, test for final metrics. | Low | Fixes evaluation bias |
| 3 | **Fix metric averaging** — Report per-class metrics AND macro average. Add explicit comparison table vs paper. | Low | Honest comparison |
| 4 | **Save model weights** — Uncomment and execute the save cell | Trivial | Assignment deliverable |

### Priority 1 — Should Fix

| # | Fix | Effort |
|---|---|---|
| 5 | Add ELA visualization (authentic vs tampered side-by-side) | Low |
| 6 | Add ROC curve and AUC metric | Low |
| 7 | Add data augmentation (horizontal flip, rotation at minimum) | Low |
| 8 | Fix section numbering (restore Section 3, fix duplicate 7s) | Trivial |
| 9 | Analyze the 6% accuracy gap (test JPEG-only subset, different seeds) | Medium |
| 10 | Add Original Image / ELA / Prediction visualization grid | Low |

### Priority 2 — Nice to Have

| # | Fix | Effort |
|---|---|---|
| 11 | Add learning rate scheduler (ReduceLROnPlateau) | Low |
| 12 | Test robustness to JPEG compression, noise, blur | Medium |
| 13 | Add class weights to handle imbalance | Low |
| 14 | Log metrics to CSV/JSON for cross-run comparison | Low |

---

## 18. Comparison with Previous Project Runs

| Version | Architecture | Task | Accuracy | Tam-F1 | Status |
|---|---|---|---|---|---|
| **vR.ETASR** | **ELA + 2-layer CNN** | **Classification** | **89.89%** | **0.8840** | **Working but classification-only** |
| v6.5 | SMP UNet + ResNet34 | Localization | 82.46% | 0.4101 | Best localization result |
| vK.10.6 | Custom UNet | Localization | 83.57% | 0.2213 | Best from-scratch localization |
| vK.11.4 | Synthesis | Localization | 41.42% | 0.1321 | Failed |
| vK.12.0b | Synthesis | Localization | 40.62% | 0.1322 | Failed |

**Key insight:** vR.ETASR has the highest image-level accuracy in project history (89.89%), but it's solving a different (easier) problem. Classification accuracy is not comparable to localization Tam-F1. The localization runs (v6.5, vK.10.6) solve the assignment's actual task.

---

## 19. Final Verdict

**This notebook is a competent paper reproduction and a poor assignment submission.**

The ELA + CNN pipeline is correctly implemented, the code is clean, and the model genuinely learns. As a standalone paper reproduction exercise, it scores 7/10.

However, the assignment asks for image tampering **detection AND localization** with pixel-level masks, Ground Truth comparisons, and visual overlays. This notebook provides none of that. Submitting this as-is would demonstrate that the student can implement a simple classifier but did not attempt the harder (and required) localization task.

The path forward is either:
1. Add a localization component (U-Net decoder, GradCAM pseudo-localization, or ELA-based thresholding for approximate masks)
2. Or position this notebook as one component of a two-part submission where a separate localization notebook handles the mask prediction

**As a standalone final submission: NOT READY.**
