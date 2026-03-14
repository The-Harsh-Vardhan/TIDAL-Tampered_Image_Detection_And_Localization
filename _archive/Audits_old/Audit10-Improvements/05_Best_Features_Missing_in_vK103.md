# Best Features Missing in vK.10.3

Actionable list of features from v8, vK.3, and vK.7.5 that should be added to vK.10.3 before submission.

Ordered by **impact on assignment grade** (highest first).

---

## PRIORITY 1: Required for Rigorous Evaluation (Assignment E3, E4, B1)

### 1. Robustness Testing (from v8)
**Source:** v8 cells 44-46
**What:** Test the trained model against 8 degradation conditions:
- JPEG compression (QF=70, QF=50)
- Gaussian noise (sigma=10, sigma=25)
- Gaussian blur (kernel=3)
- Resize (0.75x, 0.5x then back to original size)

**Why:** This is an explicit **bonus requirement** (B1). v8 showed only 0.9% drop for JPEG but 13% for noise — this kind of insight demonstrates rigorous evaluation.

**Effort:** Medium. Need a `robustness_test()` function that applies degradation → re-runs inference → compares metrics. ~60 lines of code + 1 bar chart.

---

### 2. Threshold Optimization (from v8)
**Source:** v8 cell 38
**What:** Sweep segmentation threshold from 0.05–0.80 in 0.05 steps. Evaluate F1/Dice at each threshold on the validation set. Select optimal threshold for test evaluation.

**Why:** Fixed threshold=0.5 is rarely optimal. v8 found optimal at 0.75 — a significant difference. Finding the right threshold can improve localization metrics substantially with zero retraining.

**Effort:** Low. ~30 lines: loop over thresholds, compute metrics, select best, plot F1 vs threshold curve.

---

### 3. Grad-CAM Explainability (from v8)
**Source:** v8 cells 48-49
**What:** Generate Grad-CAM heatmaps from the encoder bottleneck. Overlay on images with diagnostic coloring (TP=green, FP=red, FN=blue).

**Why:** Demonstrates the model's decision-making process. Shows the reviewer that you understand what the model learned. Addresses evaluation criterion E2 (thoughtful architecture choices) by visualizing what features the model attends to.

**Effort:** Medium. ~80 lines. Needs hook-based gradient extraction from the bottleneck layer.

---

### 4. Data Leakage Verification (from v8)
**Source:** v8 cell 10
**What:** Explicitly verify zero overlap between train/val/test image paths with assertions.

**Why:** Given the history of data leakage bugs in vK.3/vK.7.5 prior blocks, having an explicit verification cell in vK.10.3 shows awareness and rigor. Simple to add, high credibility value.

**Effort:** Very low. ~10 lines: load CSVs, check intersection is empty, assert.

---

## PRIORITY 2: Improves Evaluation Depth (Assignment E3, E6, B2)

### 5. Mask-Size Stratified Evaluation (from v8)
**Source:** v8 cell 42
**What:** Bucket test images by tampered region size (tiny <2%, small 2-5%, medium 5-15%, large >15%). Report metrics per bucket.

**Why:** Reveals where the model fails — tiny regions are much harder. This level of analysis shows deep understanding of the problem. Addresses B2 (subtle tampering detection).

**Effort:** Low. ~40 lines: compute mask area ratio, bucket, compute per-bucket metrics.

---

### 6. Forgery-Type Breakdown (from v8)
**Source:** v8 cell 43
**What:** Parse CASIA filenames to determine forgery type (splicing vs copy-move). Report metrics per type.

**Why:** v8 showed copy-move is much harder (F1=0.14 vs splicing F1=0.58). This insight directly addresses B2 and demonstrates understanding of different forgery mechanisms.

**Effort:** Low. ~30 lines: parse `Tp_*_*` filenames, group by type, compute metrics.

---

### 7. Shortcut Learning Checks (from v8)
**Source:** v8 cell 47
**What:** Two tests:
1. **Mask randomization**: Shuffle masks randomly across images. If F1 drops sharply → model actually uses image content, not shortcuts.
2. **Boundary sensitivity**: Erode/dilate predicted masks by 1px. If metrics barely change → predictions aren't just edge artifacts.

**Why:** Addresses the scientific rigor criterion (E3). These tests prove the model learned meaningful tampering features rather than dataset artifacts.

**Effort:** Low. ~40 lines: shuffle masks, re-evaluate; erode/dilate, re-evaluate.

---

### 8. Failure Case Analysis (from v8)
**Source:** v8 cell 50
**What:** Identify the 10 worst predictions (lowest per-sample F1). Display them with GT mask, predicted mask, and metadata (mask size, forgery type). Annotate why they failed.

**Why:** Shows self-awareness about model limitations. Reviewers love seeing that you understand failure modes, not just success cases.

**Effort:** Low. ~40 lines: sort samples by metric, display worst N.

---

## PRIORITY 3: Nice-to-Have Improvements

### 9. Confusion Matrix (not in any run, but standard)
**What:** Plot a 2x2 confusion matrix heatmap for image-level classification (authentic vs tampered).

**Why:** Standard classification deliverable. Complements accuracy and AUC-ROC. Shows false positive vs false negative trade-off.

**Effort:** Very low. ~10 lines with `sklearn.metrics.confusion_matrix` + `seaborn.heatmap`.

---

### 10. ROC Curve and PR Curve Plots (not in any run)
**What:** Plot ROC curve (TPR vs FPR) and Precision-Recall curve for image-level classification.

**Why:** vK.10.3 already computes AUC-ROC numerically but doesn't plot the curve. The visual is important for the assignment reviewer and is standard for binary classification presentations.

**Effort:** Very low. ~15 lines using `sklearn.metrics.roc_curve` and `precision_recall_curve`.

---

### 11. Artifact Inventory Cell (from v8)
**Source:** v8 cell 53
**What:** At the end of the notebook, list all saved files with paths, sizes, and a verification check that each exists.

**Why:** Shows completeness and professionalism. Makes it easy for the reviewer to find deliverables.

**Effort:** Very low. ~15 lines: list files, check existence, print sizes.

---

### 12. Gradient Accumulation (from v8)
**Source:** v8 training loop
**What:** Accumulate gradients over N mini-batches before optimizer step, achieving an effective batch size of `batch_size * N`.

**Why:** v8 used 4-step accumulation for effective batch=256. This stabilizes training and can improve convergence. Especially valuable on T4 where VRAM limits batch size.

**Effort:** Medium. Modify training loop to accumulate N steps before `optimizer.step()`.

---

### 13. Differential Learning Rates (from v8)
**Source:** v8 optimizer setup
**What:** Use lower LR for encoder, higher LR for decoder.

**Why:** v8 used encoder_lr=1e-4, decoder_lr=1e-3. Since vK.10.3 trains from scratch (no pretrained backbone), this is less critical, but could still help the decoder learn faster.

**Effort:** Low. Create parameter groups in Adam constructor.

---

## Summary: Implementation Priority

| Priority | Feature | Effort | Grade Impact |
|----------|---------|--------|--------------|
| **P1** | Robustness testing | Medium | HIGH (bonus B1) |
| **P1** | Threshold optimization | Low | HIGH (improves metrics) |
| **P1** | Grad-CAM explainability | Medium | HIGH (evaluation rigor) |
| **P1** | Data leakage verification | Very low | MEDIUM (credibility) |
| **P2** | Mask-size stratification | Low | MEDIUM (depth) |
| **P2** | Forgery-type breakdown | Low | MEDIUM (bonus B2) |
| **P2** | Shortcut learning checks | Low | MEDIUM (scientific rigor) |
| **P2** | Failure case analysis | Low | MEDIUM (self-awareness) |
| **P3** | Confusion matrix | Very low | LOW (standard) |
| **P3** | ROC/PR curve plots | Very low | LOW (standard) |
| **P3** | Artifact inventory | Very low | LOW (professionalism) |
| **P3** | Gradient accumulation | Medium | LOW (training quality) |
| **P3** | Differential LR | Low | LOW (marginal) |

**Total estimated effort to implement P1+P2:** ~8 new cells (4 code + 4 markdown), ~300 lines of code.
