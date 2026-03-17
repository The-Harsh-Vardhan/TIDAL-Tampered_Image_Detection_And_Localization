# 09 — Final Kill Verdict

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC  
**Date:** 2026-03-13

---

## Question 1: How many assignment requirements are truly satisfied?

| Requirement | Verdict |
|---|---|
| Dataset explanation | Partial — auto-discovery only, no EDA |
| Model architecture description | ✓ |
| Training strategy explanation | ✓ |
| Hyperparameter documentation | ✓ |
| Evaluation results | Partial — primary metric inflated |
| Visualization of predictions | ✓ |
| Tamper detection | ✓ |
| Tamper localization | ✓ |
| **Runnable Colab pipeline** | **✗ — Kaggle only** |
| Architecture reasoning | Partial — no justification |

**Score: 5.5 / 10** fully satisfied, with the Colab requirement being a hard fail.

---

## Question 2: Are the reported results trustworthy?

**No.** The primary "Pixel F1 (all)" metric is inflated by including authentic samples that trivially score F1=1.0. The tampered-only F1 may be significantly lower. Additionally:

- The model uses a from-scratch encoder on a small dataset, raising overfitting concerns
- No cross-dataset validation to verify generalization
- JPEG shortcut exploitation is untested
- Batch-averaged validation metrics introduce bias from uneven batch sizes

The **tampered-only F1** and **mask-size stratification** are the honest metrics, but they are not the primary reported metrics.

---

## Question 3: Would a senior ML engineer accept this as a credible solution?

**Partially.** The notebook demonstrates:

**Strengths (that a senior engineer would appreciate):**
- Well-structured code with centralized CONFIG
- Comprehensive training pipeline (AMP, accum, early stopping, checkpointing)
- Threshold sweep on validation set (correct methodology)
- Mask-size stratified evaluation
- Robustness testing and shortcut learning checks
- W&B integration

**Weaknesses (that would raise red flags):**
- From-scratch encoder on a small dataset is a poor architectural choice
- Primary metric is inflated — signals either carelessness or intent to mislead
- 7 pieces of dead code suggest hasty assembly
- Stderr suppression is a serious engineering malpractice
- No Colab support despite being a requirement
- No architectural justification or comparison with alternatives

---

## Final Verdict

### 🟡 Needs Significant Improvement

**Reasoning:**

The notebook is not "fundamentally flawed" — the pipeline is functional, the code is readable, and the evaluation includes genuine analyses (threshold sweep, robustness, shortcut checks). These are above-average for an assignment submission.

However, it has three critical problems:
1. **Metric inflation** — the primary reported metric is misleading
2. **Architecture choice** — training 31M params from scratch on 7K images is unjustifiable when pretrained encoders are available
3. **Platform mismatch** — notebook targets Kaggle but assignment requires Colab

A senior engineer would say: *"The pipeline engineering is solid. The science is weak. Fix the metrics, use a pretrained encoder, add Colab support, and this becomes an acceptable submission."*

---

## Required Changes for "Acceptable Submission"

1. Report tampered-only F1 as the **primary** localization metric
2. Replace custom UNet encoder with pretrained ResNet34 (via SMP or manual)
3. Add Colab compatibility (or justify the Kaggle-only decision)
4. Remove stderr suppression
5. Clean dead code and unused imports
6. Add dataset EDA section
7. Add architectural justification

**If these 7 items are fixed, the verdict would upgrade to "Acceptable Submission."**
