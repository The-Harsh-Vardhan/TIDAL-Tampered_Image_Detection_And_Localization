# 06_Evaluation_Methodology.md Review

## Purpose

Defines pixel-level and image-level metrics, threshold selection, and the full evaluation pipeline.

## Technical Accuracy Score

`6/10`

## What Is Correct

- Technical correctness: The core metric formulas are standard, and the document correctly states that threshold tuning should not happen on the test set.
- Implementability on a single Colab notebook with T4: The evaluation flow is small enough for notebook execution and does not depend on heavy infrastructure.
- Assignment alignment: IoU, Dice/F1, precision, recall, image-level detection, and visual analysis all match the assignment's evaluation expectations.

## Issues Found

- Contradictions: Threshold handling drifts across the final docs. This file treats validation-based thresholding as required, while the timeline frames threshold calibration as a later optimization step.
- Unsupported or hallucinated claims: The example result table uses plausible numbers, but they are still illustrative rather than measured. That is acceptable only if clearly labeled as examples.
- Unnecessary complexity: Oracle-F1 is acceptable as analysis, but it can easily distract from the baseline reporting protocol if not clearly subordinated.
- Missing technical details: The document averages pixel metrics over authentic images by returning `1.0` when both prediction and ground truth are empty. That inflates localization means on mixed authentic/tampered test sets.
- Additional implementation risk: Image-level detection still depends on a fragile max-probability rule, and the doc does not clearly say whether image-level thresholding should share the segmentation threshold or have its own validation-selected operating point.

## Recommendations

- Report two views of localization performance: tampered-only localization metrics and mixed-set behavior that includes authentic images.
- Clarify whether image-level detection uses the same threshold as pixel binarization or a separately validated threshold on the image score.
- Keep Oracle-F1 strictly supplementary and never part of model selection on the test set.
