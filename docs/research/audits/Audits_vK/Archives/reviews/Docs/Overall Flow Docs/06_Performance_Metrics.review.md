# Review: 06_Performance_Metrics.md

Source document path: `Docs/Overall Flow Docs/06_Performance_Metrics.md`

Purpose: Define the evaluation metrics and scoring pipeline.

Validity score: 6/10

## Assignment alignment
- Covers the required evaluation area well.
- Needs a cleaner metric policy.

## Technical correctness
- IoU and F1 formulas are correct.
- Treating authentic all-zero masks as F1 `1.0` for empty predictions can inflate average pixel-F1 when many authentic images are included (lines 60-83).
- Using `pred.max()` as the image-level tamper score is very sensitive to isolated hot pixels (lines 126-158).
- Oracle-F1 on the test set is useful for analysis, but it should not be used as the main reported threshold or tuned metric (lines 197-229, 267-283).

## Colab T4 feasibility
- The metric computations are feasible in notebook form.

## Issues found
- Major: Mean pixel-F1 can be misleading if authentic-image blanks dominate (lines 60-83).
- Major: Image-level detection score choice is brittle (lines 126-158).
- Moderate: Oracle-F1 needs explicit validation-only framing (lines 197-229).

## Contradictions with other docs
- `Docs/06_Best_Practices.md` calls AUC a pixel-level metric.
- `Docs/Copilot-Engineering-Instructions.md` omits AUC altogether.

## Recommendations
- Report tampered-only localization metrics alongside full-test metrics.
- Choose the image-level threshold on validation only.
- Keep Oracle-F1 as an appendix or diagnostic metric.

## Severity summary
- Major
