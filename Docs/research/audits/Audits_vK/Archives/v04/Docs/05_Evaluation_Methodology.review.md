# 05_Evaluation_Methodology.md Review

## Purpose

Defines metric formulas, threshold policy, authentic-image handling, reporting views, and evaluation-interface expectations.

## Accuracy Score

`9/10`

## What Docs4 Fixed Since Audit 3

- Adds the missing precision/recall helper so the metric section is now internally complete.
- Locks the single-threshold policy for both pixel and image decisions and states it explicitly.
- Preserves the mixed-set versus tampered-only reporting split and the honest note about authentic-image inflation.

## Research and Notebook Alignment

- Notebook v4 matches the documented threshold sweep and the mixed/tampered-only reporting pattern.
- The surveys and direct localization papers in the repository support overlap-based localization metrics and careful test-set separation.

## Issues Found

- The metric set is strong, but `max(prob_map)` for image-level decisions is still a heuristic rather than a research-backed detection head.
- The authentic-image inflation caveat is correct and should remain very visible because mixed-set metrics can look better than localization quality alone.
- The doc does not attempt calibration or uncertainty analysis, which is acceptable for the assignment but still a limitation.

## Suggested Improvements

- Keep the single-threshold design for MVP, but continue labeling it as a simplification.
- Retain the tampered-only reporting view as the honest localization measure.
- If the project later grows beyond the assignment, treat image-level scoring as the first place to improve.
