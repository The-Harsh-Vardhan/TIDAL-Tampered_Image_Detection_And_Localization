# 12_Final_Submission_Checklist.md Review

## Purpose

Provides the final implementation and submission checklist, grouped by MVP, optional improvements, and bonus work.

## Accuracy Score

`8/10`

## What Improved Since Audit 1

- Cleanly separated MVP-required items from Phase 2 and Phase 3 work.
- Removed hard-coded misalignment counts and replaced them with dynamic detection and logging.
- Added the gradient-accumulation flush requirement.
- Added mixed-set and tampered-only metric reporting.
- Restored the binary predicted mask as the required visualization output.

## Remaining Issues

- The pre-submission check says GPU verification means `torch.cuda.get_device_name()` shows T4. That is stricter than necessary; the implementation goal is T4 compatibility, not exclusive T4 execution.
- The Phase 2 checklist still refers to an oracle threshold marker in the F1-vs-threshold plot, while the rest of the docs now emphasize validation-selected operating thresholds and keep Oracle-F1 as supplementary analysis.

## Suggested Improvements

- Reword GPU verification to "compatible CUDA GPU available; T4 target confirmed when assigned."
- Rename the Phase 2 threshold-plot item to match the evaluation terminology used elsewhere.
