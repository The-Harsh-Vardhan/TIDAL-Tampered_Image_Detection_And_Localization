# 11_Limitations_and_Future_Work.md Review

## Purpose

Documents known limitations of the chosen approach and keeps clearly out-of-scope ideas out of the core implementation path.

## Accuracy Score

`9/10`

## What Improved Since Audit 1

- Added an explicit limitations document that the previous round lacked.
- Captured the key unresolved concerns honestly: split integrity, dataset size, image-level scoring fragility, mixed-set metric inflation, Colab session limits, and SRM placeholder quality.
- Kept advanced directions in future work instead of letting them contaminate the baseline scope.

## Remaining Issues

- The document correctly notes that `max(probability_map)` is fragile, but that also highlights that the final replacement has not yet been chosen elsewhere.
- The approximate dataset size wording is acceptable, but still intentionally approximate rather than measured.

## Suggested Improvements

- After the final image-level score is chosen, update this doc so it describes only the residual limitation, not an open design decision.
