# 05_Evaluation_Methodology.md Review

## Purpose

Defines the metric set, threshold protocol, authentic-image handling, reporting views, and the evaluation function interface.

## Accuracy Score

`9/10`

## What Improved Since Audit 2

- Fully resolves the previous pixel-threshold versus image-threshold ambiguity by locking one threshold for both.
- Keeps mixed-set and tampered-only reporting explicit.
- Clearly documents forgery-type breakdown and the final evaluation interface.
- Makes the single-threshold design choice explicit rather than leaving it implicit in code.

## Issues Found

- Precision and recall are defined in the metric table, but helper-function code is shown only for F1 and IoU.
- The doc is intentionally simple, but that simplicity still inherits the known fragility of `max(prob_map)` for image-level detection.

## Suggested Improvements

- Add a small precision/recall helper snippet for completeness if this doc is intended to be directly implementation-ready on its own.
- Keep the single-threshold policy, but preserve the limitation note so readers do not mistake simplicity for optimality.
