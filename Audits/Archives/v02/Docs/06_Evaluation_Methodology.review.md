# 06_Evaluation_Methodology.md Review

## Purpose

Defines the metric set, threshold policy, and evaluation protocol for the rewritten project plan.

## Accuracy Score

`7/10`

## What Improved Since Audit 1

- Added mixed-set and tampered-only reporting to avoid localization inflation from authentic images.
- Made validation-only threshold selection the explicit rule.
- Kept Oracle-F1 as supplementary analysis rather than model-selection logic.

## Remaining Issues

- The document says the image-level threshold may differ from the pixel threshold, but the shown `evaluate()` function accepts one `threshold` argument and uses it for both.
- The image-level score is still not frozen: `prob_map.max()` is shown, while top-k mean is mentioned as an alternative.

## Suggested Improvements

- Split the evaluation interface into `pixel_threshold` and `image_threshold` if the project intends to use different operating points.
- Lock one image-level score for the MVP and keep alternatives in the limitations or future-work section.
