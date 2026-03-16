# 08_Robustness_Testing.md Review

## Purpose

Defines the optional bonus protocol for robustness evaluation under common image degradations.

## Accuracy Score

`7/10`

## What Improved Since Audit 1

- Removed unsupported expected degradation-drop numbers and switched to measured-only reporting.
- Made threshold reuse explicit: robustness testing uses the clean validation-selected threshold.
- Fixed the conceptual resize-mask problem by stating that degradations must apply to images only and masks must stay clean.

## Remaining Issues

- The resize degradation helper is described separately, but the shown `evaluate_robustness()` loop does not demonstrate how resize-degraded images are actually fed into the dataset or loader.
- The optional degradation transforms still use version-sensitive `albumentations` parameters.
- The bonus report tracks Pixel-F1 only; that is acceptable for a compact bonus section, but it is thinner than the core evaluation design.

## Suggested Improvements

- Add one concrete resize-degradation path to the robustness evaluation example.
- Pin or validate the exact `albumentations` version if the documented parameter names are retained.
- Optionally report IoU alongside F1 for robustness testing if notebook space permits.
