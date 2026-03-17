# Review: 03_Augmentation.md

Source document path: `Docs/Overall Flow Docs/03_Augmentation.md`

Purpose: Specify augmentation strategy for training and robustness.

Validity score: 7/10

## Assignment alignment
- Strongly aligned with the augmentation and robustness parts of the brief.

## Technical correctness
- The synchronized augmentation logic and spatial-vs-pixel distinction are correct.
- The performance/adoption comparison for `albumentations` is overstated and not locally verified (lines 25-33).
- `HueSaturationValue` and vertical flips may be more aggressive than a minimal forensic baseline needs (lines 69-79, 103-110).
- The bonus section quotes cropping but only implements JPEG, noise, and resize variants (lines 167-200).

## Colab T4 feasibility
- The proposed transform set is feasible.
- Conservative JPEG/noise/resize augmentation is the most useful subset.

## Issues found
- Moderate: Some augmentation claims and transform choices are stronger than necessary (lines 25-33, 69-79).
- Minor: Cropping is referenced in the assignment mapping but not implemented in the robustness section (lines 167-200).

## Contradictions with other docs
- Better aligned with forensic needs than `Docs/Copilot-Engineering-Instructions.md`, which includes blur more casually.
- More conservative than `Docs/Overall Flow.md` in the right places.

## Recommendations
- Keep resize, flip, `RandomRotate90`, light JPEG, and light noise.
- Make color transforms optional and explain why crop is excluded from the core path.

## Severity summary
- Moderate
