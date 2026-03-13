# 08_Robustness_Testing.md Review

## Purpose

Defines the optional bonus protocol for testing robustness under JPEG compression, Gaussian noise, and resizing.

## Technical Accuracy Score

`5/10`

## What Is Correct

- Technical correctness: JPEG compression, Gaussian noise, and resizing are the right bonus degradations for this assignment.
- Implementability on a single Colab notebook with T4: Running post-training robustness evaluation on the test split is lightweight enough for Colab if the protocol is kept simple.
- Assignment alignment: The doc correctly treats robustness testing as bonus work rather than a core requirement.

## Issues Found

- Contradictions: The prose says ground-truth masks are unchanged during robustness testing, but the resize-based `albumentations` pipeline will downscale and upscale masks as well as images.
- Unsupported or hallucinated claims: The expected `5-10%` and `15-35%` F1 drops are unverified and should not be presented as likely outcomes without measured evidence.
- Unnecessary complexity: The document is reasonably scoped overall; the main problem is that the implementation details are not fully correct.
- Missing technical details: It does not say whether robustness evaluation should reuse the clean validation-selected threshold or allow per-degradation threshold retuning. The correct baseline is to reuse the clean validated threshold.
- Additional implementation risk: The code depends on `albumentations` parameter names that may vary by version, especially for `ImageCompression` and `GaussNoise`.
- Additional implementation risk: The robustness report only tracks Pixel-F1, which is acceptable for a bonus summary but thinner than the rest of the evaluation policy.

## Recommendations

- Apply degradations to images only, while masks follow the clean geometric path needed for comparison.
- Remove expected performance-drop percentages unless they are backed by actual runs.
- State explicitly that robustness testing reuses the threshold chosen on the clean validation set.
- Keep the bonus report compact: at minimum clean versus degraded Pixel-F1, with optional IoU if time allows.
