# 04_Model_Architecture.md Review

## Purpose

Describes the baseline segmentation model, encoder options, output semantics, and optional SRM-based enhancement.

## Technical Accuracy Score

`6/10`

## What Is Correct

- Technical correctness: U-Net with an ImageNet-pretrained encoder is a credible baseline for pixel-level tampering localization on CASIA.
- Implementability on a single Colab notebook with T4: The RGB-only baseline is realistic for Colab T4, especially with AMP and moderate batch sizes.
- Assignment alignment: The doc correctly avoids transformer-heavy or multi-stream architectures as mandatory requirements.

## Issues Found

- Contradictions: The baseline model object here is plain `smp.Unet`, but `05_Training_Pipeline.md` later assumes a wrapper object under `model.unet`.
- Unsupported or hallucinated claims: The parameter counts and VRAM numbers are plausible but unverified in this repository. They should be treated as estimates, not established measurements.
- Unsupported or hallucinated claims: `tamper_score = predicted_mask.max()` is presented as the image-level detection rule, but the variable naming is ambiguous about whether that means the probability map or an already-thresholded binary mask.
- Unnecessary complexity: The SRM path is described in more implementation detail than the project needs for a v1 notebook baseline.
- Missing technical details: The document does not define a more stable image-level score, such as mask area fraction or top-k average probability, even though `max()` is highly sensitive to isolated false positives.
- Additional implementation risk: The SRM section uses a placeholder-grade kernel construction that repeats a tiny base set to simulate 30 filters. That is not strong enough to present as a serious forensic enhancement without stronger caveats.

## Recommendations

- Lock one baseline encoder and keep the initial model RGB-only.
- Rephrase memory and parameter figures as approximate until they are measured in Colab.
- Define image-level detection using the probability map explicitly and consider a more stable score than plain max probability.
- Keep SRM as an optional ablation note, not a semi-implemented second baseline.
