# 03_Model_Architecture.md Review

## Purpose

Defines the baseline SMP U-Net model, the image-level scoring rule, and the optional ELA, SRM, and encoder-comparison paths.

## Accuracy Score

`8/10`

## What Docs4 Fixed Since Audit 3

- Fixes the undefined `B` variable in the `prob_map.view(B, -1)` snippet.
- Keeps ELA as a 4th-channel option and SRM as a separate path, which removes the previous contradiction.
- Preserves the "use direct SMP attributes, not `model.unet.*`" correction.

## Research and Notebook Alignment

- Notebook v4 matches the RGB baseline cleanly and is consistent with the direct SMP API.
- Research alignment is credible but conservative: the repository supports segmentation-based localization, yet its strongest direct papers are more advanced than a plain U-Net baseline.

## Issues Found

- `max(prob_map)` is pragmatic rather than research-strong. It is acceptable for MVP but remains fragile.
- The optional ELA path loses the cheap pretrained-encoder advantage because `in_channels != 3`.
- The doc correctly frames SRM and ELA as optional, but neither is validated in the notebook MVP path.

## Suggested Improvements

- Keep the baseline as-is, but describe it as a Colab-friendly baseline rather than a strong research match.
- If ELA is ever promoted beyond future work, document the exact first-layer adaptation strategy instead of only warning about pretrained-weight loss.
- Preserve the note that image-level detection is the least robust part of the current design.
