# 04_Model_Architecture.md Review

## Purpose

Defines the segmentation baseline, encoder selection, SMP model API usage, and optional SRM direction.

## Accuracy Score

`8/10`

## What Improved Since Audit 1

- Locked the MVP baseline to `smp.Unet` with `encoder_name="resnet34"`.
- Explicitly corrected the SMP model API and removed the old `model.unet.*` confusion.
- Moved SRM into a clearly optional Phase 3 ablation and downgraded the earlier placeholder implementation.
- Removed unsupported hard numbers for parameters and VRAM, replacing them with a measure-in-notebook instruction.

## Remaining Issues

- The image-level detection rule is still not fully locked. The doc uses `max()` but also recommends top-k mean or mask-area fraction as potentially better choices.
- The threshold relation between pixel masks and image-level detection is deferred rather than frozen here.

## Suggested Improvements

- Pick one final image-level score for the MVP and use it consistently across `04`, `06`, `11`, `12`, and the notebook.
- State explicitly whether the image-level threshold is the same as or separate from the pixel binarization threshold.
