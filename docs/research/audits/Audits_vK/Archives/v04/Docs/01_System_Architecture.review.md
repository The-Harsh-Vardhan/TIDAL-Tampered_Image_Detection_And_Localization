# 01_System_Architecture.md Review

## Purpose

Defines the end-to-end pipeline, the major design decisions, the runtime assumptions, and the optional ELA/SRM branches.

## Accuracy Score

`9/10`

## What Docs4 Fixed Since Audit 3

- Resolves the earlier ELA input-channel contradiction by standardizing ELA as a 4th channel and SRM as a separate 6-channel path.
- Downgrades the VRAM figure from a fact claim to an estimate.
- Keeps the single-threshold and `max(prob_map)` decisions explicit instead of leaving them ambiguous.

## Research and Notebook Alignment

- Notebook v4 matches the baseline path well: CASIA, dynamic discovery, RGB `smp.Unet`, BCE + Dice, threshold sweep, robustness testing, and guarded W&B are all present.
- Research support is good for the segmentation framing and optional forensic side channels, but the strongest papers in the repo are more advanced than this baseline.

## Issues Found

- The estimated VRAM budget is still unmeasured documentation guidance rather than validated runtime evidence.
- `max(prob_map)` remains the weakest detection choice in the pipeline because isolated false positives can flip the image-level decision.
- The architecture is intentionally simpler than the stronger edge-enhanced, multi-trace, and transformer-based localization papers in the repository.

## Suggested Improvements

- Preserve the current baseline, but label it clearly as a Colab-friendly baseline rather than a research-frontier architecture.
- Keep the VRAM estimate caveated unless measured numbers are added from a real Colab run.
- Retain the `max(prob_map)` limitation note so readers do not mistake simplicity for robustness.
