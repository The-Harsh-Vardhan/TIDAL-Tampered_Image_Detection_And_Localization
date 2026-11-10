# 05_Training_Strategy.md Review

## Purpose

Defines the baseline loss, optimizer, training loop, AMP usage, validation flow, and checkpoint strategy.

## Accuracy Score

`8/10`

## What Improved Since Audit 1

- Fixed the SMP attribute mismatch by using `model.encoder`, `model.decoder`, and `model.segmentation_head`.
- Moved the LR scheduler to Phase 2 instead of treating it as baseline.
- Added the flush for the final partial gradient-accumulation window.
- Added `best_epoch` to checkpoint state for correct early-stopping resume.
- Removed the earlier edge-loss formulation from the default path.

## Remaining Issues

- The doc explains that scheduler state should be saved if Phase 2 is used, but the shown resume helper does not include the optional scheduler branch.
- The validation example still uses a single threshold argument and therefore relies on the evaluation doc to clarify any pixel/image threshold distinction.

## Suggested Improvements

- Add one optional resume snippet showing how scheduler state is restored when the Phase 2 scheduler is enabled.
- Keep the training doc baseline simple, but reference the exact threshold contract used by the evaluation code.
