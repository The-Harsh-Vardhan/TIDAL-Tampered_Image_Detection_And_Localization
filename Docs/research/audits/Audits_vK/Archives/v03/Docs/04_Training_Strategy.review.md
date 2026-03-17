# 04_Training_Strategy.md Review

## Purpose

Defines the loss, optimizer, training hyperparameters, checkpointing, scheduler policy, and the practical data-loading defaults for the notebook.

## Accuracy Score

`9/10`

## What Improved Since Audit 2

- Preserves the corrected SMP attribute usage.
- Keeps the scheduler clearly out of the MVP baseline.
- Documents loss scaling, partial-window flush, early stopping, checkpoint state, and scheduler resume behavior cleanly.
- Integrates the MVP versus Phase 2 augmentation policy without the old baseline drift.

## Issues Found

- The optional augmentation examples still depend on the version-pinned `albumentations` API defined elsewhere.
- The training doc is broad enough that it effectively carries some data-pipeline responsibilities because `Docs3` has no standalone data-pipeline document.

## Suggested Improvements

- Cross-reference the exact version-pinning rule from `08_Engineering_Practices.md` next to the Phase 2 augmentation examples.
- If the doc set grows again, consider separating dataset/loaders from training; otherwise this structure is acceptable.
