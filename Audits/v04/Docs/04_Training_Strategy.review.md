# 04_Training_Strategy.md Review

## Purpose

Defines loss, optimizer, hyperparameters, AMP, gradient accumulation, checkpointing, scheduler policy, and data-loading behavior.

## Accuracy Score

`9/10`

## What Docs4 Fixed Since Audit 3

- Keeps the corrected direct SMP parameter grouping instead of the old `model.unet.*` mismatch.
- Preserves the partial-window gradient-accumulation flush.
- Integrates guarded W&B logging into the training-loop description instead of treating it as unconditional.

## Research and Notebook Alignment

- Notebook v4 matches this document closely on BCE + Dice, AdamW, AMP, checkpointing, and per-epoch guarded logging.
- The research papers do not force a different optimizer or loss choice; this is a practical engineering configuration for a small-dataset Colab baseline.

## Issues Found

- The document is feasible, but it still relies on estimated rather than measured Colab runtime and memory behavior.
- The scheduler remains Phase 2 only, which is consistent, but no empirical trigger is documented for when it should be enabled.
- The augmentation exclusions are sensible, though they are based on forensic intuition rather than citations to specific repository papers.

## Suggested Improvements

- Once the notebook is run, add one measured Colab note for epoch time and peak memory.
- Keep the scheduler optional, but add one short criterion for when it is worth trying.
- Preserve the current MVP simplicity; this is one of the stronger Docs4 files.
