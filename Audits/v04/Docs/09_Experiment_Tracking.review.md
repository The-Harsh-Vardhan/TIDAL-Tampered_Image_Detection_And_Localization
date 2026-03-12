# 09_Experiment_Tracking.md Review

## Purpose

Defines the optional W&B workflow, the logged metrics, the fallback artifacts, and the comparison/sweep guidance.

## Accuracy Score

`8/10`

## What Docs4 Fixed Since Audit 3

- Correctly guards install, import, login, init, logging, and finish behind `USE_WANDB`.
- Adds fallback artifacts so reproducibility does not depend on W&B.
- Keeps sweeps in future work instead of letting them leak into the MVP.

## Research and Notebook Alignment

- Notebook v4 aligns strongly with the guarded W&B behavior and logs metrics across training, evaluation, visualization, robustness, and export.
- The research papers do not require W&B specifically, but this doc now supports reproducibility without overcomplicating the MVP.

## Issues Found

- The fallback-artifact table documents `results_summary.json`, but notebook v4 writes `results_summary_v4.json`.
- The document reads like a standalone experiment-tracking module, while notebook v4 treats tracking as an integrated cross-cutting concern.
- The comparison guidance is useful, but it is more mature than what the assignment strictly requires.

## Suggested Improvements

- Standardize the results-summary filename.
- Add one sentence that W&B is implemented as an integrated optional layer across multiple notebook sections, not as a separate block of work.
- Keep sweep tooling clearly outside the MVP path.
