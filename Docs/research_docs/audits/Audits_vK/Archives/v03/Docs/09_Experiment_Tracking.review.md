# 09_Experiment_Tracking.md Review

## Purpose

Defines the optional Weights & Biases workflow for run logging, metric comparison, artifact storage, and future hyperparameter sweeps.

## Accuracy Score

`6/10`

## What Improved Since Audit 2

- States clearly that W&B is optional and that the notebook must still work without it.
- Separates core per-epoch metrics, end-of-training summaries, and artifact logging in a way that is easy to implement.
- Keeps sweeps out of the MVP path by labeling them as future work.

## Issues Found

- The setup section is effectively unconditional: it installs `wandb`, imports it directly, logs in, and initializes a run. That conflicts with the document’s own “optional” positioning and with the notebook’s guarded `USE_WANDB` flow.
- The section therefore drifts from the actual implementation pattern in `tamper_detection_v3.ipynb`, where W&B is optional rather than assumed.
- For an internship-scale Colab project, the document is slightly over-specified relative to the assignment, which only needs lightweight experiment reproducibility.

## Suggested Improvements

- Rewrite the setup example to mirror the guarded optional flow already used in the notebook.
- Add one short fallback note explaining what artifacts remain available when W&B is disabled, such as JSON summaries and checkpoint files on Drive.
- Keep sweeps and advanced comparison features clearly labeled as non-MVP so they do not blur the minimal implementation path.
