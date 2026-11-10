# 09_Engineering_Practices.md Review

## Purpose

Defines Colab setup, dependency installation, reproducibility practices, and notebook engineering guidance.

## Accuracy Score

`7/10`

## What Improved Since Audit 1

- Removed overconfident fixed resource-budget numbers and replaced them with runtime verification.
- Stopped claiming version pinning unless versions are actually pinned.
- Kept optional tooling such as W&B clearly outside the core path.
- Preserved a focused single-notebook engineering flow.

## Remaining Issues

- The default install command omits `kaggle`, even though the doc later uses the Kaggle CLI.
- The optional pinned install command covers `segmentation-models-pytorch` and `albumentations` only, so the docs still depend on Colab defaults for the rest of the environment.
- The doc is correct that T4 is the target, but real Colab GPU assignment is not guaranteed.

## Suggested Improvements

- Add `kaggle` to the default install command or explicitly state that the notebook assumes the CLI is already available in the runtime.
- Keep the "target T4" wording, but avoid implying that the notebook is invalid if Colab assigns a compatible alternative GPU.
