# 08_Engineering_Practices.md Review

## Purpose

Defines the Colab environment assumptions, dependency installation, GPU verification, storage paths, reproducibility controls, notebook structure, and excluded tooling.

## Accuracy Score

`7/10`

## What Improved Since Audit 2

- Fixes the earlier omission of `kaggle` from the default setup path.
- Correctly relaxes GPU validation from “must be T4” to “compatible CUDA GPU acceptable,” which is the right Colab-facing requirement.
- Keeps scope control explicit by listing excluded tools and avoiding unnecessary platform complexity.
- Maps the intended notebook structure in a way that now largely matches `tamper_detection_v3.ipynb`.

## Issues Found

- The install line uses an unquoted version range for `albumentations`:
  `!pip install -q kaggle segmentation-models-pytorch albumentations>=1.3.1,<2.0`
  This is a docs-to-notebook mismatch and may fail if copied literally in a shell-style context.
- The package-availability notes assume a stable Colab baseline. That is reasonable as guidance, but it is still environment-dependent rather than guaranteed.
- The section supports Colab feasibility, but it does not provide a measured resource budget; feasibility remains based on conservative design choices rather than validated runtime evidence.

## Suggested Improvements

- Quote the `albumentations` version range exactly as the notebook does: `"albumentations>=1.3.1,<2.0"`.
- Soften the “pre-installed in Colab” wording to indicate that package availability can vary across runtime images.
- Keep the environment section practical, but label hardware and package assumptions as operational guidance rather than guarantees.
