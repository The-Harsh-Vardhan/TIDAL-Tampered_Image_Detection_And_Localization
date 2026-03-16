# 09_Engineering_Practices.md Review

## Purpose

Defines Colab environment assumptions, dependency installation, reproducibility practices, and notebook engineering guidance.

## Technical Accuracy Score

`6/10`

## What Is Correct

- Technical correctness: The file keeps the project notebook-first, uses AMP, checkpoint persistence, seed setting, and a practical section order for Colab.
- Implementability on a single Colab notebook with T4: The overall engineering approach is realistic for a one-notebook internship assignment.
- Assignment alignment: It correctly excludes heavy platform choices such as Databricks, DALI, and database layers from the core path.

## Issues Found

- Contradictions: The file assumes Kaggle API setup and Google Drive checkpointing, which weakens the claim in `01_Assignment_Overview.md` that no external services are required for the core submission.
- Unsupported or hallucinated claims: The VRAM budget table is plausible but not verified. It should be labeled as a rough estimate, not a measured budget.
- Unnecessary complexity: Optional W&B logging is fine, but it should remain clearly nonessential so the notebook stays self-contained.
- Missing technical details: The doc says to pin versions for reproducibility, but the install command does not pin any versions.
- Missing technical details: The dependency list is incomplete for a strict reproducible setup, because it omits packages such as `scikit-learn` and relies on preinstalled Colab packages for some others.
- Additional implementation risk: Environment assumptions such as exactly 2 CPU cores and perfectly stable `persistent_workers=True` behavior are runtime-dependent in Colab.

## Recommendations

- Either pin package versions or stop claiming version pinning.
- Expand the dependency instructions so every imported library is either installed explicitly or documented as expected from the base Colab runtime.
- Rephrase VRAM numbers and system-resource assumptions as approximate guidance.
- Keep W&B and similar tooling behind an explicit optional flag.
