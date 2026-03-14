# 01_Assignment_Overview.md Review

## Purpose

Defines project scope, deliverables, constraints, success criteria, and scope boundaries for the consolidated final plan.

## Technical Accuracy Score

`7/10`

## What Is Correct

- Technical correctness: The core framing matches the assignment well: image-level tamper detection plus pixel-level localization in a single Colab notebook.
- Implementability on a single Colab notebook with T4: The stated baseline scope is feasible if the implementation stays close to RGB-only U-Net plus standard training and evaluation.
- Assignment alignment: The document correctly centers CASIA, segmentation, checkpointing, evaluation, and visual outputs instead of enterprise tooling or deployment.

## Issues Found

- Contradictions: The statement "No external services are required for the core submission" conflicts with the recommended Kaggle API and Google Drive workflow in `09_Engineering_Practices.md`.
- Unsupported or hallucinated claims: The exact target ranges for Pixel-F1, Pixel-IoU, image accuracy, and image AUC are unverified. They read like benchmarks, but no measured evidence is provided in the repo.
- Unnecessary complexity: The document is much cleaner than the older docs, but it still turns chosen project decisions into stronger requirements than the assignment itself requires, especially the fixed dataset and exact target metrics.
- Missing technical details: It does not clarify whether the listed metric targets are aspirational, historical, or mandatory acceptance criteria.
- Additional implementation risk: It states CASIA via Kaggle API as a fixed constraint, even though that is a project choice rather than an assignment requirement.

## Recommendations

- Rephrase the metric bands as informal expectations or remove them until they are backed by actual notebook results.
- Clarify that CASIA is the selected project dataset, not a requirement imposed by the assignment prompt.
- Change the "no external services" line to mean no deployment platform or hosted inference service is required, while Kaggle and optional Drive are part of the practical workflow.
