# 08_Engineering_Practices.md Review

## Purpose

Documents environment setup, dependency installation, artifact handling, reproducibility controls, safety checks, and notebook structure.

## Accuracy Score

`9/10`

## What Is Technically Sound

- The file now accurately describes Drive-or-local artifact behavior, dependency needs, split persistence, deterministic loaders, and the current 17-section notebook map.
- Reproducibility guidance is materially improved over earlier versions because manifest reuse and worker seeding are both documented.
- The artifact list matches the notebook's current saved files.

## Issues Found

- The document itself is aligned.
- The only residual issue is implementation-side: the notebook's last print statement still implies Google Drive even under local fallback.

## Notebook-Alignment Notes

- Section mapping and artifact names match `tamper_detection_v5.ipynb`.
- The notebook structure table correctly points to the v5 notebook rather than the older version.

## Concrete Fixes or Follow-Ups

- Fix the notebook print statement so this document and the runtime output say the same thing about artifact location.
