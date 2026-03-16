# 02_Dataset_and_Preprocessing.md Review

## Purpose

Defines CASIA v2.0 usage, cleaning, validation, mask binarization, and data splitting for the project.

## Accuracy Score

`8/10`

## What Improved Since Audit 1

- Removed hard-coded image counts and hard-coded misaligned-pair counts.
- Added dynamic alignment validation and explicit logging of excluded pairs.
- Documented the exact two-step 85 / 7.5 / 7.5 split procedure.
- Added split-manifest persistence and a clear statement of the group-splitting limitation.

## Remaining Issues

- The leakage problem is only partially resolved because CASIA source groupings are still unavailable.
- The example discovery function says it returns excluded counts, but the shown function returns only `pairs` and prints exclusions instead of returning them.
- Forgery type inference still defaults to `copy-move` for any tampered filename that does not contain `'_D_'`, which is acceptable for CASIA naming but slightly brittle.

## Suggested Improvements

- Return the excluded-pairs list or a manifest object from the discovery function instead of only printing it.
- Add a guard for unexpected tampered filename patterns so mislabeled or malformed files are not silently treated as copy-move.
