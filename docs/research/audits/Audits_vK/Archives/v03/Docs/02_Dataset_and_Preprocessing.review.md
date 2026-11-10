# 02_Dataset_and_Preprocessing.md Review

## Purpose

Defines CASIA v2.0 usage, dynamic discovery, binarization, splitting, manifest persistence, and dataset validation outputs.

## Accuracy Score

`9/10`

## What Improved Since Audit 2

- The discovery function now explicitly returns `(pairs, excluded)` instead of only printing exclusions.
- Unknown tampered filename patterns now warn rather than default silently.
- The split and manifest policy is explicit and reproducible.
- The validation summary clearly states what the notebook should print before training.

## Issues Found

- The data-leakage problem remains a known limitation because CASIA source-group metadata still does not exist.
- The code snippets are intentionally abbreviated, so they rely on surrounding context for variables such as `tp_dir`, `stem`, and `checkpoint_dir`.

## Suggested Improvements

- Keep the leakage limitation visible and avoid overclaiming generalization.
- If this doc is meant to be directly copyable, expand the snippets just enough to make the variable context self-contained.
