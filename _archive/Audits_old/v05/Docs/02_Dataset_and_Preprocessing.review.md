# 02_Dataset_and_Preprocessing.md Review

## Purpose

Documents dataset selection, Kaggle download, dynamic pair discovery, validation rules, mask binarization, and split persistence.

## Accuracy Score

`9/10`

## What Is Technically Sound

- It correctly names the CASIA Kaggle localization dataset and explains why that dataset is practical for the assignment.
- The discovery logic description matches the notebook's handling of tampered pairs, authentic images, corrupt files, unknown forgery types, and dimension mismatches.
- Split-manifest reuse is documented clearly and now matches implementation.

## Issues Found

- No material dataset-pipeline mismatch was found.
- The unavoidable limitation remains source-image leakage risk because CASIA lacks grouping metadata.

## Notebook-Alignment Notes

- `split_manifest.json` is described as a reusable artifact, matching the notebook.
- The validation and exclusion rules align with the notebook's current discovery code.

## Concrete Fixes or Follow-Ups

- If future versions add dataset statistics, include explicit counts of excluded corrupt or invalid files in the docs as well as in notebook logging.
