# 05_Evaluation_Methodology.md Review

## Purpose

Defines the metric set, threshold protocol, reporting views, and evaluation behavior.

## Accuracy Score

`5.5/10`

## What Is Technically Sound

- IoU, F1, precision, recall, and validation-only threshold selection are described clearly.
- True-negative consistency for empty authentic masks matches the v6 metric functions.

## Issues Found

- Image-level metrics are documented against `topk_mean(prob_map)`, but the v6 notebooks use max pixel probability.
- That affects both accuracy and AUC-ROC interpretation.

## Notebook-Alignment Notes

- Pixel-level metrics align with the v6 notebooks.
- Image-level scoring does not.

## Concrete Fixes or Follow-Ups

- Update the image-level metric section and any examples to the actual v6 scoring rule, or change the notebooks back to top-k mean.
