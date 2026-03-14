# 01_System_Architecture.md Review

## Purpose

Provides the end-to-end pipeline diagram and key design decisions.

## Accuracy Score

`6/10`

## What Is Technically Sound

- Segmentation-first framing is correct.
- 384 resolution, 70/15/15 split, corruption guards, BCE + Dice, AdamW, and Grad-CAM are consistent with v6.
- The pipeline is readable and easy to follow.

## Issues Found

- The doc is written as if Kaggle is the only runtime.
- It still describes image-level detection using top-k mean instead of max pixel probability.
- It does not acknowledge the real v6 Colab notebook.

## Notebook-Alignment Notes

- Core training and preprocessing stages align with both v6 notebooks.
- Runtime description aligns only with Kaggle v6.

## Concrete Fixes or Follow-Ups

- Split this file into shared pipeline behavior plus Colab-specific and Kaggle-specific runtime notes.
