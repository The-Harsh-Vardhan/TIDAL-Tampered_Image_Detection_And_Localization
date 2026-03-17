# 01 — Assignment Overview

## Purpose

This document defines the project scope, deliverables, and success criteria for the **Tampered Image Detection & Localization** assignment.

## Problem Statement

Given an image, determine:

1. **Classification** — Is the image tampered or authentic?
2. **Localization** — If tampered, which pixels were modified?

The output is a binary segmentation mask where tampered pixels are marked as 1 and authentic pixels as 0.

## Deliverables

| # | Deliverable | Format |
|---|---|---|
| 1 | Single Google Colab notebook | `.ipynb` with full implementation |
| 2 | Trained model weights | `.pt` checkpoint file |
| 3 | Dataset explanation | Inside notebook markdown cells |
| 4 | Model architecture description | Inside notebook markdown cells |
| 5 | Training strategy documentation | Inside notebook markdown cells |
| 6 | Hyperparameter choices and rationale | Inside notebook markdown cells |
| 7 | Evaluation results with visualizations | Inside notebook output cells |
| 8 | Colab notebook link with open access | Shared link |

## Constraints

- The entire project runs inside a **single Google Colab notebook**.
- The runtime target is a **T4 GPU** (15 GB VRAM, free Colab tier).
- No external services are required for the core submission.
- The dataset is **CASIA v2.0**, downloaded via Kaggle API.

## Bonus Points

The assignment awards bonus credit for:

- Robustness testing against JPEG compression, resizing, and noise.
- Detection of subtle tampering (copy-move from similar textures).

Bonus work is only attempted after the core submission is complete and stable.

## Success Criteria

| Metric | Minimum target | Notes |
|---|---|---|
| Pixel-F1 | 0.55–0.65 | Primary localization metric |
| Pixel-IoU | 0.45–0.55 | Region overlap metric |
| Image-level accuracy | 0.80–0.85 | Tampered vs. authentic classification |
| Image-level AUC | 0.80–0.85 | Threshold-independent detection quality |

These are conservative baselines. Higher scores are possible with optimization.

## Scope Boundaries

**In scope:**

- Dataset download, cleaning, and preprocessing
- U-Net segmentation model with pretrained encoder
- BCE + Dice loss training
- Pixel-level and image-level evaluation
- Visual result presentation
- Model checkpointing

**Out of scope for MVP:**

- HuggingFace dataset hosting or deployment
- Databricks, DuckDB, DynamoDB, or any database layer
- DALI data loading
- SegFormer or dual-stream architectures
- Large hyperparameter sweeps
- Multi-notebook or multi-script setups

## Related Documents

- [02_Dataset_and_Preprocessing.md](02_Dataset_and_Preprocessing.md) — Dataset details
- [10_Project_Timeline.md](10_Project_Timeline.md) — Implementation stages
- [11_Final_Submission_Checklist.md](11_Final_Submission_Checklist.md) — Pre-submission verification
