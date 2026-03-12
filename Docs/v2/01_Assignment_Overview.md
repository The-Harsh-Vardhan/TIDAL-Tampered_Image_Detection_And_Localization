# 01 — Assignment Overview

## Purpose

Define the project scope, deliverables, and constraints for the **Tampered Image Detection & Localization** assignment.

## Problem Statement

Given an image, the system must:

1. **Classify** — Is the image tampered or authentic?
2. **Localize** — If tampered, produce a pixel-level binary mask highlighting altered regions.

## Deliverables

| # | Deliverable | Format |
|---|---|---|
| 1 | Single Google Colab notebook | `.ipynb` — complete implementation |
| 2 | Trained model weights | `.pt` checkpoint file |
| 3 | Dataset explanation | Markdown cells in notebook |
| 4 | Model architecture description | Markdown cells in notebook |
| 5 | Training strategy and hyperparameter choices | Markdown cells in notebook |
| 6 | Evaluation results with visualizations | Output cells in notebook |
| 7 | Colab notebook link (open access) | Shared URL |
| 8 | Any additional scripts used | Attached or linked |

Source: Assignment.md, Section 4.

## Constraints

- The entire project runs inside a **single Google Colab notebook**.
- The runtime target is a **T4 GPU** (free Colab tier).
- The dataset is **CASIA v2.0**, selected as the project dataset. The assignment permits any public dataset with authentic images, tampered images, and ground truth masks.
- **Kaggle API** is used for dataset download — this is a workflow tool, not a deployment dependency.
- **Google Drive** is used for checkpoint persistence across Colab sessions — also a workflow tool.
- No deployment platform, hosted inference service, or external MLOps infrastructure is required.

## Bonus Points

The assignment awards bonus credit for:

- Robustness testing against JPEG compression, resizing, cropping, and noise.
- Detection of subtle tampering (copy-move manipulation, splicing from similar textures).

Bonus work is only attempted after the core submission is complete and stable.

## Scope Boundaries

**In scope:**

- Dataset download, cleaning, preprocessing
- U-Net segmentation with pretrained encoder
- BCE + Dice loss
- Pixel-level and image-level evaluation
- Visual results (original, ground truth, predicted mask, overlay)
- Model checkpointing

**Out of scope:**

- HuggingFace dataset hosting or deployment
- Databricks, DuckDB, DynamoDB, or any database layer
- NVIDIA DALI data loading
- SegFormer or dual-stream architectures
- Large hyperparameter sweeps
- Multi-notebook or multi-script setups

## Related Documents

- [02_Dataset_and_Preprocessing.md](02_Dataset_and_Preprocessing.md) — Dataset details
- [10_Project_Timeline.md](10_Project_Timeline.md) — Implementation stages
- [12_Final_Submission_Checklist.md](12_Final_Submission_Checklist.md) — Pre-submission verification
