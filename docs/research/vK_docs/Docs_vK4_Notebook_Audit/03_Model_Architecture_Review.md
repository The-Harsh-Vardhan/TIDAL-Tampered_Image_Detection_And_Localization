# 03 - Model Architecture Review

## What Was Implemented

- Custom U-Net style encoder-decoder (DoubleConv, Down, Up blocks)
- Segmentation head for tampered region mask
- Classification head on bottleneck for image-level tamper detection

## Assignment Fit

For this assignment, a U-Net style model is a valid choice. Pixel localization is the core task and skip connections are useful for spatial detail recovery.

## Architecture Reasoning Quality

## Strengths

1. Shared backbone for two related tasks is reasonable.
2. Joint detection + localization aligns with assignment scope.

## Weaknesses

1. Reasoning is mostly descriptive, not comparative.
2. No evidence that classifier head improves segmentation or detection stability.
3. No baseline comparison against pretrained encoder variants.

## About ResNet34 Encoder

This notebook does **not** use ResNet34 as encoder. It uses a custom encoder stack. If the candidate verbally claims ResNet34 rationale, that claim is inconsistent with implementation.

## Is the architecture technically sound

- Yes, as a baseline.
- No, as a fully justified final architecture, because the decision process is under-evidenced.

## Senior-level expectation

1. One ablation: with vs without classifier head.
2. One comparison: custom encoder vs pretrained encoder baseline.
3. A short tradeoff statement: quality, speed, memory.

## Verdict

- Appropriateness for assignment: **Yes**
- Reasoning quality: **Partial**
- Interview signal: **mid-level implementation skill, weak decision validation discipline**
