# 03_Model_Architecture.md Review

## Purpose

Explains the baseline model, image-level detection strategy, and future encoder options.

## Accuracy Score

`5/10`

## What Is Technically Sound

- `smp.Unet(resnet34)` is documented correctly as the baseline.
- The segmentation framing and pretrained-backbone rationale are sound.

## Issues Found

- The image-level detection description is wrong for v6: the notebooks use max pixel probability, not top-k mean.
- The doc does not meaningfully compare the baseline against ViT or DeepLabV3, which weakens interview readiness.
- Runtime language is still Kaggle-centric.

## Notebook-Alignment Notes

- Model instantiation aligns with both v6 notebooks.
- Image-level detection logic does not align.

## Concrete Fixes or Follow-Ups

- Correct the scoring rule and add a tradeoff paragraph for ResNet34 vs EfficientNet vs DeepLabV3 vs ViT.
