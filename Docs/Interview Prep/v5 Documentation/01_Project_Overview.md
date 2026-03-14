# Project Overview

## How to explain this in an interview

Start with this:

"This project detects whether an image has been tampered with and also localizes the manipulated region at the pixel level. I built it as a segmentation problem using a U-Net with a ResNet34 encoder, because I wanted one model that could both highlight altered regions and support an image-level tamper decision."

## What the project does

This system takes an input image and produces three useful outputs:

- a pixel-level tamper mask
- an image-level tampered vs authentic decision
- visual overlays showing where the model thinks manipulation happened

The goal is not just to say "this image looks suspicious." The goal is to show where the suspicious region is.

## Problem statement

Tampered image detection is the task of identifying whether an image has been edited in a misleading way. Tampered image localization goes one step further and highlights the manipulated pixels.

That matters because many real use cases need more than a binary label:

- media verification
- document fraud analysis
- digital forensics
- insurance claim review
- content moderation

A binary classifier alone cannot show evidence. A localization model gives a much stronger explanation.

## Motivation

I chose this problem because it combines computer vision, anomaly detection, and practical ML engineering. It is also a good interview project because it shows:

- data pipeline design
- segmentation modeling
- careful evaluation
- explainability
- robustness testing
- reproducibility under limited hardware

## System overview

At a high level, the system works like this:

1. Load a CASIA-based tampering dataset with image-mask pairs.
2. Validate the pairs, clean invalid samples, and create reproducible train, validation, and test splits.
3. Train a segmentation model to predict a tamper probability map.
4. Convert that probability map into a binary mask using a threshold chosen on the validation set.
5. Derive an image-level tamper score from the predicted map.
6. Evaluate localization quality, detection quality, explainability outputs, and robustness under image degradation.

## Why the project is framed as segmentation

The real target is localization. If I only trained a classifier, I would know whether the image might be tampered, but I would not know where.

A segmentation model solves two problems at once:

- it predicts the tampered region directly
- it gives a probability map that can also support image-level detection

That makes the pipeline simpler and better aligned with the assignment.

## Final pipeline

The final v5 pipeline is:

1. Download the CASIA Splicing Detection + Localization dataset.
2. Discover tampered and authentic samples dynamically.
3. Validate readability, mask alignment, and mask format.
4. Binarize masks and store a persistent split manifest.
5. Train `smp.Unet` with a ResNet34 encoder on RGB images.
6. Use BCE + Dice loss and AdamW with mixed precision training.
7. Select the best threshold on the validation set only.
8. Evaluate with F1, IoU, precision, recall, image accuracy, and AUC-ROC.
9. Generate prediction grids, overlays, Grad-CAM views, and robustness results.
10. Optionally track the experiment with Weights & Biases.

## Why this is a strong baseline

I would describe this system as a strong baseline, not a state-of-the-art forensic model.

It was chosen because it is:

- technically sound
- reproducible
- explainable enough for debugging
- feasible on a single Colab T4 GPU
- easy to discuss clearly in an interview

## Key tradeoffs

Some design choices are intentionally pragmatic:

- I used a CNN-based U-Net instead of a transformer because the dataset is relatively small and the compute budget is limited.
- I used one segmentation model instead of a separate detector plus localizer because it simplifies training and interpretation.
- I used a top-k mean image score instead of a dedicated classification head because it keeps the MVP lightweight, even though a learned classifier could be stronger later.

## Real-world relevance

Tamper detection systems need to survive imperfect real-world conditions:

- recompressed images
- noisy uploads
- resized social-media images
- subtle copy-move edits

That is why this project includes not only segmentation and detection, but also robustness testing and visual diagnostics.

## How I would close this explanation

"The main idea is that I treated tamper analysis as a segmentation problem first. That let me build a model that produces interpretable masks, then reuse the same probability map for image-level detection. From an interview perspective, the interesting part is not just the model choice, but how the whole pipeline was made reproducible, explainable, and practical on limited hardware."
