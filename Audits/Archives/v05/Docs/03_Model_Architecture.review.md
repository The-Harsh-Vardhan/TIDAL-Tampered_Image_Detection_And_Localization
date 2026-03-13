# 03_Model_Architecture.md Review

## Purpose

Defines the baseline network, image-level scoring rule, and optional extensions such as ELA, SRM, and dual-task classification.

## Accuracy Score

`8.7/10`

## What Is Technically Sound

- The baseline model definition matches the notebook exactly: `smp.Unet` with a ResNet34 encoder and single-channel output.
- The file honestly labels the image-level score as a top-k mean heuristic rather than a learned classifier.
- Optional architecture extensions are kept out of the MVP path, which prevents confusion about what is actually implemented.

## Issues Found

- The documented optional paths are credible, but they are design notes rather than implemented branches. That distinction is mostly clear and should stay explicit.
- The image-level decision remains the weakest part of the architecture because it is heuristic.

## Notebook-Alignment Notes

- The MVP baseline is fully aligned.
- The notebook uses the documented threshold-sharing rule for pixel and image-level decisions.

## Concrete Fixes or Follow-Ups

- If image-level performance becomes important, promote the dual-task classification-head option from future work into the next concrete implementation plan.
