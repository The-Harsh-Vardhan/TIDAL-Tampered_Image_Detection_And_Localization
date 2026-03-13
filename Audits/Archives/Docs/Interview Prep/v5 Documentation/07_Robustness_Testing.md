# Robustness Testing

## How to explain this in an interview

Start with this:

"Tampered images are rarely seen in perfect form in the real world. They get compressed, resized, and re-uploaded, so I added robustness tests to check whether the model still works after common image degradations."

## Why robustness matters

In a clean benchmark setting, a model may perform well on the exact format it was trained on. Real deployments are messier.

Tampered images are often:

- saved as JPEG multiple times
- resized by social-media platforms
- blurred by image pipelines
- contaminated with sensor or upload noise

If the model fails under those conditions, then its benchmark score is not enough.

## What robustness testing means here

The project evaluates the trained model on degraded copies of the test images while keeping the masks unchanged.

That setup answers a practical question:

"Can the model still localize the tampered region after realistic image corruption?"

## Degradations used in this project

The current robustness suite includes:

- JPEG compression
- Gaussian noise
- Gaussian blur
- resizing

## JPEG compression

### What it is

The image is recompressed at lower quality.

### Why it matters

Tampered images are often shared in compressed form. Compression can remove subtle forensic clues or create new artifacts that confuse the model.

### Why it was included

JPEG robustness is one of the most realistic tests for tamper detection.

## Gaussian noise

### What it is

Random noise is added to the image.

### Why it matters

Noise can hide weak manipulation boundaries or make the model fire on irrelevant patterns.

### Why it was included

It is a simple way to test whether the model depends too heavily on fragile low-level texture cues.

## Blur

### What it is

The image is smoothed with Gaussian blur.

### Why it matters

Tamper localization often relies on edges, boundaries, and local inconsistencies. Blur weakens those clues.

### Why it was included

It is a good stress test for models that depend on fine spatial detail.

## Resizing

### What it is

The image is downscaled and restored or otherwise degraded through resizing operations.

### Why it matters

Many real image pipelines resize images automatically. That can distort the exact local evidence the model relies on.

### Why it was included

It tests whether the model is learning robust structure or only fragile pixel-level artifacts.

## How the current tests are structured

The design is intentionally clean:

1. Train on the normal pipeline.
2. Choose the threshold on the validation set.
3. Freeze that threshold.
4. Evaluate on degraded test images only.
5. Keep masks unchanged.

That matters because it avoids accidentally tuning the model specifically for the degraded data.

## Why the same threshold is reused

This is a good interview detail.

If I retuned the threshold for every degradation, I would make the model look more robust than it really is.

By reusing the validation-selected threshold, the robustness test stays honest.

## Alternatives that could have been used

- cropping
- color jitter
- brightness shifts
- stronger compression sweeps
- adversarial perturbations
- cross-dataset robustness tests

## Why those were not selected in the MVP

The current goal was to cover the most realistic and common degradations without making the robustness section larger than the core training pipeline.

Cropping is a good example:

- it is useful
- it was mentioned in the assignment bonus ideas
- but it was not necessary to prove the main robustness concept in the first version

## What robustness results tell me

Robustness testing helps answer:

- Does performance drop sharply under compression?
- Is the model relying on brittle pixel patterns?
- Which degradation hurts it most?
- Is the system usable outside a clean benchmark?

That is very valuable in interviews because it shows the project was evaluated like a real system, not just a leaderboard exercise.

## Future improvements

If I expanded robustness testing, I would add:

- cropping
- stronger JPEG sweeps
- multiple resize scales
- color and illumination shifts
- cross-dataset evaluation
- robustness-aware training augmentations

## How I would summarize robustness testing

"I treated robustness as a realism check. The model was evaluated on JPEG-compressed, noisy, blurred, and resized versions of the test images using the same threshold selected on the clean validation set. That tells me whether the system can survive common real-world image transformations rather than only working on ideal benchmark inputs."
