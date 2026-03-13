# Dataset Explanation

## How to explain this in an interview

Start with this:

"I used the CASIA Splicing Detection + Localization dataset because it already provides tampered images, authentic images, and localization masks in a structure that is practical for a single-notebook training pipeline."

## What dataset was used

The project uses a Kaggle-hosted CASIA localization dataset derived from CASIA-style image forgery data. It includes:

- tampered images
- authentic images
- pixel-level ground-truth masks for tampered samples

The main tampering categories represented in the project are:

- splicing
- copy-move

That makes it a good fit for a localization task, because the model can learn to predict exactly where the manipulated region is.

## What problem this dataset solves

To train a localization model, I need more than class labels. I need paired supervision:

- the image
- the mask showing the manipulated region

Without masks, I could train a classifier, but not a localizer. This dataset solves that by giving the project direct pixel-level supervision.

## Dataset structure

The project uses a dataset layout with clear `Image/` and `Mask/` directories. The notebook discovers image-mask pairs dynamically rather than hardcoding filenames.

Conceptually, the dataset pipeline expects:

- tampered image files
- matching tamper masks
- authentic images that are assigned zero masks

This matters because it keeps the notebook robust to minor folder-layout changes and makes the preprocessing pipeline easier to maintain.

## Preprocessing pipeline

The preprocessing pipeline does more than just resize images. It also protects the training run from bad data.

### 1. Dynamic discovery

The notebook scans the dataset folders and builds a structured record for each sample:

- image path
- mask path if it exists
- image-level label
- forgery type

### 2. Validation

Each pair is checked for:

- readable image file
- readable mask file for tampered samples
- matching spatial dimensions
- known tampering category

Invalid samples are excluded before training.

### 3. Mask binarization

Masks are converted into binary targets. This matters because the model is solving a binary segmentation problem:

- tampered pixel = 1
- authentic pixel = 0

### 4. Authentic image handling

Authentic images do not have tamper masks, so the pipeline creates all-zero masks for them. That allows the model to learn both:

- where tampering exists
- what a clean image looks like

### 5. Train, validation, and test split

The split is stratified and persisted to `split_manifest.json`, so the same split can be reused on later runs. That improves reproducibility and makes comparisons across experiments fairer.

## Why this dataset was chosen

I chose this dataset for practical reasons, not just because CASIA is a well-known name.

It was a good fit because:

- it supports localization, not just classification
- it is manageable on Colab-scale hardware
- it covers classical tampering categories relevant to the assignment
- it has a structure that works well with an automated notebook pipeline

In interview terms, this was the best balance between usefulness and implementation speed.

## Alternatives that could have been used

### DocTamper

What it is:
DocTamper is a document-focused tampering dataset.

What problem it solves:
It is useful if the project is about document fraud rather than general natural-image tampering.

Why I did not use it here:
This project is framed around general image tampering and localization, not only document images. Using a document-specific dataset would narrow the scope too early.

### IMD2020

What it is:
IMD2020 is a more modern image manipulation dataset often discussed in research settings.

What problem it solves:
It is useful for broader or harder tampering scenarios and can support more ambitious benchmarking.

Why I did not use it here:
It increases dataset-management complexity and is less convenient for a lightweight Colab-first baseline. For this project, CASIA was the more practical starting point.

### Coverage and CoMoFoD

What they are:
These are useful datasets for copy-move and related forgery analysis.

Why they were not selected:
They are strong alternatives for specialized evaluation, but the chosen CASIA pipeline already covered the main assignment need with a simpler setup.

## Why the alternatives were not selected

The main reason is project scope.

I optimized for:

- one clean notebook
- single-GPU training
- paired localization masks
- a reproducible baseline

CASIA was not the only possible choice, but it was the best choice for the assignment constraints.

## Dataset limitations

This is one of the most important things to say in an interview.

The dataset has real limitations:

- It is relatively small by modern deep learning standards.
- It focuses on classical tampering, not AI-generated manipulation.
- It may contain annotation noise or coarse boundaries.
- CASIA does not expose source-image grouping metadata, so related images may leak across splits even when the split is reproducible.

That last point is especially important. The project improves reproducibility with a saved split manifest, but that does not fully solve data leakage risk if related source content appears in multiple splits.

## Future improvements

If I extended the project, I would improve the data side by:

- adding larger and more diverse datasets
- testing cross-dataset generalization
- incorporating harder post-processed samples
- adding better label-quality checks
- evaluating newer manipulation categories

## How I would summarize the dataset choice

"I used CASIA because it gave me a practical localization dataset with image-mask pairs, which is exactly what a segmentation pipeline needs. It is not perfect, and I would openly mention the leakage and dataset-size limitations, but it was a strong choice for building a reproducible baseline under Colab constraints."
