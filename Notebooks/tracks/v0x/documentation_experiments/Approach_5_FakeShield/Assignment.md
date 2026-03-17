# Big Vision Internship Assignment:

# Tampered Image Detection & Localization

**Objective** Develop a deep learning model to detect and localize tampered (edited or
manipulated) regions in images. The model should not only classify whether an image is
tampered, but also generate a pixel-level mask highlighting altered regions. We are looking for
strong problem-solving skills, thoughtful architecture choices, and rigorous evaluation
methodologies.

## 1. Dataset Selection & Preparation

```
● Dataset Choice: Use one or more publicly available datasets containing original
(authentic) images, tampered images, and ground truth masks for localization.
Examples include the CASIA Image Tampering Dataset, Coverage Dataset, CoMoFoD
Dataset, or relevant Kaggle datasets.
● Data Pipeline: You are responsible for all dataset cleaning, preprocessing, and
ensuring mask alignment. Properly split your data into train, validation, and test sets.
● Augmentation: Apply relevant data augmentation techniques to ensure model
robustness.
```
## 2. Model Architecture & Learning

```
● Architecture: Train a model to predict tampered regions. The choice of architecture
and loss functions is entirely up to you.
● Resource Constraints: Optimize for performance while keeping the solution runnable
on Google Colab (T4 GPU compatible) or similar cloud platform.
```
## 3. Testing & Evaluation

```
● Performance Metrics: Thoroughly evaluate your model's localization performance and
image-level detection accuracy using standard, industry-accepted metrics.
● Visual Results: Provide clear visual results comparing the Original Image, Ground
Truth, Predicted output, and an Overlay Visualization.
```
## 4. Deliverables & Documentation

```
● The Code: The entire implementation must be done in a single Google Colab
Notebook. Ensure the notebook includes your dataset explanation, model architecture
description, training strategy, hyperparameter choices, evaluation results, and clear
visualizations.
● Assets: Provide the Colab Notebook Link (with appropriate access permissions),
trained model weights, and any additional scripts used.
```

## 🌟 Bonus Points For:

```
● Testing robustness against distortions such as JPEG compression, resizing, cropping,
and noise.
● Successfully detecting subtle tampering such as copy-move manipulation or splicing
from similar textures.
```