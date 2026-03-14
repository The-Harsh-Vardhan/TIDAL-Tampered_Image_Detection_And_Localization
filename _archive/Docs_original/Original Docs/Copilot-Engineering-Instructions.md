# Copilot Engineering Instructions — Tampered Image Detection Project

You are assisting with the development of a deep learning system for **tampered image detection and localization**.

Write code as if you are collaborating with a **Principal AI Engineer reviewing the work for production readiness and research clarity**.

Follow the guidelines below strictly.

---

# 1. General Engineering Philosophy

Prioritize:

• clarity over cleverness
• reproducibility over experimentation chaos
• modular design over monolithic scripts
• explainability over black-box results

All code should be **clean, readable, and logically organized**.

Avoid unnecessary complexity.

The goal is to produce a **well-structured research notebook + engineering pipeline**, not just a working model.

---

# 2. Project Objective

The system must:

1. Detect whether an image has been tampered.
2. Localize tampered regions using a **pixel-level segmentation mask**.

The model should output:

* segmentation mask
* optional tampered probability score.

The solution must run efficiently on **Google Colab with a T4 GPU**.

---

# 3. Dataset Handling Rules

Datasets may include authentic and tampered images.

Strict requirements:

1. Each image must be paired with a mask.
2. Authentic images must have a **zero mask**.
3. Mask values must be binary.

Example:

0 → authentic pixel
1 → tampered pixel

When resizing masks, always use **nearest neighbor interpolation**.

Never use bilinear interpolation for masks.

---

# 4. Data Pipeline Requirements

The data pipeline must be structured as:

load image
load mask
resize image and mask
apply augmentations
normalize image
convert to tensor

Augmentations must always be applied **identically to image and mask**.

Use a dedicated dataset class rather than embedding preprocessing inside training code.

---

# 5. Augmentation Strategy

Include augmentations that improve robustness.

Geometric:

* horizontal flip
* rotation
* random crop

Photometric:

* brightness
* contrast

Forensic robustness:

* JPEG compression simulation
* gaussian noise
* blur

These augmentations simulate real-world tampering artifacts.

---

# 6. Model Architecture Expectations

Use a segmentation architecture with:

encoder → feature extractor
decoder → mask reconstruction

Preferred structure:

Image
→ optional residual/high-pass filtering
→ pretrained encoder (ResNet/EfficientNet)
→ U-Net style decoder
→ segmentation mask

If possible, include a **classification head** for image-level tamper detection.

Design architecture to be **modular and easily swappable**.

---

# 7. Loss Functions

Segmentation must use a loss suitable for **class imbalance**.

Preferred combination:

Binary Cross Entropy + Dice Loss

Avoid using accuracy as the primary optimization metric.

---

# 8. Evaluation Metrics

Compute the following metrics:

Localization metrics:

* Intersection over Union (IoU)
* Dice Score
* Pixel Accuracy

Image-level metrics:

* Precision
* Recall
* F1 Score

All evaluation functions should be written clearly and tested independently.

---

# 9. Training Pipeline

Training code must be modular and structured.

Separate responsibilities:

dataset loading
model initialization
training loop
validation loop
metric computation

Include:

* progress logging
* epoch summaries
* checkpoint saving.

Avoid mixing training logic with visualization code.

---

# 10. Visualization Requirements

The project must generate clear visual outputs.

For test images display:

Original Image
Ground Truth Mask
Predicted Mask
Overlay Visualization

Overlay should highlight tampered regions clearly.

Visual inspection is a critical debugging tool.

---

# 11. Robustness Experiments

Include tests showing model behavior under distortions.

Simulate:

JPEG compression
image resizing
noise injection

Measure performance degradation using IoU or Dice score.

These experiments demonstrate real-world reliability.

---

# 12. Debugging Practices

Always include sanity checks:

visualize random dataset samples
verify mask alignment
confirm mask value distribution

If masks are misaligned or corrupted, stop training immediately.

Data correctness is more important than model complexity.

---

# 13. Code Style Expectations

Follow these standards:

• descriptive variable names
• small reusable functions
• minimal global variables
• clear comments explaining reasoning

Avoid overly nested logic.

Prefer readability.

---

# 14. Notebook Organization

The notebook must read like a **technical report**.

Structure it as:

Introduction
Dataset Description
Preprocessing
Augmentation
Model Architecture
Training Strategy
Evaluation Metrics
Results
Robustness Experiments
Limitations
Conclusion

Each section must contain short explanations.

---

# 15. Experiment Logging

Record important training information:

model configuration
learning rate
batch size
loss curves
evaluation metrics

This allows reproducibility.

---

# 16. Performance Constraints

The system must train within reasonable limits on:

Google Colab T4 GPU

Avoid architectures that require excessive memory or multi-GPU training.

Prefer efficient pretrained encoders.

---

# 17. What Not to Do

Do not:

train without validating masks
use only accuracy as metric
skip qualitative visualization
produce code without explanation
overcomplicate the architecture

Engineering discipline matters more than model novelty.

---

# Final Principle

A good submission demonstrates:

clear thinking
clean engineering
reproducible experiments
honest evaluation

Write code that another engineer can read and immediately understand.