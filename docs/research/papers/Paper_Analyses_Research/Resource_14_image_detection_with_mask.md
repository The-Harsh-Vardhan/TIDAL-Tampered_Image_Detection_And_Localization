# Resource 14: image-detection-with-mask.ipynb

## 1. Resource Overview
- Title: image-detection-with-mask
- Source: Kaggle notebook
- Author: Author not identified from local resource
- Year: Year not identified from local resource

This is the most assignment-aligned implementation artifact in the folder. It builds a tampered-image detection and localization pipeline around CASIA-style image and mask pairs, then trains a custom U-Net with a classification head.

## 2. Technical Summary
The notebook first scans Kaggle dataset folders to build image-mask metadata CSVs, then splits the data into train, validation, and test sets with stratification on the label. The model is a custom `UNetWithClassifier` that returns both `cls_logits` and `seg_logits`, so it predicts image-level authenticity and pixel-wise tampered regions in one forward pass.

The notebook contains two training variants. The earlier version uses `CrossEntropyLoss` for classification, `BCEWithLogitsLoss` for segmentation, Adam, and `ReduceLROnPlateau`. The later version improves the classification branch with focal loss and the segmentation branch with a 50-50 mix of BCE and Dice, then switches to `CosineAnnealingLR`. It also reports Dice, accuracy, and visualizes predicted masks. That is a legitimate assignment-shaped pipeline, even if the engineering is messy.

## 3. Key Techniques Used
- Custom dual-head U-Net for classification and segmentation
- Mask-aware dataset construction from Kaggle CASIA directories
- Focal loss plus BCE/Dice hybrid training
- Qualitative visualization of predicted tampered regions

These techniques are useful because they attack the actual assignment rather than dodging it with image-level classification.

## 4. Strengths of the Approach
The notebook understands the task shape correctly. It uses masks, trains for pixel output, and provides a learned image-level score through a classifier head instead of a crude heuristic like `max(prob_map)`.

It is also Kaggle-friendly. That matters because the assignment explicitly expects something that can run on commodity hosted GPU notebooks.

## 5. Weaknesses or Limitations
The engineering is sloppy. The notebook contains duplicated sections, evolving training code, dataset-path assumptions, and an earlier block where CSV assignments are clearly confused. That is not catastrophic, but it does mean the notebook needs cleanup before it deserves trust.

The model is also still plain by modern forensic standards. There are no explicit forensic side channels, no edge supervision, and no serious robustness evaluation. It is a competent baseline, not a research-grade solution.

## 6. Alignment With Assignment
Alignment: High

It directly supports tampered image detection and localization, uses a viable architecture, and is clearly intended for Kaggle-scale hardware. It misses depth in validation and robustness, but the task framing is correct.

## 7. Relevance to My Project
Useful parts:
- Dual-task head design
- Dataset-to-mask plumbing
- Practical loss combinations for localization

Less useful parts:
- Notebook duplication and inconsistent polish
- Treating this baseline as if it answers modern forensic research questions

## 8. Should This Be Used?
Use partially for inspiration.

The notebook is close enough to the assignment to influence the design, but not clean enough to be copied blindly.

## 9. Integration Ideas
- Reuse the dual-head idea to replace heuristic image-level detection.
- Keep the BCE plus Dice segmentation loss and focal classification loss as tested starting points.
- Port the data-loading logic into a cleaned, configuration-driven training pipeline with explicit evaluation splits and stronger metrics.

## 10. Citation
image-detection-with-mask. Local notebook copy: `Research Papers/image-detection-with-mask.ipynb`. Kaggle notebook. Author and year not identified from local resource.
