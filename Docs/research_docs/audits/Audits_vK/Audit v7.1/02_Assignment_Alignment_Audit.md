# 2. Assignment Alignment Audit

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

## Dataset Selection

The notebook uses a CASIA-style authentic/tampered dataset with masks, discovered through a Kaggle-first pipeline with optional Colab/Drive and Kaggle API fallbacks (cells 6, 9, 13, 15, 21, 23, 25). This satisfies the assignment's dataset requirement in principle.

## Mask Usage for Localization

Mask usage is implemented correctly at a basic level. The dataset class loads grayscale masks, binarizes them, and returns them alongside the input image and image-level label (cell 59). The segmentation head produces a one-channel output map and is trained with BCE-plus-Dice supervision (cells 61, 65, 69).

## Segmentation Model Correctness

The model is a legitimate dual-task CNN for this assignment. It uses a shared U-Net-like backbone with a bottleneck classifier head for image-level prediction and a decoder head for localization (cell 61). That satisfies the requirement that the system perform both image-level detection and pixel-level localization.

## Evaluation Metrics

Quantitative metrics are implemented in code. The notebook computes image-level accuracy plus Dice, IoU, and F1 for segmentation (cells 67, 69, 73). However, the effective training and final test evaluation cells are not executed in the saved notebook (cells 71, 73, 75), so those metrics are implemented but not demonstrated in the artifact.

## Required Visual Outputs

The required side-by-side visualization of original image, ground-truth mask, predicted mask, and overlay exists in code in the submission-ready panel function (cell 90). But that final cell is unexecuted and has no output in the saved artifact. Earlier qualitative visualization cells are executed and show overlays or predicted masks (cells 82, 84, 87), but they do not fully substitute for the exact final assignment panel.

## Verdict

**PARTIALLY ALIGNED**

Explanation: the notebook contains the required dataset logic, mask handling, multitask model, and evaluation/visualization code, but the saved notebook artifact does not actually show the key quantitative results or the final required four-panel visualization. As code it is aligned. As a submission artifact it is only partially aligned.
