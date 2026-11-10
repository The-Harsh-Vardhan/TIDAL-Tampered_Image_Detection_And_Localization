# 3. Dataset and Data Pipeline Review

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

The runtime setup is practical for Kaggle and reasonably thoughtful about fallbacks. The notebook standardizes around `/kaggle/input` and `/kaggle/working`, supports Drive search in Colab, and falls back to Kaggle API download only when necessary (cells 6, 9, 11, 13, 15). That is a sensible operational design.

The metadata builder is simple and readable, but technically brittle. It assumes that every image has a mask with the exact same filename in the mirrored `MASK/<class>` directory (cell 23). That assumption is common, but it is not validated beyond filesystem existence.

The more serious issue is that rows with missing masks are still included in the metadata CSV. The notebook explicitly records `mask_exists` and even prints missing-mask examples (cells 23, 25), but there is no explicit filtering step before the downstream dataset loader reads `mask_path` and attempts `cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)` unconditionally (cell 59). If missing masks are present, the pipeline does not fail early and cleanly; it fails later during training or evaluation.

The split logic is only label-stratified random splitting (cell 27). This preserves the authentic/tampered ratio, which is good, but it does nothing to control for near-duplicates, derivative manipulations from the same source image, or source-level leakage. In image forensics, that matters. If multiple manipulated variants or closely related images are spread across train/val/test, measured performance can be overly optimistic.

The augmentation pipeline is a mixed bag. Resizing, horizontal flipping, brightness/contrast perturbation, and mild affine jitter are standard and defensible (cells 37, 57). The effective block adds Gaussian noise and JPEG compression (cell 57), which are actually relevant to forensic robustness. That said, these augmentations can also blur or distort the very traces the model is supposed to localize, especially around fine boundaries. They may help coarse tamper detection more than precise mask quality.
