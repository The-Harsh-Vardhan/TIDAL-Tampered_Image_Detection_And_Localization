# 5. Training Pipeline Audit

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

The effective training stack is reasonable at a baseline level. Adam with learning rate `1e-4`, CosineAnnealingLR, a focal-style classification loss, BCE-with-logits plus Dice for segmentation, and gradient clipping are all defensible choices (cells 65, 69). Batch size 8 and `256x256` inputs are realistic for Kaggle/Colab hardware (cell 63).

## Strengths

- Losses are appropriate for a binary classification plus binary segmentation setup.
- Class weighting is used for the classifier branch (cell 65).
- Gradient clipping is included for some stability (cell 69).
- The reporting metrics cover more than just loss on the segmentation side (cells 67, 69).

## Weaknesses

- No mixed precision or AMP is used, so efficiency is worse than it should be.
- No global seed is set for Python, NumPy, or PyTorch.
- No early stopping is used.
- Checkpoint selection is based only on validation accuracy (cell 71), not Dice/IoU/F1 or a multitask criterion.

That last point is the most important one. For a dual-task system, saving the model that maximizes classification accuracy is not the same thing as saving the model that best localizes tampered regions. The notebook's training logic can therefore produce a model that looks good on image-level detection while underperforming on the pixel-level task the assignment explicitly requires.
