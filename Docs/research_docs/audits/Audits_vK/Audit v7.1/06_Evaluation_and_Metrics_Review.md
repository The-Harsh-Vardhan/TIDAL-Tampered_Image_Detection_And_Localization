# 6. Evaluation and Metrics Review

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

The notebook separates detection and localization conceptually, but the evaluation design is incomplete.

## Detection Performance

Detection is effectively measured by accuracy only (cells 69, 71, 73). That is too weak for a binary forensic classifier. The notebook does not report precision, recall, F1, confusion matrix, ROC-AUC, or PR-AUC. Accuracy alone can hide class-asymmetric failure modes and does not tell the reviewer whether the classifier is conservative, over-sensitive, or badly calibrated.

## Localization Performance

Localization metrics are better in intent. Dice, IoU, and F1 are all implemented and reported from thresholded masks at `0.5` (cells 67, 69, 73). That is a reasonable starting point.

But there are two technical weaknesses:

1. Metrics are averaged at the batch level rather than aggregated over the full dataset, which introduces small weighting distortions when batch sizes differ.
2. Localization metrics are computed over all samples in the split, including authentic images with empty masks (cell 69).

That second issue is the major flaw. If the model predicts empty masks on authentic images, Dice/IoU/F1 can be inflated even if tampered-region localization is mediocre. The notebook does not report tampered-only localization metrics, which would be much more meaningful.

## Demonstrated Evidence

Even the existing metric design is not convincingly demonstrated in the saved artifact. The effective training loop, final test evaluation, and training-curves cells are unexecuted (cells 71, 73, 75), so the notebook does not actually display the final quantitative evidence it claims to provide.
