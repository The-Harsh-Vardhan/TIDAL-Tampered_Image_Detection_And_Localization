# 12. Improvement Recommendations

Review target: `vK.7.1 Image Detection and Localisation.ipynb`

## High impact improvements

- Execute the effective training loop, final test evaluation, training-curves cell, and the final four-panel visualization cell, then save the notebook with those outputs present.
- Change checkpoint selection from validation accuracy to a criterion that reflects localization quality, such as validation Dice on tampered images or a multitask score.
- Report tampered-only localization metrics so empty authentic masks do not inflate the apparent segmentation quality.
- Add stronger image-level detection metrics: precision, recall, F1, confusion matrix, and ROC-AUC.

## Medium improvements

- Filter or repair `mask_exists == 0` rows before training, rather than recording them and failing later in the dataset loader.
- Add proper seeding for Python, NumPy, and PyTorch.
- Remove or isolate the duplicated earlier experiment block so only one authoritative training path remains.
- Replace remaining hardcoded `/kaggle/working/...` strings with the notebook's path variables for consistency.

## Nice-to-have improvements

- Add AMP for faster, more memory-efficient training on Kaggle and Colab GPUs.
- Add a short failure-analysis section with representative false positives and false negatives.
- Consider a pretrained encoder or a lightweight forensic-aware feature strategy if you want stronger performance without redesigning the project.
- Add a brief discussion of leakage risk and why the split strategy is acceptable or limited.
