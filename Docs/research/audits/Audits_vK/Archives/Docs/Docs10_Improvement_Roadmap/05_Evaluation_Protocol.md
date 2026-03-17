# 5. Evaluation Protocol

All 10.x notebooks must use the same frozen comparison pipeline unless the experiment itself changes the split policy.

## Segmentation Metrics

- `Pixel F1` on tampered images only
- `IoU` on tampered images only
- Optional auxiliary metric: overall Dice

## Detection Metrics

- Image-level accuracy
- ROC-AUC

## Required Visual Outputs

Every experiment must show:

- Original image
- Ground truth mask
- Predicted mask
- Overlay visualization

## Frozen Constants

These must remain fixed across experiments unless the notebook's single change is the split policy:

- same dataset source
- same image resolution
- same training budget
- same test set
- same mask threshold `0.5`

## Qualitative Comparison Rule

Use the same saved sample IDs from the baseline control run for all final visual panels so qualitative comparisons are directly comparable.

## Artifact Completeness Rule

A notebook run is invalid unless:

- metrics are printed and saved in the notebook outputs
- final visual panels are rendered and saved in the notebook outputs
- the saved `.ipynb` artifact preserves those outputs
