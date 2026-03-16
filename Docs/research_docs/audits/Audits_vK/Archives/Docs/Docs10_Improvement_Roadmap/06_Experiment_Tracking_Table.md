# 6. Experiment Tracking Table

Use this table to track every baseline or 10.x run.

| Experiment | Single Change | Val Pixel F1 (Tampered-Only) | Val IoU (Tampered-Only) | Test Pixel F1 (Tampered-Only) | Test IoU (Tampered-Only) | Image Accuracy | ROC-AUC | Runtime / GPU Notes | Merge Decision | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| Baseline Control (vK.7.1 executed) | none |  |  |  |  |  |  |  |  |  |
| 10.1 | Evaluation-hardened baseline |  |  |  |  |  |  |  |  |  |
| 10.2 | Missing-mask filter |  |  |  |  |  |  |  |  |  |
| 10.3 | Leakage-aware split |  |  |  |  |  |  |  |  |  |
| 10.4 | Localization-aware checkpointing |  |  |  |  |  |  |  |  |  |
| 10.5 | Boundary-preserving augmentations |  |  |  |  |  |  |  |  |  |
| 10.6 | Pretrained encoder |  |  |  |  |  |  |  |  |  |
| 10.7 | Seeded reproducible run |  |  |  |  |  |  |  |  |  |
| 10.8 | AMP training |  |  |  |  |  |  |  |  |  |
