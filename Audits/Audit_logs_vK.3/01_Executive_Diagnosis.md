# 1. Executive Diagnosis

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

The headline is simple: classification improves, but the localization evidence is not trustworthy enough to support strong claims. The log shows a model that can learn image-level tamper discrimination reasonably well, but the mask metrics are either broken, inflated, miswired, or being reported through an evaluation setup that hides the real localization quality.

The first run is not a credible benchmark. It appears to use the wrong split wiring, it reports a validation Dice that freezes at `0.5949` for almost the entire run, and its final "test" result is too entangled with the validation setup to be treated as an independent holdout measure. The second run looks more serious on the classification side, climbing to `0.8864` validation accuracy by epoch 30, but the segmentation metrics remain erratic, underwhelming, and incomplete because the run is truncated before a final test summary. In plain language: the logs support "the detector learns something," not "the localization pipeline is solid."
