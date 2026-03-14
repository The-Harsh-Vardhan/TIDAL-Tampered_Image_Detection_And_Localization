# 7. Priority Fixes

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

1. Validate the segmentation metrics on tampered-only samples and on a small hand-checked batch before trusting any reported Dice or IoU.
2. Stop selecting checkpoints by validation accuracy alone. Use a localization-aware criterion, or at minimum a multitask score with a real localization component.
3. Separate detection and localization reporting cleanly so the classifier cannot hide a weak mask predictor.
4. Remove or quarantine the legacy duplicate pipeline so only one authoritative training path remains.
5. Rerun the improved experiment to completion and save final test metrics from a genuinely independent test split.
6. Clean up experiment tracking, dependency warnings, and version discipline so the next log reads like one experiment instead of three half-overlapping stories.
