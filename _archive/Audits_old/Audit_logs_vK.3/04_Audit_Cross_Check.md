# 4. Audit Cross-Check

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

The prior `Audit-vK.7.1` made several criticisms that the log now supports with actual training evidence.

First, the audit warned that checkpointing by validation accuracy would bias the selected model toward image-level detection rather than localization. The log confirms exactly that. Models are repeatedly promoted on the basis of accuracy gains while Dice moves sideways or deteriorates.

Second, the audit warned that localization evaluation might be misleading or inflated, especially if authentic images with empty masks are included in the aggregate segmentation metrics. The log does not prove the exact implementation bug by itself, but it absolutely supports the suspicion. A Dice curve that freezes at `0.5949` for long stretches, paired with classification improvements and later instability, is not something a reviewer should trust.

Third, the audit criticized the duplicated experiment structure and weak execution hygiene. The training log confirms that this was not merely a cosmetic issue. The file contains multiple run segments, duplicated baseline-style content, offline-tracking fallbacks, and a later run that stops without a final summary. Provenance is messy enough that a reviewer would have to reconstruct the experiment by hand.

Fourth, the audit argued that the project looked stronger at image-level detection than at localization. The log now provides the empirical version of that statement: accuracy can climb into the high `0.8x` range while localization stays roughly in the low-to-mid `0.5x` band and sometimes much worse.
