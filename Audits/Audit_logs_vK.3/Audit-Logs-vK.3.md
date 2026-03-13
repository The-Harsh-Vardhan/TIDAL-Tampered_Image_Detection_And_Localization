# Audit-Logs-vK.3

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

This document is a training-log postmortem, not a notebook rerun. It focuses on what the logs actually prove, what they imply about model behavior and metric validity, and which earlier audit concerns are now confirmed by evidence.

## 1. Executive Diagnosis

The headline is simple: classification improves, but the localization evidence is not trustworthy enough to support strong claims. The log shows a model that can learn image-level tamper discrimination reasonably well, but the mask metrics are either broken, inflated, miswired, or being reported through an evaluation setup that hides the real localization quality.

The first run is not a credible benchmark. It appears to use the wrong split wiring, it reports a validation Dice that freezes at `0.5949` for almost the entire run, and its final "test" result is too entangled with the validation setup to be treated as an independent holdout measure. The second run looks more serious on the classification side, climbing to `0.8864` validation accuracy by epoch 30, but the segmentation metrics remain erratic, underwhelming, and incomplete because the run is truncated before a final test summary. In plain language: the logs support "the detector learns something," not "the localization pipeline is solid."

## 2. Log Forensics

The log file has `529` lines and contains multiple experiment fragments rather than one clean, linear training story.

Near lines `24-108`, the data preparation section reports:

- dataset size: `12614`
- class counts: `7491` authentic, `5123` tampered
- split sizes: `8829 / 1892 / 1893`
- no missing masks in the discovered metadata preview

That looks reasonable at face value, but the first actual training run near lines `137-208` does not use those full split sizes. Instead, it reports `Train: 1893 Val: 1892 Test: 1892`, which is already a red flag because the train split has collapsed to the size of the nominal test split.

Cross-checking the project's later notebook structure explains why this matters. The earlier source-preserved experiment path is duplicated, and its old CSV wiring points `TRAIN_CSV` to `test_metadata.csv` and `TEST_CSV` to `val_metadata.csv`. The prior audit warned that the duplicated experiment block made it unclear which path was authoritative; the log now shows the practical consequence of that ambiguity. The first run is very likely not training on the intended full training split at all.

Near lines `209-211`, the log records that W&B fell back to offline mode because no API key was configured. That is not the main problem, but it is one more sign that experiment tracking was not tightly controlled.

Near lines `413-493`, the file shifts into a later run with richer metrics and the correct-looking full training size (`Train: 8829 Val: 1892 Test: 1893`). This is clearly the more serious run. It also emits `JpegCompression` deprecation warnings earlier in the file, which tells us the augmentation stack was not fully cleaned up.

The later run, however, is truncated. The file stops at epoch 32 without a `Training finished` line, without a final test summary, and without a closure artifact that would let a reviewer treat it as a complete experiment.

## 3. Empirical Training Review

### First Run

The first run, near lines `137-208`, rises from roughly `0.6268` validation accuracy at epoch 1 to a best validation accuracy of about `0.7077` by epoch 29, with a matching final test accuracy of `0.7077`.

That sounds acceptable until the segmentation side is examined. `Val Dice` moves in the first three epochs (`0.5212`, `0.4714`, `0.5398`), then snaps to `0.5949` at epoch 4 and stays there essentially unchanged for the rest of training. The final reported test Dice is also `0.5949`.

That is not a believable learning curve. A metric can plateau, but not in this mechanically flat, four-decimal, epoch-after-epoch way while the classification branch and losses are still moving. Either the metric computation is defective, the evaluation set composition is inflating the score, the predictions have collapsed into a degenerate regime that the metric fails to penalize properly, or multiple problems are happening at once.

Worse, because the run appears to use `Train: 1893 Val: 1892 Test: 1892`, and because the duplicated legacy block is known to miswire CSVs, this run is not suitable as a benchmark. The final "test" result is not something I would accept as an independent performance number in review.

### Second Run

The later run, near lines `413-493`, is the real signal in this file. It uses the expected full training size and reports richer metrics: validation accuracy, Dice, IoU, and F1.

The classification branch improves substantially:

- epoch 1 validation accuracy: `0.5201`
- epoch 20 validation accuracy: `0.7674`
- epoch 24 validation accuracy: `0.8705`
- epoch 30 validation accuracy: `0.8864`

So the detector is clearly learning better than in the earlier run.

The localization story is weaker and much less trustworthy:

- epochs 1-9 keep `Val Dice`, `Val IoU`, and `Val F1` pinned at `0.5949`
- epoch 10 still reports `Val Dice` and `Val IoU` both at `0.5881`
- later epochs become unstable, with Dice dropping as low as `0.3411` at epoch 17
- by epoch 30, validation accuracy reaches `0.8864` while validation Dice is only `0.5286`

The pattern matters more than the absolute numbers. Detection keeps improving, but localization remains mediocre and noisy. That is exactly the kind of mismatch you get when the shared backbone is learning image-level cues more successfully than pixel-level structure.

There is also a checkpoint-selection problem visible directly in the logs. New "best model" saves are triggered by validation accuracy even when localization quality gets worse. Examples:

- epoch 16 saves a new best at `Val Acc: 0.6718` while Dice is `0.4798`
- epoch 23 saves a new best at `Val Acc: 0.8388` while Dice is `0.4913`, lower than epoch 22's `0.5379`
- epoch 30 saves a new best at `Val Acc: 0.8864` while Dice is `0.5286`, still below epoch 25's `0.5441`

That is not a multitask success criterion. It is a classifier-first checkpoint policy pretending to be a joint detection-localization training loop.

## 4. Audit Cross-Check

The prior `Audit-vK.7.1` made several criticisms that the log now supports with actual training evidence.

First, the audit warned that checkpointing by validation accuracy would bias the selected model toward image-level detection rather than localization. The log confirms exactly that. Models are repeatedly promoted on the basis of accuracy gains while Dice moves sideways or deteriorates.

Second, the audit warned that localization evaluation might be misleading or inflated, especially if authentic images with empty masks are included in the aggregate segmentation metrics. The log does not prove the exact implementation bug by itself, but it absolutely supports the suspicion. A Dice curve that freezes at `0.5949` for long stretches, paired with classification improvements and later instability, is not something a reviewer should trust.

Third, the audit criticized the duplicated experiment structure and weak execution hygiene. The training log confirms that this was not merely a cosmetic issue. The file contains multiple run segments, duplicated baseline-style content, offline-tracking fallbacks, and a later run that stops without a final summary. Provenance is messy enough that a reviewer would have to reconstruct the experiment by hand.

Fourth, the audit argued that the project looked stronger at image-level detection than at localization. The log now provides the empirical version of that statement: accuracy can climb into the high `0.8x` range while localization stays roughly in the low-to-mid `0.5x` band and sometimes much worse.

## 5. Root-Cause Review

Several likely causes are supported by the evidence.

The first is evaluation inflation or distortion from empty-mask samples. If authentic images with all-zero masks are included in the segmentation metrics, a model that predicts low-activation or empty masks too often can still look better than it deserves. The prior audit flagged this, and the training curves are consistent with that problem.

The second is metric computation weakness. The early second-run behavior where Dice, IoU, and F1 all sit at `0.5949` is suspicious. Dice and F1 matching is not inherently wrong in binary segmentation, but Dice and IoU tracking identically at the same value over many epochs is a serious warning sign. At minimum, the metric path needs validation on a controlled batch.

The third is task imbalance. The classifier head appears to be winning the representation battle inside the shared backbone. The model is learning to answer "is this image tampered?" more effectively than "where exactly is the tampered region?"

The fourth is checkpoint policy. Even if the segmentation branch had moments of relative strength, the training loop is not selecting for them. It is explicitly rewarding image-level accuracy instead.

The fifth is run management. The first run appears polluted by legacy split wiring, and the second run is incomplete. That means the project is not just suffering from a modeling problem. It is suffering from experiment-control problems.

## 6. Brutal Roast

Here is the blunt version.

This log is what happens when a project starts acting impressed with its classifier before it has earned trust on localization. The accuracy curve is climbing, everyone wants to celebrate, and the mask metrics are quietly screaming that something is off.

The flat `0.5949` segmentation story is not "stable." It is suspicious. It looks like a broken thermometer being presented as climate science. If I saw this in an internship submission, my first assumption would not be "good localization." My first assumption would be "your metric pipeline is lying to you."

The first run is worse than weak. It is structurally compromised. If the model is training on `1893` samples when the prepared training split was `8829`, and the legacy block is miswiring train/test CSVs, then that run is not a result. It is a cautionary tale about why duplicated pipelines are dangerous.

Then the second run shows up, finally uses the right training size, pushes validation accuracy much higher, and still fails to make the localization story convincing. And even that run does not finish cleanly. No final test summary. No clean closeout. Just a training trace that stops midstream and leaves the reviewer to guess what happened next.

That is not research discipline. That is experiment sprawl.

## 7. Priority Fixes

1. Validate the segmentation metrics on tampered-only samples and on a small hand-checked batch before trusting any reported Dice or IoU.
2. Stop selecting checkpoints by validation accuracy alone. Use a localization-aware criterion, or at minimum a multitask score with a real localization component.
3. Separate detection and localization reporting cleanly so the classifier cannot hide a weak mask predictor.
4. Remove or quarantine the legacy duplicate pipeline so only one authoritative training path remains.
5. Rerun the improved experiment to completion and save final test metrics from a genuinely independent test split.
6. Clean up experiment tracking, dependency warnings, and version discipline so the next log reads like one experiment instead of three half-overlapping stories.

## 8. Final Verdict

Training quality is mixed. There is evidence that the classifier branch learns useful signal, especially in the later run. There is not enough trustworthy evidence to say the localization branch is performing well.

Metric trustworthiness is low. The first run is effectively invalid as a benchmark, and the second run still contains segmentation behavior that is too suspicious to accept without metric verification.

Experiment hygiene is weak. The duplicated legacy path, split confusion, offline-tracking fallback, deprecated augmentation warning, and truncated later run all point to a workflow that was not under tight control.

Readiness for submission claims is poor. These logs do not justify strong claims about pixel-level tamper localization. At best, they support a partially promising detector with an unproven, and possibly mismeasured, localization pipeline.
