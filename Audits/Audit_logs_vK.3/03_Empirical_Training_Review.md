# 3. Empirical Training Review

Review sources:

- `Notebooks/vK.3_training_logs.txt`
- `Audits/Audit v7.1/Audit-vK.7.1.md`

## First Run

The first run, near lines `137-208`, rises from roughly `0.6268` validation accuracy at epoch 1 to a best validation accuracy of about `0.7077` by epoch 29, with a matching final test accuracy of `0.7077`.

That sounds acceptable until the segmentation side is examined. `Val Dice` moves in the first three epochs (`0.5212`, `0.4714`, `0.5398`), then snaps to `0.5949` at epoch 4 and stays there essentially unchanged for the rest of training. The final reported test Dice is also `0.5949`.

That is not a believable learning curve. A metric can plateau, but not in this mechanically flat, four-decimal, epoch-after-epoch way while the classification branch and losses are still moving. Either the metric computation is defective, the evaluation set composition is inflating the score, the predictions have collapsed into a degenerate regime that the metric fails to penalize properly, or multiple problems are happening at once.

Worse, because the run appears to use `Train: 1893 Val: 1892 Test: 1892`, and because the duplicated legacy block is known to miswire CSVs, this run is not suitable as a benchmark. The final "test" result is not something I would accept as an independent performance number in review.

## Second Run

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
