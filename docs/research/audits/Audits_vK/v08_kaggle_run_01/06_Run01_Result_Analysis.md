# Run-01 Result Analysis

## Training dynamics

Cell 29 shows a real learning signal:

- train loss falls from `2.2359` to `1.4455`
- validation F1 improves from `0.0826` to a best of `0.3585`
- best checkpoint is at epoch `17`

So the model is not dead. It learned something.

The problem is what it learned and how unstable that learning was.

## Validation behavior is noisy and not cleanly converged

Validation loss:

- starts at `2.1599`
- reaches a low of `1.9313` around epoch 8
- then mostly stays worse than that while train loss keeps dropping

Validation F1:

- jumps around instead of climbing smoothly
- spikes at epoch 10 (`0.3335`)
- falls back
- peaks at epoch 17 (`0.3585`)
- never establishes stable dominance afterward

That pattern looks like partial learning plus unstable calibration, not robust convergence.

## Overfitting signals are obvious

The model keeps improving train loss while validation loss drifts upward. That is textbook overfitting behavior, even if validation F1 occasionally spikes upward due to thresholded metric noise.

The scheduler helps slow the damage, but it does not fix the underlying issue. Early stopping at epoch 27 is reasonable operationally, but the run still looks like it is fitting easy training structure faster than it generalizes.

## The threshold result is a red flag

Cell 32 selects:

- best threshold: `0.7500`
- best validation F1 at that threshold: `0.5198`

The notebook itself prints:

`pos_weight may be too aggressive. Consider reducing.`

That is the run confessing calibration trouble. If probabilities require a threshold that high to behave, your outputs are not well calibrated. Stop pretending this is a nice clean tuning result.

## Test results: the model learned the easy half of the task

Cell 33 tells the real story.

Tampered-only localization:

- Pixel-F1 `0.2949`
- Pixel-IoU `0.2321`

Forgery-type breakdown:

- splicing F1 `0.5758`
- copy-move F1 `0.1394`

Mask-size breakdown:

- tiny under 2 percent: F1 `0.1432`
- small 2 to 5 percent: F1 `0.2429`
- medium 5 to 15 percent: F1 `0.4057`
- large over 15 percent: F1 `0.5573`

This is not a generally strong tamper localizer. This is a model that finds larger and easier splicing regions and falls apart on harder copy-move and small-region cases.

## Mixed-set metrics flatter the model

Mixed-set results:

- Pixel-F1 `0.5181`
- Pixel-IoU `0.4926`

Those numbers are much better than tampered-only results because authentic empty-mask samples are easy wins when the model predicts nothing. That is why mixed-set averages should never be used to sell this run as healthy.

## Image-level results are merely okay, and methodologically weak

Cell 33 reports:

- image accuracy `0.7190`
- image AUC `0.8170`

Those numbers are not embarrassing for a heuristic. They are also not strong evidence of a properly solved detection problem because the detector is not trained.

This is the equivalent of saying, "the segmentation map happened to correlate with class labels often enough." That is not the same thing as solving detection.

## Failure analysis confirms the exact weak spots you would expect

Cell 44 shows:

- worst 10 predictions mean F1 `0.0000`
- 9 of 10 are copy-move
- 7 of 10 have mask area under 2 percent

So the notebook's own analysis agrees with the metric breakdown. The model is weak exactly where a serious tamper detector needs to be strong.

## Did the model learn meaningful features?

Yes, but narrowly.

It learned enough to:

- localize many splicing cases
- produce nontrivial masks
- outperform trivial behavior on some manipulations

It did not learn enough to:

- localize hard or subtle manipulations reliably
- generalize evenly across forgery types
- serve as a trustworthy image-level detector

This run is evidence of partial task understanding by the model, not evidence of a submission-ready system.
