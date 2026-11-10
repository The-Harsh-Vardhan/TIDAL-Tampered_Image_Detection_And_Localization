# 04 - Training Strategy Audit

## Bottom line

The training setup is competent enough to run and sloppy enough to mislead. The notebook contains several standard good practices, but the actual optimization logic still makes avoidable mistakes in class-balance accounting, threshold dependence, and objective selection.

## 1. BCE plus Dice is fine, but the justification is shallow

Cell 23 uses `BCEWithLogitsLoss` plus Dice. That is a reasonable binary segmentation baseline. No complaint there. The complaint is that the notebook never really justifies why this combination is the right one for the dataset properties, the target mask distribution, or the failure mode the author claims to be fixing.

There is too much "we added this because v8 says so" and not enough "here is the mechanism we expect and the failure signal that would prove it."

## 2. `pos_weight` is computed inconsistently

Cell 22 is one of the most important cells in the notebook, and it is still sloppy.

For tampered images, the code reads the raw mask file and counts foreground and background in the original mask resolution. For authentic images, it does not read anything. It simply adds `CONFIG['image_size'] ** 2` background pixels per sample.

That means the accounting mixes:

- native-resolution tampered masks, and
- resized synthetic authentic masks.

Those are not the same unit. If source image sizes vary, the ratio is distorted. The notebook is pretending to compute a principled class-balance statistic while quietly mixing inconsistent pixel spaces.

## 3. The scheduler and early stopping monitor the wrong thing

Cell 29 runs validation with `validate_model(...)`, and Cell 27 defaults that validation threshold to `0.5`. Later, Cell 32 sweeps thresholds and admits the useful operating point may be elsewhere.

So what are scheduler and early stopping actually tracking?

- thresholded Pixel-F1 at 0.5,
- before calibration is known,
- on a mixed validation set that includes authentic images.

That is a bad control signal. The optimizer is being steered by a brittle threshold choice, not by a threshold-free metric or even by tampered-only behavior.

## 4. The augmentation expansion is still ordinary

Cell 15 adds:

- flips,
- 90-degree rotations,
- color jitter,
- JPEG compression,
- Gaussian noise,
- Gaussian blur.

That is better than the old minimal setup. It is still generic vision augmentation, not task-shaped forensic augmentation. There is no attempt to preserve aspect ratio, no crop strategy, no copy-move-focused stressor, no content-aware hard negative generation, and no analysis of whether augmentation corrupts the forensic signal more than it regularizes the model.

This is the kind of augmentation block people write when they know they should "add more augmentation" but have not yet thought deeply about the task.

## 5. The notebook hand-waves calibration

Cell 32 prints a warning if the best threshold falls outside an "expected range" of `0.20-0.55`. That is not calibration analysis. That is superstition with formatted output. A threshold range is not a proof of healthy probabilities.

If calibration matters, then measure calibration. Plot reliability, inspect score distributions, separate pixel and image-level operating points, or stop talking like threshold sanity messages are science.

## 6. There are some real strengths

The training loop in Cells 27 through 29 has several solid engineering decisions:

- AMP support,
- gradient accumulation,
- gradient clipping,
- checkpoint resume,
- differential encoder and decoder learning rates,
- logging of learning rates and gradient norms.

Those are legitimate strengths. The author is not clueless. They are just not holding themselves to a high enough evidentiary standard.

## 7. Small but real training-quality issues

- `drop_last=True` in the train loader discards samples every epoch for no clearly justified reason.
- `encoder_warmup_epochs` exists in config but is set to zero, which makes the warmup feature more decoration than strategy.
- `train_ratio` exists in config and then gets ignored because Cell 12 hard-codes `0.30` for the temp split.
- `use_multi_gpu=True` is enabled by default even though the assignment is centered on Colab-class hardware where this path will never trigger.

These are not fatal. They are quality signals. Right now they signal "good enough to demo, not tight enough to trust."

## Verdict

The training strategy is not nonsense, but it is not as rigorous as the notebook tries to sound. The biggest problems are:

1. inconsistent `pos_weight` accounting,
2. scheduler and early stopping driven by a bad thresholded metric,
3. generic augmentation rather than task-shaped augmentation,
4. no real calibration analysis.

This is baseline-quality training logic. It is not principal-level work yet.
