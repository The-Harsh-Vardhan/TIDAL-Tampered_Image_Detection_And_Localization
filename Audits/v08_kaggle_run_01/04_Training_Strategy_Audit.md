# Training Strategy Audit

## What is solid

The training setup is not amateur-hour chaos. There are several good decisions here:

- AdamW with separate encoder and decoder learning rates in cell 24
- AMP enabled in cells 24 and 27
- gradient accumulation and clipping in cell 27
- checkpointing and resume support in cells 28 and 29
- `ReduceLROnPlateau` in cell 24
- per-sample Dice in cell 23 to reduce large-mask dominance

That is real engineering effort. Now the parts that are still wrong.

## BCE plus Dice is acceptable, but the imbalance handling is shaky

The combined BCE plus Dice loss in cell 23 is a normal choice. The problem is how `pos_weight` is computed in cell 22.

The code counts foreground and background pixels from raw tampered masks, but for authentic images it adds:

`CONFIG['image_size'] ** 2`

That means authentic samples are counted as if they were already resized to `384x384`, while tampered masks use their raw mask sizes. Those are inconsistent units. It is a sloppy approximation disguised as a principled foreground prior.

Then the output from cell 22 lands at:

- `pos_weight: 30.01`
- `Foreground fraction: 3.2246%`

And cell 32 later picks threshold `0.7500` while warning that `pos_weight may be too aggressive`.

That sequence is not subtle. The notebook is telling you the weighting likely overshot.

## The optimization target is misaligned with the final operating point

This is one of the dumbest design mistakes in the notebook.

Validation in cell 27 computes F1 and IoU using a fixed threshold of `0.5`.

Then:

- scheduler steps in cell 29 based on that validation F1
- early stopping in cell 29 is driven by that validation F1
- best checkpoint in cell 29 is picked from that validation F1

After all that, cell 32 tunes the threshold and decides the real operating point is `0.75`.

So the notebook spends the whole training process optimizing one decision rule, then deploys another. That is incoherent.

Senior expectation: either validate with a threshold-free metric for model selection, or integrate threshold tuning into the validation protocol consistently.

## Augmentation strategy has the right instinct, but it is not enough

Cell 15 includes:

- horizontal and vertical flips
- 90 degree rotations
- `ColorJitter`
- `ImageCompression`
- `GaussNoise`
- `GaussianBlur`

That is better than a bare resize-plus-normalize pipeline. It directly addresses some forgery shortcuts and robustness concerns.

What is still weak:

- no aspect-ratio preservation
- no scale-aware crop policy for tiny masks
- no reasoned tie between augmentations and the copy-move failure mode
- no ablation showing which augmentations actually helped

The notebook chose a bag of common transforms and hoped the right ones mattered.

## Learning-rate scheduling helped training last longer, but not look smarter

The scheduler clearly extended training beyond a quick collapse. Cell 29 shows LR drops at epochs 14, 21, and 25, and best validation F1 is reached at epoch 17 instead of dying early.

That is useful.

It still does not hide the bigger truth:

- train loss keeps falling from `2.2359` to `1.4455`
- validation loss mostly trends upward after its best early values
- validation F1 is noisy rather than steadily improving

This is "helpful bandage," not "training strategy solved."

## Labels are present, but unused during training

The dataset returns `label` in cell 16. The training loop in cell 27 ignores it completely. That means the notebook spends compute propagating supervision it refuses to learn from.

For an assignment explicitly asking for detection and localization, that is an avoidable design failure.

## Training verdict

Competent implementation, questionable reasoning.

The author knows how to wire together modern training conveniences. The weak part is the logic:

- imbalance handling is only approximately correct
- threshold calibration is clearly off
- model selection is misaligned with final deployment threshold
- image-level supervision is left on the table

That is the difference between building a working pipeline and actually understanding what the pipeline should optimize.
