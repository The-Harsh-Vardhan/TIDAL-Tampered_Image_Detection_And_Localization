# Assignment Compliance And Run-01 Verdict

## Evidence base

Primary evidence is `notebooks/v8-tampered-image-detection-localization-kaggle-run-01.ipynb`. The assignment baseline is `Assignment.md`, especially:

- `Model Architecture & Learning` for region prediction and cloud viability
- `Testing & Evaluation` for localization plus image-level detection
- `Deliverables & Documentation` for clear visualizations inside one notebook

## Requirement-by-requirement audit

### 1. Tampered image detection and localization

Localization: yes, partially credible. The notebook trains a segmentation model, evaluates pixel metrics, and visualizes prediction grids and diagnostic overlays in Sections 9 and 10 (cells 35 to 44).

Detection: still half-baked. In cell 33, image-level detection is not learned. The score is just:

`tamper_score = probs[i].view(-1).max().item()`

That means the image label is derived from the single hottest pixel in the segmentation map. The assignment asked for detection and localization. The notebook only truly learns localization.

Verdict on this requirement: Partial.

### 2. A model that predicts tampered regions

Yes. `smp.Unet` in cell 20 outputs a one-channel mask, and cell 33 reports pixel metrics on predicted masks.

Verdict on this requirement: Pass.

### 3. Freedom to choose architecture and loss functions

Formally satisfied, but used lazily. The notebook chooses U-Net with ResNet34 and BCE plus Dice. That is allowed. The problem is not permission. The problem is the reasoning quality behind the choice.

Verdict on this requirement: Pass, but weakly justified.

### 4. Runnable on Google Colab or similar GPU environments

Kaggle viability is proven. Cells 3 to 7, 20, and 55 show a complete Kaggle run with dependencies, W&B setup, checkpointing, plots, and saved artifacts.

Colab viability is not proven. The executed run used:

- `DataParallel enabled across 2 GPUs` (cell 20 output)
- `batch_size=64` and `accumulation_steps=4` for effective batch size `256` (cell 5 output)
- `kaggle_secrets` and `/kaggle/working` assumptions (cells 7 and 55)

That is not the same thing as showing a clean one-GPU Colab path.

Verdict on this requirement: Partial.

### 5. Clear reasoning behind architecture choices

This is where the notebook ducks the hard thinking. Cell 19 says the architecture is retained from v6.5 because Docs8 decided architecture was not the main bottleneck. That is not reasoning. That is inheritance.

What is missing:

- why U-Net is specifically right for this forgery problem
- why ResNet34 is the right encoder under these data and runtime constraints
- why no learned image-level head exists despite the assignment requiring detection
- why no alternative baseline such as DeepLabV3+ or FPN was tested

Verdict on this requirement: Partial.

## What Run-01 proves versus what it only implies

Run-01 proves:

- the notebook runs end-to-end on Kaggle
- checkpoints and artifacts were saved
- the model learned enough to beat zero effort on splicing
- the author can instrument a training notebook beyond toy level

Run-01 does not prove:

- that the model is good at tamper localization in general
- that image-level detection is solved
- that the notebook is Colab-ready
- that the current architecture is well justified

## Unnecessary complexity check

The notebook is not absurdly overbuilt, but it is starting to drift. Grad-CAM, robustness charts, shortcut tests, and W&B artifact management are fine once the core model is strong. Right now they are accessories on top of a model that still posts:

- tampered-only Pixel-F1 `0.2949`
- copy-move F1 `0.1394`
- tiny-mask F1 `0.1432`

That is not the time to act like the core problem is already solved.

## Final verdict

Partial

This is a real training notebook, not a superficial demo, and that matters. But the assignment is not "train a segmentation model and extract a fake detector from its hottest pixel." The notebook only partially satisfies the deliverable because detection is still a heuristic, architectural reasoning is thin, and single-GPU Colab portability is still implied rather than demonstrated.
