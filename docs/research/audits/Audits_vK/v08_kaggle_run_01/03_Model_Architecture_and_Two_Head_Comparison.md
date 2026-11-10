# Model Architecture And Two-Head Comparison

## What the notebook actually uses

Cells 19 and 20 use `segmentation_models_pytorch.Unet` with:

- encoder: `resnet34`
- pretrained weights: `imagenet`
- input channels: `3`
- output classes: `1`

The output is a single segmentation logit map. There is no learned classification head.

## Why U-Net is a reasonable baseline

U-Net is still a sane default for localization because skip connections help preserve spatial detail, and the decoder structure is easy to train on modest GPU budgets. If the assignment only asked for pixel masks, U-Net would be a perfectly boring and acceptable first answer.

The problem is not that U-Net is wrong. The problem is that the notebook treats "U-Net is standard" as if that finishes the discussion.

## Why ResNet34 is a reasonable baseline

ResNet34 is cheap enough, pretrained, and widely supported in SMP. It is a practical encoder when you care about cloud runtime and do not want a giant backbone.

Again, that is baseline logic, not task-specific reasoning.

## The notebook's architectural reasoning is thin

Cell 19 basically says:

- the architecture is retained from v6.5
- Docs8 concluded architecture is not the main bottleneck
- SMP makes swaps easy

That is not a serious architecture argument. It avoids the actual questions:

- Why is global context not explicitly modeled for image-level detection?
- Why is a pure segmentation output enough when the assignment also requires detection?
- Why is a mid-size CNN encoder preferred over DeepLabV3+, FPN, or even a smaller model if runtime is a concern?
- Why should anyone believe architecture is not the bottleneck when copy-move F1 is `0.1394`?

The notebook did not answer those questions. It inherited a baseline and spent the rest of its energy on training tweaks.

## Why not DeepLab or transformer-based models?

The notebook never benchmarks them, so any strong claim would be fake. The sensible position is:

- DeepLabV3+ would have been a credible comparison baseline because it handles multi-scale context better than plain U-Net and is still practical in cloud notebooks.
- Transformer-heavy models are probably unnecessary for this assignment unless the baseline is already solid and runtime headroom is proven.

So the mistake is not "you failed to use a transformer." The mistake is "you did not justify why your chosen baseline was enough."

## The bigger architectural flaw: no learned detection head

The assignment explicitly requires detection and localization. The notebook only learns localization. Labels are available in the dataset object (cell 16), but training ignores them.

That means the architecture is incomplete with respect to the assignment, not because it cannot localize, but because it refuses to model detection directly.

## Reference notebook comparison

`Pre-exisiting Notebooks/image-detection-with-mask.ipynb` uses a dual-head design in cell 3:

- `seg_logits` for localization
- `cls_logits` for image classification

It trains both heads with:

- `criterion_cls = nn.CrossEntropyLoss(...)`
- `criterion_seg = nn.BCEWithLogitsLoss()`
- total loss = `ALPHA * cls_loss + BETA * seg_loss`

That reference notebook is not production-grade. It is a scratch-built U-Net, model selection is driven by validation accuracy, and the qualitative views still do not show ground-truth masks. But the core architectural idea is better aligned with the assignment than the current Run-01 notebook.

## Should a two-head architecture replace the current heuristic detection?

Yes.

### Advantages

1. It turns detection into a learned objective instead of a post-hoc trick.
2. It lets image-level and pixel-level decisions have separate thresholds and calibration behavior.
3. It gives the encoder pressure to learn global manipulation cues, not just local blobs.
4. It uses labels the pipeline already carries around anyway.

### Disadvantages

1. You need to balance classification and segmentation losses.
2. Poor loss weighting can let classification dominate and hurt mask quality.
3. It adds one more axis to tune and debug.

None of those disadvantages are serious enough to justify the current shortcut. The current notebook already has the supervision needed. It is just not using it.

## Recommendation

Keep the practical part of the current design:

- SMP backbone
- pretrained encoder
- segmentation head

Then add a classification head on the encoder bottleneck using global average pooling plus a small MLP. Train:

- segmentation loss on masks
- classification loss on image labels

That is the cleanest way to satisfy the assignment honestly.
