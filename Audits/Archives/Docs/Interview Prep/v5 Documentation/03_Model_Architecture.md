# Model Architecture

## How to explain this in an interview

Start with this:

"The core model is a U-Net segmentation network with a ResNet34 encoder. I chose it because the project needs pixel-level localization, and this architecture gives a good balance of accuracy, training stability, and compute efficiency on a single GPU."

## What the model is

The project uses:

- `smp.Unet`
- `encoder_name="resnet34"`
- `in_channels=3`
- `classes=1`

That means the model takes an RGB image and outputs a single-channel tamper logit map. After a sigmoid, that becomes a tamper probability map.

## What problem this architecture solves

The project needs localization, not just classification.

A segmentation model solves that by assigning a probability to every pixel:

- high probability means the pixel is likely manipulated
- low probability means the pixel is likely authentic

This is exactly the right setup when the target output is a tamper mask.

## Why segmentation is needed

If I used a standard image classifier, I would only get:

- tampered
- authentic

That is not enough for this project. The user also wants:

- pixel-level evidence
- overlays for visualization
- interpretable outputs for debugging

Segmentation gives all of that directly.

## What U-Net does

U-Net has two main parts:

- an encoder that compresses the image into deep feature representations
- a decoder that upsamples those features back to full resolution

Its key idea is the skip connection. Features from early encoder layers are passed to the decoder so that the model keeps fine spatial detail while still using deeper semantic information.

That is important here because tampered regions can be:

- small
- thin
- boundary-sensitive
- hard to localize precisely

## Why U-Net was chosen for this project

### What it is

U-Net is a widely used segmentation architecture that works well when you need detailed masks.

### What problem it solves

It solves the problem of combining:

- high-level context
- low-level spatial precision

### Why it was chosen here

I chose U-Net because it is:

- well understood
- stable to train
- easy to implement with `segmentation_models_pytorch`
- strong enough for a baseline
- practical on a Colab T4 GPU

For this project, reliability mattered more than chasing the newest architecture.

## Why ResNet34 was chosen as the backbone

### What it is

ResNet34 is a convolutional encoder pretrained on ImageNet.

### What problem it solves

The dataset is not huge, so training a deep encoder from scratch would be harder and more data hungry. A pretrained encoder gives the model strong low-level and mid-level visual features from the start.

### Why it was chosen here

ResNet34 was a good middle ground:

- lighter than deeper ResNets
- stronger and more expressive than very small backbones
- mature and stable
- easy to fine-tune

It also fits the project goal of building a robust baseline without overcomplicating the training setup.

## How image-level detection is derived

The project does not use a separate classification head in the MVP. Instead, it derives an image-level tamper score from the segmentation probability map using a top-k mean.

Why this was chosen:

- it keeps the architecture simple
- it reuses the segmentation output directly
- it avoids building a second head before the localization baseline is validated

Tradeoff:

- it is simpler, but it is still a heuristic
- a dedicated classifier head would likely be stronger in a future version

## Alternatives that could have been used

### DeepLabV3

What it is:
DeepLabV3 is a stronger semantic-segmentation family that uses atrous convolutions and multi-scale context.

What problem it solves:
It often performs well when larger receptive fields and multi-scale context matter.

Why it was not selected:

- it is heavier than the chosen U-Net baseline
- it adds more complexity than needed for an MVP
- U-Net is easier to explain and faster to iterate on in a notebook setting

### Vision Transformers

What they are:
Transformer-based vision models use self-attention to model long-range relationships.

What problem they solve:
They can capture global context better than standard CNNs and can be very strong on challenging segmentation tasks.

Why they were not selected:

- they are usually more compute intensive
- they often need more data or stronger pretraining
- they are harder to justify as the first baseline on a small forensic dataset

For this project, that tradeoff was not worth it.

### EfficientNet-based encoder

What it is:
EfficientNet is a family of CNN backbones optimized for strong accuracy-to-parameter efficiency.

What problem it solves:
It can provide a lighter encoder or better parameter efficiency than a standard ResNet.

Why it was not selected:

- ResNet34 is more established and simpler to reason about in this baseline
- the improvement was not guaranteed to justify changing the main reference architecture
- the project benefited more from stability and clarity than from marginal encoder experimentation

## Tradeoff summary

| Option | Strength | Weakness | Why it was not the MVP choice |
|---|---|---|---|
| U-Net + ResNet34 | Stable, simple, efficient, good localization baseline | Not the most advanced architecture | Best balance for the project |
| DeepLabV3 | Strong multi-scale segmentation | Heavier and more complex | More than needed for first baseline |
| Vision Transformer models | Better global context potential | Higher compute and data demand | Too expensive for Colab-first baseline |
| EfficientNet encoder | Good parameter efficiency | Less standard for this setup | Lower priority than baseline clarity |

## Future improvements

The most natural architecture upgrades would be:

- a dual-head model with a learned classification branch
- a stronger encoder such as EfficientNet or ConvNeXt
- multi-scale segmentation heads
- edge-aware supervision
- transformer-based hybrids
- extra forensic channels such as ELA or SRM

## How I would summarize this architecture

"I used U-Net because the real target is localization, and U-Net is strong at producing detailed masks. I used ResNet34 because it gave me a pretrained encoder that was light enough for Colab and stable enough for a small dataset. I deliberately treated it as a strong baseline rather than trying to start with a more expensive transformer model."
