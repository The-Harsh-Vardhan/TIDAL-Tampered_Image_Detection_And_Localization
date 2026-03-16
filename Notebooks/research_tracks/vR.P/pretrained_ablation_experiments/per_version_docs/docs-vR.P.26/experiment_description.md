# vR.P.26 -- Experiment Description

## Dual-Task Segmentation + Classification Head

### Hypothesis

A shared encoder with two task heads (pixel-level segmentation + image-level classification) provides mutual regularization: the classification head forces the encoder to learn globally discriminative features, while the segmentation head forces it to learn spatially precise features. The combined signal improves both tasks.

### Motivation

The project has two parallel tracks: ETASR classification (best: 90.23% accuracy) and pretrained localization (best: 87.32% image accuracy). A dual-task model unifies both in a single architecture. The classification head uses Global Average Pooling on the bottleneck features + a linear classifier, adding minimal parameters (~1K).

This is inspired by MTI-Net and other multi-task forensic architectures that combine detection + localization.

### Single Variable Changed from vR.P.3

**Architecture** -- Add a classification head to UNet's bottleneck. Training loss combines segmentation + classification.

### Key Configuration

| Parameter | P.3 (parent) | P.26 (this) |
|-----------|-------------|-------------|
| Architecture | UNet (segmentation only) | UNet + Classification head (dual-task) |
| Classification head | None | GAP -> FC(512, 256) -> FC(256, 1) -> Sigmoid |
| Loss | BCE + Dice (pixel) | BCE + Dice (pixel) + 0.5 * BCE (image-level) |
| Extra params | 0 | ~131K (negligible vs 3.17M) |
| Everything else | Same | Same |

### Pipeline

```
ELA input -> ResNet-34 encoder
    |
    +--> UNet decoder -> 384x384 segmentation mask
    |
    +--> GAP(bottleneck) -> FC(512,256,1) -> image-level tampered/authentic
    |
    v
Loss = seg_loss + 0.5 * cls_loss
```

### Expected Impact

+1-2pp Pixel F1, +2-3pp Image Accuracy. Classification head provides additional gradient signal to encoder, potentially improving feature quality for both tasks.

### Risk

Classification loss may dominate segmentation loss if not properly weighted. Image-level labels are derived from mask presence (tampered = any mask pixels > 0), which is binary and may not provide strong gradients.
