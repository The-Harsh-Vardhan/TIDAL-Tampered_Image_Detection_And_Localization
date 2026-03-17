# 04 — Model Architecture

## Purpose

This document specifies the segmentation model architecture, encoder selection, and optional SRM enhancement.

## Baseline Architecture: U-Net

The model is a standard U-Net with a pretrained encoder, implemented via the `segmentation_models_pytorch` (SMP) library.

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",         # or "efficientnet-b0" / "efficientnet-b1"
    encoder_weights="imagenet",
    in_channels=3,                   # RGB input for MVP
    classes=1,                       # Single-channel binary mask output
    activation=None,                 # Raw logits; sigmoid applied externally
)
```

### Why U-Net

- Proven architecture for binary segmentation with skip connections that preserve spatial detail.
- SMP provides a clean API with pretrained encoders.
- Fits comfortably on a T4 GPU.

### Encoder Options

| Encoder | Parameters | T4 VRAM (batch=4, 512×512) | Notes |
|---|---|---|---|
| ResNet34 | ~21.8M | ~4.5 GB with AMP | Reliable default, fast training |
| EfficientNet-B0 | ~5.3M | ~4.0 GB with AMP | Lighter, good accuracy-to-size ratio |
| EfficientNet-B1 | ~7.8M | ~5.0 GB with AMP | Slightly larger, potentially better features |

**Recommended starting encoder:** ResNet34 or EfficientNet-B0. The choice can be compared as a Stage 3 bonus experiment.

### Output Format

- The model outputs **raw logits** with shape `(batch, 1, 512, 512)`.
- During training, sigmoid is applied inside `BCEWithLogitsLoss`.
- During inference, sigmoid is applied explicitly to produce a probability map in [0, 1].
- A threshold (default 0.5) converts the probability map to a binary mask.

### Image-Level Detection

Image-level tamper detection is derived from the predicted mask — no separate classification head is needed.

```python
tamper_score = predicted_mask.max()  # Max pixel probability
is_tampered = tamper_score >= threshold
```

The threshold is chosen on the validation set. Do not tune it on the test set.

## Optional Enhancement: SRM Preprocessing (Stage 3)

SRM (Spatial Rich Model) filters extract noise residuals from images, which can reveal forensic artifacts invisible in RGB.

This is an **optional bonus enhancement**, not required for the MVP.

### SRM Module

```python
import torch
import torch.nn as nn
import numpy as np

class SRMFilterLayer(nn.Module):
    """Fixed (non-trainable) SRM high-pass filter bank."""
    
    def __init__(self):
        super().__init__()
        # 30 handcrafted 5x5 high-pass kernels
        srm_kernels = self._get_srm_kernels()  # Shape: (30, 3, 5, 5)
        self.srm = nn.Conv2d(3, 30, kernel_size=5, padding=2, bias=False)
        self.srm.weight = nn.Parameter(
            torch.from_numpy(srm_kernels).float(),
            requires_grad=False,
        )
        # Learnable channel reduction: 30 -> 3
        self.reduce = nn.Conv2d(30, 3, kernel_size=1)

    def forward(self, x):
        noise = self.srm(x)
        return self.reduce(noise)

    def _get_srm_kernels(self):
        # Standard SRM kernel set (simplified example for 3 base kernels)
        # Full implementation uses 30 kernels from the SRM literature
        q = np.array([
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, -2, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, -2, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0]],
        ], dtype=np.float32)
        # Expand across 3 input channels and tile to 30 kernels
        kernels = np.repeat(q, 10, axis=0)           # (30, 5, 5)
        kernels = kernels[:, np.newaxis, :, :]        # (30, 1, 5, 5)
        kernels = np.repeat(kernels, 3, axis=1)       # (30, 3, 5, 5)
        return kernels
```

### Model with SRM

When SRM is enabled, RGB and SRM features are concatenated before entering the U-Net:

```python
class ForensicUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", use_srm=False):
        super().__init__()
        self.use_srm = use_srm
        in_ch = 6 if use_srm else 3
        
        if use_srm:
            self.srm = SRMFilterLayer()
        
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        if self.use_srm:
            noise = self.srm(x)
            x = torch.cat([x, noise], dim=1)  # (B, 6, H, W)
        return self.unet(x)
```

SRM is treated as an ablation study. The recommended comparison is:

| Model | Input channels | Description |
|---|---|---|
| RGB-only (MVP) | 3 | Baseline |
| RGB + SRM (Bonus) | 6 | SRM noise residuals concatenated with RGB |

## Architecture Decisions Not Taken

| Architecture | Why excluded |
|---|---|
| SegFormer | Transformer-based; added complexity without proven benefit for this dataset size |
| Dual-stream networks | Research-grade complexity; not justified for a single-notebook assignment |
| BayarConv / Noiseprint++ | Specialized forensic modules with complex implementations |
| Separate classification head | Image-level detection is derived from the mask; no extra head needed |

These may be mentioned as future work but are not part of the implementation scope.

## Related Documents

- [03_Data_Pipeline.md](03_Data_Pipeline.md) — Input data format and augmentation
- [05_Training_Pipeline.md](05_Training_Pipeline.md) — Training loop and loss function
