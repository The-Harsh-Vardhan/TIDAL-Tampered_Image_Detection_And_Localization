# 04 — Model Architecture

## Purpose

Specify the segmentation model architecture and encoder selection.

## Baseline Architecture: U-Net

The model is a standard U-Net with a pretrained encoder, implemented via `segmentation_models_pytorch` (SMP).

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None,
)
```

### Why U-Net

- Proven architecture for binary segmentation with skip connections preserving spatial detail.
- SMP provides a clean API with pretrained encoders.
- Fits comfortably on a T4 GPU.

### Baseline Encoder: ResNet34

ResNet34 is the locked baseline encoder for the MVP. Other encoders are optional experiments for Phase 3.

| Encoder | Notes |
|---|---|
| **ResNet34** | **Baseline.** Reliable, fast training, well-supported in SMP. |
| EfficientNet-B0 | Optional Phase 3 comparison. Lighter, may offer good accuracy-to-size ratio. |
| EfficientNet-B1 | Optional Phase 3 comparison. Slightly larger. |

Approximate parameter counts and VRAM usage should be measured in the actual Colab notebook, not assumed from documentation.

### SMP Model API

When accessing model subcomponents (e.g., for differential learning rates), use the SMP `Unet` attributes directly:

```python
# Correct — these are direct attributes of smp.Unet
model.encoder       # Pretrained encoder
model.decoder       # Decoder with skip connections
model.segmentation_head  # Final 1x1 conv
```

Do **not** use `model.unet.encoder` or similar nested paths — these do not exist on `smp.Unet`.

### Output Format

- The model outputs **raw logits** with shape `(batch, 1, 512, 512)`.
- During training: sigmoid is applied inside `BCEWithLogitsLoss`.
- During inference: apply sigmoid explicitly to get probabilities in [0, 1].
- A threshold converts the probability map to a binary mask.

### Image-Level Detection

Image-level tamper detection is derived from the predicted probability map. No separate classification head is needed.

```python
# Use probability map (before thresholding), not binary mask
prob_map = torch.sigmoid(logits)
tamper_score = prob_map.max().item()
is_tampered = tamper_score >= threshold
```

**Note:** `max()` is sensitive to isolated false positives. A more stable alternative is mean of top-k probabilities or mask area fraction. Choose one approach and document it.

The threshold is selected on the validation set only. Do not tune on the test set.

## Optional: SRM Preprocessing (Phase 3 Only)

SRM (Spatial Rich Model) filters extract noise residuals that may reveal forensic artifacts invisible in RGB. This is a **bonus ablation**, not part of the MVP.

If implemented, SRM output is concatenated with RGB before the encoder, changing `in_channels` from 3 to 6. The SRM implementation should be treated as experimental — the 3-kernel placeholder repeated to 30 channels from earlier docs is not a serious forensic enhancement.

## Excluded Architectures

| Architecture | Why excluded |
|---|---|
| SegFormer | Transformer-based; unnecessary complexity for this dataset |
| Dual-stream networks | Research-grade complexity for a notebook assignment |
| BayarConv / Noiseprint++ | Specialized forensic modules with complex implementations |
| Separate classification head | Image detection is derived from the mask |

## Related Documents

- [03_Data_Pipeline.md](03_Data_Pipeline.md) — Input data format
- [05_Training_Strategy.md](05_Training_Strategy.md) — Training loop and loss
