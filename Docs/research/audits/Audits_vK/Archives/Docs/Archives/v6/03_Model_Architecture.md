# Model Architecture

---

## Baseline Model

The architecture is a **baseline aligned with assignment constraints** — a pretrained U-Net suitable for Kaggle's T4 GPU. Research-frontier models (edge-enhanced, multi-trace, transformer hybrids) are documented in `11_Research_Alignment.md` as future work.

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,        # RGB only
    classes=1,
    activation=None,      # Raw logits
)
```

| Property | Value |
|---|---|
| Encoder | ResNet34, ImageNet pretrained |
| Decoder | U-Net (SMP default) |
| Input | `(B, 3, 384, 384)` — RGB |
| Output | `(B, 1, 384, 384)` — raw logits |
| Activation | None during training; sigmoid during inference |
| Parameters | ~24M |

### Why This Architecture

- **U-Net** is a proven architecture for dense prediction tasks including forgery localization. It provides multi-scale feature fusion through skip connections, which is critical for detecting tampering at various scales.
- **ResNet34** pretrained on ImageNet provides strong low/mid-level feature representations that transfer well to forensic tasks, compensating for the small CASIA dataset size.
- **SMP library** provides a clean, well-tested implementation with easy encoder swapping for future experiments.
- This architecture is consistent with the transfer-learning patterns described in survey papers and validated by the reference notebook `image-detection-with-mask.ipynb`.

---

## SMP API Reference

Access model components directly:

```python
model.encoder              # Pretrained ResNet34 backbone
model.decoder              # U-Net decoder
model.segmentation_head    # Final 1×1 conv
```

---

## Image-Level Detection

The tamper score is derived from the pixel-level probability map using a top-k mean:

```python
prob_map = torch.sigmoid(logits)           # (B, 1, H, W)
flat = prob_map.view(prob_map.size(0), -1)
k = max(int(flat.size(1) * 0.001), 32)
tamper_score = flat.topk(k=min(k, flat.size(1)), dim=1).values.mean(dim=1)
is_tampered = tamper_score >= pixel_threshold
```

The **same threshold** selected on the validation set is used for both pixel-level binarization and image-level detection. This keeps the system simple and avoids a second sweep.

**Known limitation:** Top-k mean is a handcrafted heuristic rather than a learned classifier. A dual-task classification head (as in the reference notebook `image-detection-with-mask.ipynb`) would be stronger but adds complexity beyond the assignment scope.

---

## Encoder Alternatives (Future Work)

| Encoder | Parameters | Notes |
|---|---|---|
| ResNet34 | ~24M | **Current baseline** |
| EfficientNet-B0 | ~5M | Lighter; worth testing if overfitting observed |
| EfficientNet-B1 | ~8M | Middle ground |

Encoder comparison is future/bonus work only.
