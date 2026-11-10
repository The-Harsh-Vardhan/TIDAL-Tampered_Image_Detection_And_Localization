# Model Architecture

---

## Baseline Model (MVP)

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,        # RGB only for MVP
    classes=1,
    activation=None,      # Raw logits
)
```

| Property | Value |
|---|---|
| Encoder | ResNet34, ImageNet pretrained |
| Decoder | U-Net (SMP default) |
| Input | `(B, 3, 512, 512)` — RGB |
| Output | `(B, 1, 512, 512)` — raw logits |
| Activation | None during training; sigmoid during inference |
| Parameters | ~24M (measure in notebook) |

---

## SMP API Reference

Access model components directly — do **not** use `model.unet.*`:

```python
model.encoder              # Pretrained ResNet34 backbone
model.decoder              # U-Net decoder
model.segmentation_head    # Final 1×1 conv
```

---

## Image-Level Detection

**Locked decision for MVP:** Derive the tamper score from the pixel-level probability map. No separate classification head.

```python
prob_map = torch.sigmoid(logits)           # (B, 1, H, W)
B = prob_map.size(0)
tamper_score = prob_map.view(B, -1).max(dim=1).values  # Per-image scalar
is_tampered = tamper_score >= pixel_threshold
```

The **same threshold** selected on the validation set is used for both pixel-level binarization and image-level detection. This is intentional: a single operating point keeps the system simple and avoids a second sweep.

**Known limitation:** `max(prob_map)` is sensitive to single false-positive pixels. Alternatives (top-k mean, mask-area fraction) are documented in Limitations but not used in MVP.

---

## Optional: Error Level Analysis (Phase 2)

ELA highlights regions with inconsistent compression artifacts — a signal that may complement RGB for splicing detection. This technique exploits the fact that JPEG re-compression produces different error levels in tampered vs. authentic regions.

```python
def compute_ela(image_rgb, quality=90):
    """Compute Error Level Analysis map.

    1. Re-save image at a fixed JPEG quality.
    2. Compute absolute difference with original.
    3. Return the difference as a grayscale intensity map.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), encode_param)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    recompressed = cv2.cvtColor(recompressed, cv2.COLOR_BGR2RGB)
    ela = np.abs(image_rgb.astype(np.float32) - recompressed.astype(np.float32))
    ela_gray = np.mean(ela, axis=2)  # Average across channels
    return ela_gray
```

If ELA is enabled, the dataset class concatenates it as a **4th channel** (RGB + ELA grayscale):

```python
# Phase 2 only — changes in_channels from 3 to 4
model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                  in_channels=4, classes=1, activation=None)
```

**Note:** When `in_channels != 3`, ImageNet pretrained weights cannot be used directly (the first conv layer shape changes). Either start from scratch or manually adapt the first layer.

---

## Optional: SRM Features (Phase 3)

Spatial Rich Model kernels extract noise residuals. If implemented:

- Convolve input with 3 high-pass SRM kernels.
- Concatenate with RGB → `in_channels=6`.
- Requires custom first-layer initialization.

**Status:** Phase 3 experimental only. Not part of MVP or Phase 2. SRM and ELA are **separate** experimental paths.

---

## Encoder Alternatives (Phase 3)

| Encoder | Parameters | Notes |
|---|---|---|
| ResNet34 | ~24M | **Locked MVP baseline** |
| EfficientNet-B0 | ~5M | Lighter; worth testing if overfitting observed |
| EfficientNet-B1 | ~8M | Middle ground |

Encoder comparison is Phase 3 bonus work only.
