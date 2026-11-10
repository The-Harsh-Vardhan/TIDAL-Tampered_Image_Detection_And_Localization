# Model Architecture

---

## Baseline Model (MVP)

The architecture is a **baseline aligned with assignment constraints** — a pretrained U-Net suitable for Colab's T4 GPU. Research-frontier models (edge-enhanced, multi-trace, transformer hybrids) are documented in `11_Research_Alignment.md` as future work.

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

**Research context:** This architecture follows the transfer-learning pattern described in comprehensive survey papers (P14, P15). U-Net with pretrained encoders is a well-established baseline for dense prediction tasks including forgery localization. The reference notebook `image-detection-with-mask.ipynb` also uses a U-Net architecture with classification and segmentation outputs, confirming the viability of this approach for CASIA-based tamper detection.

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

**Locked decision for MVP:** Derive the tamper score from the pixel-level probability map using a top-k mean. No separate classification head.

```python
prob_map = torch.sigmoid(logits)           # (B, 1, H, W)
flat = prob_map.view(prob_map.size(0), -1)
k = max(int(flat.size(1) * 0.001), 32)
tamper_score = flat.topk(k=min(k, flat.size(1)), dim=1).values.mean(dim=1)
is_tampered = tamper_score >= pixel_threshold
```

The **same threshold** selected on the validation set is used for both pixel-level binarization and image-level detection. This is intentional: a single operating point keeps the system simple and avoids a second sweep.

**Known limitation:** top-k mean is less sensitive than `max(prob_map)`, but it is still a handcrafted image-level heuristic rather than a learned classifier.

**Research note:** The `image-detection-with-mask.ipynb` reference notebook implements a dual-output architecture (`UNetWithClassifier`) with a separate classification head using `AdaptiveAvgPool2d` on the bottleneck features. This is a viable Phase 2 alternative that eliminates dependence on handcrafted top-k pooling. See the research paper P2 (Dual-task Classification + Segmentation) for academic support.

---

## Optional: Error Level Analysis (Phase 2)

ELA highlights regions with inconsistent compression artifacts — a signal that may complement RGB for splicing detection. This technique exploits the fact that JPEG re-compression produces different error levels in tampered vs. authentic regions.

**Research support:** The `document-forensics-using-ela-and-rpa.ipynb` reference notebook demonstrates ELA-based forensic analysis achieving meaningful separation between authentic and tampered images. Research papers P1 (ELA-CNN Hybrid) and P7 (Enhanced ELA + CNN, 96.21% accuracy on CASIA v2.0) provide additional support.

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

**ELA parameter guidance:** The reference notebook `document-forensics-using-ela-and-rpa.ipynb` performs grid search over scale (10, 20, 30), quality (90), and threshold (5, 7, 9) parameters, finding optimal performance at scale=30, quality=90, threshold=9. For the Phase 2 ELA channel, a quality of 90 is recommended as the default compression level.

---

## Optional: SRM Features (Phase 3)

Spatial Rich Model kernels extract noise residuals. Research papers P13 (EMT-Net, AUC=0.987) and P17 (ME-Net, F1=0.905) demonstrate that noise-domain features captured by SRM are critical for strong localization performance.

If implemented:

- Convolve input with 3 high-pass SRM kernels.
- Concatenate with RGB → `in_channels=6`.
- Requires custom first-layer initialization.

**Status:** Phase 3 experimental only. Not part of MVP or Phase 2. SRM and ELA are **separate** experimental paths.

---

## Optional: Dual-Task Architecture (Phase 2)

Inspired by the `image-detection-with-mask.ipynb` reference notebook and research paper P2, a dual-task architecture adds a classification head to the U-Net bottleneck:

```python
# Conceptual — not part of MVP
classifier = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 256),   # 512 = bottleneck channels for ResNet34
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, 2),     # 2 classes: authentic vs tampered
)
```

This replaces the handcrafted top-k mean score with a learned classifier, potentially improving image-level detection further. The reference notebook uses `CrossEntropyLoss` (with Focal Loss variant) for classification and `BCEWithLogitsLoss + DiceLoss` for segmentation, weighted by separate alpha/beta coefficients.

**Status:** Phase 2 optional enhancement. Not part of MVP.

---

## Encoder Alternatives (Phase 3)

| Encoder | Parameters | Notes |
|---|---|---|
| ResNet34 | ~24M | **Locked MVP baseline** |
| EfficientNet-B0 | ~5M | Lighter; worth testing if overfitting observed |
| EfficientNet-B1 | ~8M | Middle ground |

Encoder comparison is Phase 3 bonus work only.
