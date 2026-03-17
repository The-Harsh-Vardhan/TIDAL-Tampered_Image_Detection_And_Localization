# Expected Outcomes — vR.P.12: ELA + Data Augmentation

| Field | Value |
|-------|-------|
| **Version** | vR.P.12 |
| **Parent** | vR.P.3 (ELA as input, BCE+Dice loss, Pixel F1 = 0.6920) |
| **Change** | Add controlled Albumentations augmentation + Focal+Dice loss |
| **Risk Level** | LOW |

---

## 1. Scenarios

### Scenario A: POSITIVE (55%)

The augmented training data forces the model to learn orientation-invariant and position-invariant forensic features. The model generalizes better to unseen tampering patterns in the test set. Flips and rotations are particularly effective because CASIA v2.0 contains directional splicing patterns that the model previously memorized.

**Expected metrics:**
- Pixel F1: 0.72-0.80 (+3 to +11pp from P.3)
- IoU: proportional improvement
- Image accuracy: stable or improved

### Scenario B: NEUTRAL (30%)

The CASIA v2.0 training set (~8,800 images) is already diverse enough that augmentation provides minimal benefit. The model's current limitations are driven by encoder capacity or ELA signal quality, not training data diversity. Augmentation adds training time without meaningful improvement.

**Expected metrics:**
- Pixel F1: 0.67-0.72 (within +/-2pp of P.3)

### Scenario C: NEGATIVE (15%)

Even mild augmentations (ShiftScaleRotate, GaussianBlur) introduce enough interpolation artifacts to slightly corrupt the ELA forensic signal. The model learns to ignore these artifacts, which unfortunately overlap with genuine tampering signals. This is unlikely with the conservative parameters chosen but possible.

**Expected metrics:**
- Pixel F1: < 0.67 (more than 2pp below P.3)

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| STRONG POSITIVE | Pixel F1 >= 0.80 (>= +10.8pp from P.3) |
| POSITIVE | Pixel F1 >= 0.7120 (>= +2pp from P.3) |
| NEUTRAL | Pixel F1 within +/-2pp of 0.6920 |
| NEGATIVE | Pixel F1 < 0.6720 |

---

## 3. What to Watch For

1. **Train vs val gap** — Augmentation should reduce the gap between training and validation loss. If the gap is still large, augmentation may be insufficient.
2. **Convergence speed** — Augmented training typically converges slower (more variation per epoch). This is expected and why EPOCHS=50.
3. **Val loss stability** — With augmentation, validation loss should be smoother (less overfitting oscillation).
4. **Edge quality** — Compare predicted mask boundaries with P.3. If augmentation corrupts fine edges, ShiftScaleRotate limits may be too aggressive.
5. **Orientation sensitivity** — If possible, test on rotated/flipped versions of test images to verify orientation invariance.
6. **Focal loss interaction** — Both augmentation and focal loss address different aspects of learning quality. Their combination should be additive, not conflicting.

---

## 4. If NEGATIVE

1. **Remove ShiftScaleRotate** — The interpolation from shift/scale/rotate is most likely to corrupt ELA signals. Try flips and rotations only.
2. **Remove GaussianBlur and BrightnessContrast** — Test with purely geometric (lossless) augmentations only.
3. **Reduce augmentation probability** — Lower p values to reduce how often augmentation is applied.
4. **Try BCE+Dice loss** — Isolate whether the issue is augmentation or the loss change by reverting to P.3's loss.
5. **The conclusion is still valuable** — Proves that this ELA pipeline is sensitive to even mild augmentation, suggesting the forensic signal is fragile.

---

## 5. Results Table Template

| Version | Input | Encoder | Resolution | Loss | Augmentation | Pixel F1 | IoU | Pixel AUC | Image Acc |
|---------|-------|---------|------------|------|-------------|----------|-----|-----------|-----------|
| vR.P.3 | ELA | ResNet-34 | 384x384 | BCE + Dice | None | 0.6920 | -- | -- | -- |
| **vR.P.12** | **ELA** | **ResNet-34** | **384x384** | **Focal + Dice** | **Albumentations (6 transforms)** | **Pending** | **Pending** | **Pending** | **Pending** |
