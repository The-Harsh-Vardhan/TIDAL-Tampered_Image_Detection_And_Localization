# Expected Outcomes — vR.P.3: ELA as Input

| Field | Value |
|-------|-------|
| **Version** | vR.P.3 |
| **Parent** | vR.P.2 (Gradual unfreeze, RGB input) |
| **Change** | ELA as input (replace RGB), encoder frozen + BN unfrozen |
| **Risk Level** | MODERATE-HIGH |

---

## 1. Predictions

### Scenario A: POSITIVE (35%)

ELA provides a stronger forensic signal than RGB for localization. The brightness-scaled ELA map directly highlights compression artifacts at tampered boundaries. The pretrained encoder, with BN adapted to ELA statistics, can still extract useful spatial features (edges, textures, gradients) even though the input distribution differs from ImageNet.

**Expected:** Pixel F1 improves over vR.P.2 (if vR.P.2 achieves ~0.15-0.30, expect 0.25-0.40).

### Scenario B: NEUTRAL (30%)

ELA provides a different signal that is roughly as informative as RGB for pretrained features. The encoder's low-level features (edges, gradients in conv1/layer1) still fire on ELA's brightness patterns, and the decoder learns to interpret them. Performance is similar to vR.P.2.

### Scenario C: NEGATIVE (35%)

ELA's distribution is too different from ImageNet for effective feature transfer. The frozen encoder weights were learned on natural images with specific color/texture statistics. ELA maps are sparse, high-contrast, and lack the texture diversity of natural images. Even with BN adaptation, the mid-level and high-level features (layer3/layer4) may not activate meaningfully. The model may produce worse localization than RGB input.

**Risk factors:**
- ELA maps are fundamentally different from ImageNet images
- Pretrained features for "cat fur" or "building edges" may not transfer to "compression artifact boundaries"
- BN adaptation is a weaker form of domain adaptation than fine-tuning (vR.P.2)
- Going back to frozen encoder removes the domain adaptation gains from vR.P.2

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| **POSITIVE** | Pixel F1 > vR.P.2's Pixel F1 by ≥ 2pp |
| **NEUTRAL** | Pixel F1 within ±2pp of vR.P.2 |
| **NEGATIVE** | Pixel F1 < vR.P.2's Pixel F1 by > 2pp |

---

## 3. What to Watch For

1. **Training loss convergence:** If ELA is too foreign for the encoder, training loss may plateau at a high value (decoder can't learn useful mappings from meaningless encoder features).

2. **Pixel F1 vs classification accuracy trade-off:** ELA might be better for image-level detection (bright = tampered) but worse for precise pixel boundaries (ELA boundaries are blurry).

3. **Authentic image masks:** ELA should produce near-zero maps for authentic images (no compression mismatch). This could help reduce false positives at pixel level.

4. **Non-JPEG images:** CASIA v2.0 contains TIF and BMP images. ELA requires JPEG re-compression — these formats may produce misleading ELA maps (everything appears as an artifact).

5. **LR scheduler behavior:** With a frozen encoder and simpler optimization landscape, the scheduler may trigger earlier or not at all.

---

## 4. Comparison with Track 1 (ETASR)

| Aspect | ETASR (vR.1.x) | vR.P.3 |
|--------|----------------|--------|
| Input | ELA 128×128 | ELA 384×384 (3× resolution) |
| Encoder | Trained from scratch (29.5M params) | Pretrained ResNet-34 (frozen) |
| Decoder | Dense (classification only) | UNet (pixel-level localization) |
| Output | 2-class softmax | 384×384 binary mask |
| ELA preprocessing | Same (Q=90, brightness scale) | Same (Q=90, brightness scale) |

vR.P.3 tests whether pretrained features + UNet decoder can **localize** forgery from ELA at higher resolution, whereas Track 1 can only **classify** at lower resolution.

---

## 5. If NEGATIVE — Next Steps

- vR.P.4 (RGB + ELA 4-channel) may combine the best of both signals
- Consider unfreezing encoder with ELA (combine vR.P.2 + vR.P.3 strategies)
- The result still informs whether ELA signal is valuable for pretrained encoders
