# Expected Outcomes — vR.P.6: EfficientNet-B0 Encoder

| Field | Value |
|-------|-------|
| **Version** | vR.P.6 |
| **Parent** | vR.P.1 (ResNet-34, frozen, RGB) |
| **Change** | Replace ResNet-34 with EfficientNet-B0 encoder |
| **Risk Level** | MODERATE |

---

## 1. Predictions

### Scenario A: POSITIVE (30%)

EfficientNet-B0's squeeze-excite attention and higher ImageNet accuracy translate to better forensic features. The decoder receives more discriminative skip connections despite smaller channel sizes. Pixel F1 improves over vR.P.1's ResNet-34 baseline.

**Expected:** Pixel F1 > vR.P.1 by >= 2pp. Faster convergence due to better feature quality.

### Scenario B: NEUTRAL (40%)

Both encoders extract features of comparable quality for forensic segmentation. EfficientNet-B0's advantages (attention, efficiency) are offset by ResNet-34's larger skip connection channels. Performance is within +/-2pp of vR.P.1.

**Expected:** Similar metrics, but with ~35% smaller model footprint. Still valuable as an efficiency result.

### Scenario C: NEGATIVE (30%)

EfficientNet-B0's smaller skip connection channels ([16, 24, 40, 112, 320] vs [64, 64, 128, 256, 512]) provide insufficient spatial detail for precise localization. The forensic task requires different features than ImageNet classification, and ResNet-34's simpler, wider architecture transfers better.

**Risk factors:**
- Skip connection channels are 2-4x smaller at every stage — decoder has less spatial information
- MBConv blocks optimized for classification, not segmentation
- EfficientNet's depthwise separable convolutions may miss cross-channel forensic patterns
- No project evidence (scored 3/10 on project evidence in architecture analysis)

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| **POSITIVE** | Pixel F1 > vR.P.1 by >= 2pp |
| **NEUTRAL** | Pixel F1 within +/-2pp of vR.P.1 |
| **NEGATIVE** | Pixel F1 < vR.P.1 by > 2pp |

---

## 3. What to Watch For

1. **Skip connection quality:** EfficientNet-B0 skip channels are much narrower. If the decoder struggles to reconstruct fine-grained masks, this is the bottleneck.

2. **Squeeze-excite activation patterns:** After training, check if the SE blocks in the frozen encoder are weighting forensically-relevant channels higher.

3. **Convergence speed:** EfficientNet-B0 has fewer decoder parameters (~400K vs ~500K). If training converges faster, the features are more transfer-friendly.

4. **Memory usage:** Expect ~2.5 GB vs ~3 GB for ResNet-34. If memory savings allow batch_size=24 or 32, this could indirectly improve training.

5. **Edge quality in predictions:** With narrower skip connections, predicted masks may have softer edges. Compare mask sharpness visually.

---

## 4. Architecture Decision Matrix

From the project's architecture analysis:

| Criterion (Weight) | ResNet-34 | EfficientNet-B0 |
|---------------------|-----------|-----------------|
| Project evidence (30%) | **10** (v6.5 proven) | 3 (untested) |
| Literature support (20%) | 8 | 6 (survey mention) |
| Parameter efficiency (10%) | 6 | **10** |
| T4 compatibility (10%) | 9 | **10** |
| Feature quality (15%) | 7 | 8 |
| Implementation risk (15%) | **10** | 8 |
| **Weighted Score** | **8.4** | **6.5** |

EfficientNet-B0 wins on efficiency metrics but lacks forensic evidence. This experiment provides that missing evidence.

---

## 5. If NEGATIVE — Next Steps

- The result confirms ResNet-34 as the optimal encoder for this dataset/task
- Skip connection width is likely the bottleneck — consider EfficientNet-B4 (wider channels) if efficiency is still desired
- The information value is high regardless: it closes the encoder comparison loop
