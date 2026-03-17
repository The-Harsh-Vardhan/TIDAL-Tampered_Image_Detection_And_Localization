# Expected Outcomes — vR.P.4: 4-Channel Input (RGB + ELA)

| Field | Value |
|-------|-------|
| **Version** | vR.P.4 |
| **Parent** | vR.P.3 (ELA as input) |
| **Change** | 4-channel input (3 RGB + 1 ELA grayscale), conv1 unfrozen |
| **Risk Level** | MODERATE |

---

## 1. Predictions

### Scenario A: POSITIVE (40%)

The combined RGB + ELA signal provides richer information than either alone. The encoder receives natural image features through RGB channels (where pretrained weights are optimal) AND forensic features through the ELA channel. The decoder can learn to fuse both signals for precise localization. The conv1 layer quickly adapts its 4th-channel filter to extract meaningful ELA features.

**Expected:** Pixel F1 improves over both vR.P.2 (RGB only) and vR.P.3 (ELA only).

### Scenario B: NEUTRAL (35%)

The ELA channel adds marginal value. The model predominantly relies on RGB channels (where pretrained features are strong) and effectively ignores or lightly uses the ELA channel. Performance is similar to the better of vR.P.2 and vR.P.3.

### Scenario C: NEGATIVE (25%)

The 4th channel introduces noise or conflicts with the pretrained feature pipeline. Possible causes:
- Conv1's new weights destabilize early feature extraction
- The ELA grayscale signal is too sparse/noisy to provide useful gradients
- The mixed normalization (ImageNet + ELA) creates numerical issues

**Risk factors:**
- Modified conv1 breaks the pretrained initialization symmetry
- The VRAM increase from 4ch might require smaller effective batch size
- Previous attempt (vK.11-12) with 4ch failed (though confounded by 4 other changes)

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| **POSITIVE** | Pixel F1 > max(vR.P.2, vR.P.3) by >= 2pp |
| **NEUTRAL** | Pixel F1 within +/-2pp of max(vR.P.2, vR.P.3) |
| **NEGATIVE** | Pixel F1 < max(vR.P.2, vR.P.3) by > 2pp |

---

## 3. What to Watch For

1. **Conv1 gradient magnitudes:** If the 4th channel weights receive very small or very large gradients relative to the decoder, this indicates the ELA signal is either irrelevant or destabilizing.

2. **ELA channel feature maps:** After training, visualize the conv1 filters for channel 3. If they learn edge-detection-like patterns, ELA is being used meaningfully.

3. **Authentic vs tampered ELA patterns:** Authentic images should produce near-zero ELA maps, making the 4th channel effectively zero. The model should learn that ch3 ~= 0 correlates with authentic (all-zero mask).

4. **Training loss convergence speed:** With both RGB and ELA, the model has more information. If convergence is faster than P.3 (ELA only), this suggests the RGB channels provide a useful initialization signal.

5. **VRAM usage:** 4-channel input increases memory by ~33% vs 3-channel. Watch for OOM errors with batch_size=16 on T4 (15GB).

---

## 4. Comparison with Prior Approaches

| Version | Input | Channels | Encoder Strategy | Status |
|---------|-------|----------|-----------------|--------|
| vR.P.2 | RGB only | 3 | Partially unfrozen (layer3+4) | Completed |
| vR.P.3 | ELA only | 3 | Frozen + BN unfrozen | Completed |
| **vR.P.4** | **RGB + ELA gray** | **4** | **Frozen + conv1 unfrozen + BN unfrozen** | **This** |

### Why vR.P.4 Should Succeed Where vK.11-12 Failed

vK.11-12 changed 5 variables simultaneously: 4-channel input, edge loss, classification head, focal loss, and resolution. The failure was confounded — impossible to attribute to any single change. vR.P.4 isolates the 4-channel input as the **sole variable**, following the strict single-variable ablation protocol.

---

## 5. If NEGATIVE — Next Steps

- Consider unfreezing more encoder layers to allow deeper adaptation (combine P.2 + P.4 strategies)
- Try 6-channel (RGB 3ch + ELA RGB 3ch) instead of 4-channel
- Consider a separate ELA branch (dual-encoder architecture) instead of early concatenation
- The result still informs whether multi-signal input adds value for pretrained localization
