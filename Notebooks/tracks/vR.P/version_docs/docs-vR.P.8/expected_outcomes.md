# Expected Outcomes — vR.P.8: ELA + Gradual Encoder Unfreeze

| Field | Value |
|-------|-------|
| **Version** | vR.P.8 |
| **Parent** | vR.P.3 (ELA as input, Pixel F1 = 0.6920) |
| **Change** | Progressive encoder unfreeze (3 stages, 40 epochs) |
| **Risk Level** | MODERATE |

---

## 1. Predictions

### Scenario A: POSITIVE (40%)

Progressive unfreezing allows the encoder to adapt mid/high-level features to ELA's unique distribution. ELA maps have fundamentally different spatial statistics from ImageNet (sparse, high-contrast, artifact-focused). By gradually unfreezing deeper layers, the encoder learns ELA-specific feature detectors while preserving low-level edge/texture representations. The 100× lower encoder LR prevents catastrophic forgetting.

**Expected:** Pixel F1 = 0.74–0.80 (+5 to +11pp from P.3). IoU improves proportionally. The improvement comes primarily from Stage 1 (layer4 adaptation), with Stage 2 providing a smaller incremental gain.

### Scenario B: NEUTRAL (35%)

The BN-only adaptation from P.3 was already sufficient for ELA. The frozen convolutional weights, despite being trained on ImageNet, extract useful spatial features from ELA maps (edges are edges, regardless of the input domain). Unfreezing deeper layers adds trainable parameters but the ELA signal does not benefit significantly from adapted conv features because the low-level edge/texture detectors already fire appropriately on ELA patterns.

**Expected:** Pixel F1 within ±2pp of 0.6920. The 40-epoch budget provides longer training but no structural improvement over P.3.

### Scenario C: NEGATIVE (25%)

Unfreezing disrupts previously learned features. The transition dips at epoch 11 and 26 cause early stopping before the model can recover (even with patience reset). Or: the larger number of trainable parameters (~5M at Stage 2 vs ~500K at Stage 0) causes overfitting on the relatively small CASIA v2.0 training set. AMP may also interact poorly with the optimizer rebuild.

**Expected:** Pixel F1 < 0.67. The model fails to recover from transition disruption, or overfits with the unfrozen encoder.

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| **STRONG POSITIVE** | Pixel F1 ≥ 0.78 (≥ +8.8pp from P.3) |
| **POSITIVE** | Pixel F1 ≥ 0.7120 (≥ +2pp from P.3) |
| **NEUTRAL** | Pixel F1 within ±2pp of 0.6920 |
| **NEGATIVE** | Pixel F1 < 0.6720 (< −2pp from P.3) |

---

## 3. What to Watch For

1. **Stage transition dips:** Val loss should spike briefly at epochs 11 and 26 when new layers start updating. Watch whether the model recovers within 3–5 epochs. If it doesn't, the unfreeze may be too aggressive.

2. **Early stopping vs. stage transitions:** Patience is reset at transitions. Monitor whether the model triggers early stopping within a single stage (OK — means that stage converged) vs. being unable to improve at all after a transition (bad — means the unfreeze disrupted features).

3. **Train-val gap after Stage 2:** With ~5M trainable parameters and CASIA's ~8.8K training images, overfitting is a real risk. If train loss drops much faster than val loss after epoch 26, the model is memorizing.

4. **Optimizer rebuild correctness:** After each transition, verify the optimizer has the correct number of param groups (1 for Stage 0, 2 for Stages 1–2) and the LRs match (1e-5 encoder, 1e-3 decoder).

5. **Per-stage Pixel F1 trajectory:** Does Pixel F1 improve at each stage, or does the improvement only come from one stage? This informs whether layer3 adaptation is valuable or only layer4 matters.

6. **AMP compatibility:** GradScaler is recreated at transitions. Watch for NaN/Inf loss values in the first batch after a transition — these indicate scale factor mismatch.

---

## 4. Comparison: P.3 vs P.8 vs P.2

| Aspect | vR.P.2 (RGB + static unfreeze) | vR.P.3 (ELA + frozen) | vR.P.8 (ELA + progressive) |
|--------|------|------|------|
| Input | RGB | ELA | ELA |
| Encoder adaptation | layer3+4 from epoch 1 | BN only | **Progressive: frozen → L4 → L3+L4** |
| Encoder LR | 1e-5 (constant) | N/A | 1e-5 (from Stage 1) |
| Max epochs | 25 | 25 | 40 |
| Trainable params | ~5M | ~500K | ~500K → ~2M → ~5M |
| Optimizer rebuild | Never | Never | At epochs 11, 26 |

---

## 5. If NEGATIVE — Next Steps

- **Increase Stage 0 warmup:** Try 15 epochs frozen before any unfreezing (the decoder may need more warmup)
- **Freeze layer3 entirely:** Only unfreeze layer4 (2-stage schedule instead of 3-stage)
- **Lower encoder LR further:** Try 1e-6 instead of 1e-5 for more conservative adaptation
- **Combine with P.7:** Run ELA + extended training (50 epochs, fully frozen) as a simpler alternative
- The result still informs whether ELA features benefit from encoder adaptation at all
