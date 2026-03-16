# Expected Outcomes — vR.P.10: ELA + Attention Modules (CBAM)

| Field | Value |
|-------|-------|
| **Version** | vR.P.10 |
| **Parent** | vR.P.3 (Pixel F1 = 0.6920, IoU = 0.5291, Img Acc = 86.79%) |
| **Change** | Focal+Dice loss + CBAM attention in UNet decoder |
| **Risk Level** | LOW-MODERATE |

---

## 1. Predictions

### Scenario A: POSITIVE (40%)

CBAM attention helps the decoder selectively amplify forensic features (compression artifacts, boundary signals) while suppressing background noise. The spatial attention component focuses on tampered regions, improving segmentation boundary quality. Combined with Focal loss's hard-pixel focus, both mechanisms work synergistically — Focal loss provides better gradients for boundary pixels, and attention provides better features for processing those gradients.

**Expected:** Pixel F1 improves to 0.72–0.76 (+3–7pp from P.3). IoU improves proportionally. Image accuracy may improve slightly due to better binary mask quality.

### Scenario B: NEUTRAL (35%)

CBAM attention learns reasonable channel and spatial weights but does not significantly improve over the already-effective decoder features. The ResNet-34 encoder with BN unfrozen already provides good ELA feature extraction. The ~11K additional attention parameters are too few to meaningfully reshape the feature landscape. Focal loss alone (without attention) may capture most of the possible improvement.

**Expected:** Pixel F1 within ±2pp of P.3 (0.67–0.71).

### Scenario C: NEGATIVE (25%)

CBAM attention introduces a slight regularization effect that slows convergence, or the attention weights collapse to near-uniform values (effectively becoming identity). Two simultaneous changes (loss + attention) may interact poorly — Focal loss changes the gradient landscape while attention changes the feature landscape, potentially creating optimization instability.

**Expected:** Pixel F1 drops below 0.67 (< P.3 - 2pp). Training curves may show increased variance.

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| **STRONG POSITIVE** | Pixel F1 ≥ 0.7700 (+7.8pp from P.3) |
| **POSITIVE** | Pixel F1 ≥ 0.7120 (+2.0pp from P.3) |
| **NEUTRAL** | Pixel F1 in [0.6720, 0.7120] (±2pp from P.3) |
| **NEGATIVE** | Pixel F1 < 0.6720 (> 2pp below P.3) |

---

## 3. What to Watch For

1. **Attention weight visualization:** Do the learned spatial attention maps correlate with actual tampered regions? If attention focuses on arbitrary locations, it may not be learning forensically useful patterns.

2. **Per-decoder-block contribution:** Attention in early decoder blocks (256ch) vs late decoder blocks (16ch) may vary in effectiveness. Early blocks have more channels to recalibrate; late blocks have more spatial detail to focus on.

3. **Training loss convergence speed:** CBAM adds learnable parameters that start randomly initialized. This may cause higher initial loss or slower early convergence compared to P.3.

4. **Pixel F1 vs IoU correlation:** Attention that improves boundary quality should improve both F1 and IoU. If F1 improves but IoU doesn't, the improvement may be from threshold effects rather than genuine attention benefit.

5. **FP rate vs FN rate trade-off:** Spatial attention should reduce false positives (by learning to suppress predictions in non-tampered regions) more than false negatives.

6. **Comparison with P.9 (Focal+Dice only):** P.9 tests the loss change alone. Comparing P.10 vs P.9 isolates the attention effect. Comparing P.10 vs P.3 shows the combined effect.

---

## 4. Comparison Points

### vs P.3 (Parent — direct comparison)

| Metric | P.3 | P.10 Expected | Delta |
|--------|-----|---------------|-------|
| Pixel F1 | 0.6920 | 0.70–0.76 | +1–7pp |
| Pixel IoU | 0.5291 | 0.54–0.62 | +1–9pp |
| Pixel AUC | 0.9528 | 0.95–0.97 | 0–2pp |
| Image Accuracy | 86.79% | 85–89% | -2–+2pp |

### vs P.9 (Focal+Dice only — isolates attention effect)

If P.9 results are available, compare P.10 - P.9 to isolate the attention module contribution.

### vs P.4 (Best absolute Pixel F1 = 0.7053)

If P.10 exceeds 0.7053, it becomes the new Pixel F1 champion. This would demonstrate that attention + loss optimization > 4-channel input fusion.

---

## 5. If NEUTRAL or NEGATIVE — Next Steps

**If NEUTRAL:**
- The attention parameters may need more training time. Consider P.10.1 with extended training (50 epochs, patience=10) similar to P.7.
- Try SE instead of CBAM — simpler channel-only attention may work better on this dataset.
- Try SMP's built-in `decoder_attention_type='scse'` for comparison.

**If NEGATIVE:**
- Two simultaneous changes may blame either loss or attention. Check P.9 results to determine if Focal loss alone was beneficial.
- The CBAM reduction ratio (r=16) may be too aggressive for small decoder channels (e.g., 16//16=1). Try r=8 or r=4.
- Add attention only to the first 2-3 decoder blocks (higher channels) rather than all 5.

---

## 6. Results Table Template

| Metric | P.3 (Parent) | P.10 (This Run) | Delta | Assessment |
|--------|-------------|-----------------|-------|------------|
| Pixel F1 | 0.6920 | _____ | _____ | |
| Pixel IoU | 0.5291 | _____ | _____ | |
| Pixel AUC | 0.9528 | _____ | _____ | |
| Pixel Precision | 0.8356 | _____ | _____ | |
| Pixel Recall | 0.5905 | _____ | _____ | |
| Image Accuracy | 86.79% | _____ | _____ | |
| Image Macro F1 | 0.8560 | _____ | _____ | |
| Image ROC-AUC | 0.9502 | _____ | _____ | |
