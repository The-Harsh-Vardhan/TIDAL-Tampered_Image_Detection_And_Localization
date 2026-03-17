# Expected Outcomes — vR.P.9: Focal + Dice Loss

| Field | Value |
|-------|-------|
| **Version** | vR.P.9 |
| **Parent** | vR.P.3 (ELA as input, BCE+Dice loss, Pixel F1 = 0.6920) |
| **Change** | Replace BCE+Dice with Focal+Dice loss |
| **Risk Level** | LOW-MODERATE |

---

## 1. Scenarios

### Scenario A: POSITIVE (45%)

Focal loss focuses gradient on hard boundary pixels, improving localization precision at tampered region edges. The Dice component maintains global region overlap quality. Expected improvement in both Pixel F1 and IoU, particularly for small tampered regions.

**Expected metrics:**
- Pixel F1: 0.72–0.78 (+3 to +9pp from P.3)
- IoU: proportional improvement
- Image accuracy: stable or improved

### Scenario B: NEUTRAL (35%)

The class imbalance in CASIA v2.0 is moderate enough that BCE loss already handles it well. Focal loss provides marginal benefit that falls within noise. The ELA signal is already strong enough that the loss function is not the bottleneck.

**Expected metrics:**
- Pixel F1: 0.67–0.72 (within ±2pp of P.3)

### Scenario C: NEGATIVE (20%)

Focal loss with gamma=2.0 is too aggressive for this dataset — the down-weighting of easy pixels removes useful gradient signal. Convergence is slower and the model underfits, or the alpha=0.25 weighting is mismatched for CASIA v2.0's particular class distribution.

**Expected metrics:**
- Pixel F1: < 0.67 (more than 2pp below P.3)

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| STRONG POSITIVE | Pixel F1 >= 0.78 (>= +8.8pp from P.3) |
| POSITIVE | Pixel F1 >= 0.7120 (>= +2pp from P.3) |
| NEUTRAL | Pixel F1 within ±2pp of 0.6920 |
| NEGATIVE | Pixel F1 < 0.6720 |

---

## 3. What to Watch For

1. **Convergence speed** — Focal loss can slow early training since it down-weights easy examples. If val_loss plateaus early, the gamma may be too high.
2. **Training loss magnitude** — Focal loss produces smaller loss values than BCE (by design). Don't compare raw loss numbers across P.3 and P.9.
3. **Boundary quality** — Visually inspect predicted masks. Focal loss should produce sharper boundaries at tampered region edges.
4. **Small region detection** — Check if tiny tampered regions (< 1% of image) are better detected.
5. **False positive rate** — Focal loss can sometimes increase false positives if gamma is too aggressive.
6. **Early stopping behavior** — The loss magnitude difference may affect ReduceLROnPlateau and early stopping timing.

---

## 4. If NEGATIVE

1. **Try gamma=1.0** — Less aggressive focal modulation, closer to standard CE behavior
2. **Try alpha=0.5** — Equal weighting for foreground/background
3. **Try Focal only** — Remove Dice component to isolate focal loss effect
4. **The conclusion is still valuable** — Proves that loss function is not the bottleneck for this dataset

---

## 5. Results Table Template

| Version | Input | Encoder | Loss Function | Pixel F1 | IoU | Pixel AUC | Image Acc |
|---------|-------|---------|---------------|----------|-----|-----------|-----------|
| vR.P.3 | ELA | ResNet-34 | BCE + Dice | 0.6920 | — | — | — |
| **vR.P.9** | **ELA** | **ResNet-34** | **Focal + Dice** | **Pending** | **Pending** | **Pending** | **Pending** |
