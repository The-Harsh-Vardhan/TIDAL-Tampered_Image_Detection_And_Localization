# Expected Outcomes — vR.P.11: Higher Resolution (512x512)

| Field | Value |
|-------|-------|
| **Version** | vR.P.11 |
| **Parent** | vR.P.3 (ELA as input, BCE+Dice loss, Pixel F1 = 0.6920) |
| **Change** | Higher resolution (512x512) + Focal+Dice loss + gradient accumulation |
| **Risk Level** | LOW-MODERATE |

---

## 1. Scenarios

### Scenario A: POSITIVE (50%)

Higher resolution preserves fine boundary details and small tampered regions that are lost at 384x384. The ELA signal benefits from more spatial detail — compression artifact patterns are inherently spatial and resolution-dependent. Combined with Focal+Dice loss, the model achieves meaningfully better boundary precision.

**Expected metrics:**
- Pixel F1: 0.72-0.80 (+3 to +11pp from P.3)
- IoU: proportional improvement
- Image accuracy: stable or improved

### Scenario B: NEUTRAL (30%)

The ELA signal at 384x384 already captures sufficient forensic information. Higher resolution adds more pixels but not more useful signal — the additional detail is dominated by noise or irrelevant compression artifacts. Gradient accumulation successfully maintains optimization quality, but the resolution gain is marginal.

**Expected metrics:**
- Pixel F1: 0.67-0.72 (within +/-2pp of P.3)

### Scenario C: NEGATIVE (20%)

Higher resolution with batch_size=8 introduces training instability despite gradient accumulation. The per-mini-batch gradient variance is higher with fewer samples, and the 2-step accumulation doesn't fully compensate. Alternatively, the frozen ResNet-34 encoder doesn't have sufficient capacity to leverage the additional spatial detail from 512x512 input.

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

1. **VRAM usage** — Monitor GPU memory with batch_size=8 at 512x512. If OOM occurs, reduce to batch_size=4 with GRAD_ACCUM_STEPS=4.
2. **Training speed** — Expect ~2x slower per epoch vs P.3 (1.78x more pixels, but half batch size with accumulation means ~same GPU throughput per step).
3. **Convergence rate** — Higher resolution may need more epochs to converge. Watch if early stopping triggers before epoch 30.
4. **Boundary quality** — Visually inspect predicted masks at 512x512. Boundaries should be sharper and more precise than 384x384 predictions.
5. **Small region detection** — Check if tiny tampered regions (< 1% of image) are better detected at higher resolution.
6. **Gradient accumulation correctness** — Verify that training loss is comparable to P.3's range (accounting for Focal vs BCE magnitude difference).

---

## 4. If NEGATIVE

1. **Check VRAM** — Ensure no silent OOM or memory pressure affecting training
2. **Try batch_size=4, accum=4** — More aggressive accumulation for stable gradients
3. **Try 448x448** — Intermediate resolution as a compromise
4. **Try BCE+Dice at 512** — Isolate whether the issue is resolution or the loss change
5. **The conclusion is still valuable** — Proves that resolution is not the bottleneck at the current encoder capacity

---

## 5. Results Table Template

| Version | Input | Resolution | Encoder | Loss Function | Pixel F1 | IoU | Pixel AUC | Image Acc |
|---------|-------|------------|---------|---------------|----------|-----|-----------|-----------|
| vR.P.3 | ELA | 384x384 | ResNet-34 | BCE + Dice | 0.6920 | -- | -- | -- |
| **vR.P.11** | **ELA** | **512x512** | **ResNet-34** | **Focal + Dice** | **Pending** | **Pending** | **Pending** | **Pending** |
