# Expected Outcomes — vR.P.7: ELA + Extended Training

| Field | Value |
|-------|-------|
| **Version** | vR.P.7 |
| **Parent** | vR.P.3 (ELA as input, Pixel F1 = 0.6920, best epoch = 25/25) |
| **Change** | max_epochs: 25 → 50, patience: 7 → 10 |
| **Risk Level** | LOW |

---

## 1. Predictions

### Scenario A: POSITIVE (55%)

The model continues learning past epoch 25. The ReduceLROnPlateau scheduler reduces LR 2-3 more times, allowing the model to refine boundary predictions. The decoder learns finer-grained ELA patterns that weren't captured in 25 epochs.

**Expected:** Pixel F1 = 0.72-0.76 (+3 to +7pp from P.3's 0.6920). IoU improves proportionally. Image accuracy may reach 88-90%.

**Reasoning:**
- P.3's val Pixel F1 trajectory was still steeply rising at epoch 25 (0.4051 → 0.7243)
- LR reductions (1e-3 → 5e-4 at epoch 9, → 2.5e-4 at epoch 20) leave room for further 0.5× steps
- With 3.17M trainable params and 8,829 training images (1:359 ratio), overfitting risk is moderate but manageable

### Scenario B: NEUTRAL (30%)

The model plateaus shortly after epoch 25. The val loss flattens, LR reduces to minimum, and early stopping triggers around epoch 30-35. Final metrics are within ±2pp of P.3.

**Expected:** Pixel F1 = 0.68-0.71 (within ±2pp of 0.6920). Training effectively ended near where P.3 stopped.

**Reasoning:** The steep improvement in P.3 may have been approaching an asymptote. The frozen encoder's features may have been fully exploited by epoch 25. Further training just overcooks the same features.

### Scenario C: NEGATIVE (15%)

Extended training causes overfitting despite early stopping. The model starts memorizing training patterns, and val metrics degrade. This is unlikely given the small number of trainable parameters but possible if the decoder overfits to specific GT mask patterns.

**Expected:** Pixel F1 < 0.67 (more than 2pp below P.3). Early stopping triggers but not before some degradation.

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| **STRONG POSITIVE** | Pixel F1 ≥ 0.75 (≥ +5.8pp from P.3) |
| **POSITIVE** | Pixel F1 ≥ 0.7120 (≥ +2pp from P.3) |
| **NEUTRAL** | Pixel F1 within ±2pp of 0.6920 (0.6720 to 0.7120) |
| **NEGATIVE** | Pixel F1 < 0.6720 (> 2pp below P.3) |

---

## 3. What to Watch For

1. **Best epoch position:** If best epoch falls in 26-40 range, extended training was clearly needed. If best epoch remains at 25 or earlier, the original budget was sufficient.

2. **LR reduction schedule:** Track when each LR reduction occurs. If multiple reductions happen after epoch 25, the model is finding new learning opportunities. If LR hits min_lr quickly, the model has converged.

3. **Train-val gap evolution:** If the gap widens significantly after epoch 25, overfitting is occurring despite the small trainable parameter count. Monitor train_loss vs val_loss divergence.

4. **Pixel F1 trajectory shape:** Logarithmic (fast early, slow late) suggests diminishing returns. Linear suggests more room. Stepped (flat → jump after LR reduction) suggests the scheduler is doing its job.

5. **Image accuracy vs Pixel F1 correlation:** If image accuracy improves but Pixel F1 plateaus, the model is getting better at binary classification but not at mask quality. This would suggest the decoder needs more capacity, not more training.

6. **FP rate and FN rate trajectories:** P.3 achieved 2.7% FP and 28.6% FN. Watch whether extended training reduces FN further without increasing FP.

---

## 4. Comparison Points

### vs P.3 (Parent)

| Metric | P.3 | P.7 Expected (Scenario A) | Delta |
|--------|-----|---------------------------|-------|
| Pixel F1 | 0.6920 | 0.72-0.76 | +3 to +7pp |
| IoU | 0.5291 | 0.56-0.61 | +3 to +8pp |
| Pixel AUC | 0.9528 | 0.96-0.97 | +1 to +2pp |
| Image Accuracy | 86.79% | 88-90% | +1 to +3pp |
| Best Epoch | 25 | 30-45 | Extended |
| Epochs Trained | 25 | 35-50 | Extended |

### vs P.4 (4ch RGB+ELA)

P.4 achieved Pixel F1 = 0.7053 with a fundamentally different input strategy. If P.7 exceeds 0.7053 with ELA-only input and merely more training, this proves that **ELA-only + sufficient training beats the 4-channel approach** — a significant finding.

### vs ETASR Best (vR.1.6)

vR.1.6 achieved 90.23% image accuracy. If P.7 approaches or exceeds this on image accuracy while also providing localization masks, it would establish the pretrained track as strictly superior.

---

## 5. If NEUTRAL or NEGATIVE — Next Steps

- **If NEUTRAL:** The model has converged at ~0.69 Pixel F1 with this architecture. Next experiment should change a more impactful variable:
  - vR.P.8: ELA + gradual unfreeze (combine P.3's input with P.2's strategy)
  - vR.P.9: Focal+Dice loss (address hard-example mining)

- **If NEGATIVE:** Overfitting occurred despite safeguards. Diagnose by examining where train-val gap diverged. Consider:
  - Reducing max_epochs to 35 instead of 50
  - Adding weight decay to optimizer
  - Adding Dropout to decoder

---

## 6. Runtime Estimate

| Component | Time |
|-----------|------|
| ELA computation per image | ~50ms |
| ELA stats computation (500 images) | ~30s |
| Training per epoch | ~3-4 min (with AMP) |
| Total (50 epochs max, likely early stop ~35-45) | ~120-180 min |
| Evaluation | ~5 min |
| **Total session** | **~130-190 min** |

Well within Kaggle T4/P100 session limits (9-12 hours).
