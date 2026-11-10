# Expected Outcomes — vR.1.5

| Field | Value |
|-------|-------|
| **Version** | vR.1.5 |
| **Parent** | vR.1.4 (BatchNorm — NEUTRAL, 88.75%) |
| **Change** | ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6) |
| **Risk Level** | LOW |

---

## 1. Predictions

### Scenario A: POSITIVE (Most Likely — 60%)

| Metric | Predicted | Parent (vR.1.4) | Delta |
|--------|-----------|-----------------|-------|
| Test Accuracy | 89.5–91.0% | 88.75% | +0.75–2.25pp |
| Macro F1 | 0.89–0.91 | 0.8852 | +0.005–0.025 |
| ROC-AUC | 0.96–0.97 | 0.9536 | +0.007–0.016 |
| Epochs | 15–30 (best 10–20) | 8 (best 3) | Much longer |

**Rationale:** ReduceLROnPlateau is one of the most reliable training improvements. It allows the optimizer to escape shallow local minima by reducing the step size. The BN warmup instability from vR.1.4 should be dampened because the LR will automatically reduce after the epoch 1 spike. Combined with BatchNorm's normalization and class weights, this creates a well-configured training pipeline.

### Scenario B: NEUTRAL (Possible — 30%)

| Metric | Predicted | Parent (vR.1.4) | Delta |
|--------|-----------|-----------------|-------|
| Test Accuracy | 88.5–89.2% | 88.75% | ±0.5pp |
| Macro F1 | 0.88–0.89 | 0.8852 | ±0.005 |
| Epochs | 10–20 | 8 | Slightly longer |

**Rationale:** The initial LR (1e-4) may already be well-suited for this architecture. If the model reaches its capacity at ~89% regardless of LR scheduling, the scheduler simply delays convergence without improving the final result. The fundamental bottleneck may be the Flatten→Dense architecture (29.5M params, most in one layer), which the scheduler cannot fix.

### Scenario C: NEGATIVE (Unlikely — 10%)

| Metric | Predicted | Parent (vR.1.4) | Delta |
|--------|-----------|-----------------|-------|
| Test Accuracy | <88.0% | 88.75% | >−0.75pp |
| Macro F1 | <0.88 | 0.8852 | >−0.005 |

**Rationale:** Extremely unlikely. ReduceLROnPlateau only reduces LR when validation loss plateaus — it cannot make training worse than fixed LR in principle. The only risk is if the scheduler reduces LR too aggressively (patience=3 is moderately aggressive), causing the model to converge prematurely to a worse minimum. However, min_lr=1e-6 provides a safety floor.

---

## 2. Success Criteria

### POSITIVE Verdict

- **Primary:** Macro F1 ≥ 0.8902 (≥ +0.5pp over vR.1.4's 0.8852)
- **OR:** Test accuracy ≥ 89.25% (≥ +0.5pp over vR.1.4's 88.75%)
- **Bonus:** Training stability — no epoch with val_loss > 5.0 (vs vR.1.4's 16.13)

### NEUTRAL Verdict

- All metrics within ±0.5pp of vR.1.4
- Training may be more stable even if metrics don't improve

### NEGATIVE Verdict (would reject)

- Macro F1 < 0.8802 (>0.5pp drop)
- OR Test accuracy < 88.25%

---

## 3. What to Watch For

### Training Dynamics

1. **Epoch 1 val_loss:** Should be lower than vR.1.4's 16.13 (same initial LR, but scheduler will reduce it for epoch 2 if epoch 1 spikes)
2. **Number of LR reductions:** Track how many times the scheduler triggers. Expect 2-4 reductions in a 15-30 epoch run.
3. **Final LR:** Check the LR at the best epoch. If it's still 1e-4, the scheduler didn't contribute.
4. **Training length:** Expect 15-30 epochs vs vR.1.4's 8. Longer training is a positive sign.
5. **Train-val gap:** Should narrow compared to vR.1.4 (which showed rapid overfitting: train_acc=0.93 vs val_acc=0.89)

### Metric Interactions

- **Tp Recall vs Tp Precision tradeoff:** vR.1.4 had best Tp recall (0.9194) but lower Tp precision (0.8240). The scheduler may rebalance this.
- **FP/FN rates:** vR.1.4 had 13.4% FP and 8.1% FN. Watch for changes in this balance.

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LR reduces too fast | Low | NEUTRAL result | patience=3 is standard |
| No effect (model at capacity) | Medium | NEUTRAL result | Architecture changes (vR.1.6-1.7) address this |
| Interference with BN | Very Low | Slight instability | BN and LR scheduler are complementary |
| Longer training time | Medium | +5 min | Acceptable on Kaggle |

---

## 5. Comparison Across Series

| Version | Change | Test Acc | Macro F1 | Epochs | Verdict |
|---------|--------|----------|----------|--------|---------|
| vR.1.1 | Eval fix | 88.38% | 0.8805 | 13 (8) | Honest baseline |
| vR.1.2 | Augmentation | 85.53% | 0.8505 | 6 (1) | REJECTED |
| vR.1.3 | Class weights | 89.17% | 0.8889 | 14 (9) | POSITIVE |
| vR.1.4 | BatchNorm | 88.75% | 0.8852 | 8 (3) | NEUTRAL |
| **vR.1.5** | **LR Scheduler** | **?** | **?** | **?** | **?** |

The trend shows: eval fix → augmentation hurt → class weights helped → BN neutral. The LR scheduler is a low-risk, standard improvement that should complement BN and class weights.

---

## 6. If NEGATIVE — Next Steps

If vR.1.5 is NEGATIVE (unlikely):
1. Investigate whether the scheduler reduced LR too aggressively
2. Try `patience=5` instead of 3 (match early stopping patience)
3. Try `factor=0.2` instead of 0.5 (more gradual reduction)
4. Branch vR.1.6 from vR.1.4 instead (skip scheduler)

If vR.1.5 is POSITIVE or NEUTRAL:
1. Proceed to vR.1.6 (deeper CNN — add 3rd Conv2D layer)
2. The scheduler is kept for all future versions
