# Expected Outcomes: vR.1.3 — Class Weights

---

## Hypothesis

Adding inverse-frequency class weights will improve the model's sensitivity to the tampered minority class by increasing the loss penalty for tampered misclassifications by ~46%. This should reduce the FP/FN rate imbalance and improve tampered precision/recall.

---

## Predicted Outcomes

### Optimistic Scenario

| Metric | vR.1.1 | Predicted | Change |
|--------|--------|-----------|--------|
| Test Accuracy | 88.38% | 89–90% | +0.5–1.5pp |
| Tampered Precision | 0.8393 | 0.87–0.89 | Significant improvement |
| Tampered Recall | 0.8830 | 0.90–0.93 | Moderate improvement |
| Tampered F1 | 0.8606 | 0.89–0.91 | Significant improvement |
| Macro F1 | 0.8805 | 0.89–0.91 | Improvement |
| FN rate | 11.7% | 7–9% | Substantially reduced |
| FP rate | 11.6% | 12–14% | May slightly increase (tradeoff) |
| ROC-AUC | 0.9601 | 0.96–0.97 | Slight improvement |
| Best epoch | 8 | 8–12 | May shift slightly |

### Expected Scenario

| Metric | vR.1.1 | Predicted | Change |
|--------|--------|-----------|--------|
| Test Accuracy | 88.38% | 88–89% | ~Neutral to slight improvement |
| Tampered Precision | 0.8393 | 0.85–0.87 | Moderate improvement |
| Tampered Recall | 0.8830 | 0.89–0.91 | Slight improvement |
| Tampered F1 | 0.8606 | 0.87–0.89 | Moderate improvement |
| FN rate | 11.7% | 9–11% | Slightly reduced |
| FP rate | 11.6% | 12–13% | Slight tradeoff |

Class weights typically improve minority class metrics at a small cost to majority class metrics. The net effect on accuracy may be neutral if FP and FN tradeoffs cancel out. However, macro F1 should improve because it weights both classes equally.

### Pessimistic Scenario

| Metric | vR.1.1 | Predicted | Change |
|--------|--------|-----------|--------|
| Test Accuracy | 88.38% | 87–88% | Slight drop |
| Tampered Recall | 0.8830 | 0.90+ | Improved |
| Authentic Recall | 0.8843 | 0.85–0.87 | Slight drop (tradeoff) |

If class weights are too aggressive, they may over-compensate: the model predicts "tampered" too aggressively, improving tampered recall but hurting authentic precision/recall. Net accuracy may drop slightly.

**This would still be NEUTRAL or mildly POSITIVE** because macro F1 should still improve (better tampered metrics outweigh small authentic regression when both are weighted equally).

---

## Success Criteria

### POSITIVE verdict (proceed to vR.1.4):
- Macro F1 ≥ 0.8855 (≥ +0.5pp from vR.1.1's 0.8805)
- OR Tampered F1 ≥ 0.8856 (≥ +2.5pp from vR.1.1's 0.8606)

### NEUTRAL verdict (still proceed):
- Macro F1 between 0.8755 and 0.8855 (within ±0.5pp)
- AND no metric regression > 2pp

### NEGATIVE verdict (investigate):
- Macro F1 < 0.8755 (> 0.5pp drop)
- OR test accuracy < 87.38% (> 1pp drop)

---

## Ablation Control

The only variable that changes:
```
vR.1.1: class_weight=None (default)
vR.1.3: class_weight={0: ~0.842, 1: ~1.231}  (computed from training set)
```

All other parameters are frozen at vR.1.1 values. Any performance difference is attributable to class weighting.

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Over-compensation (too aggressive toward tampered) | Low | Medium | `'balanced'` mode is conservative — only 46% weight increase for minority class |
| Training destabilization | Very low | Medium | Class weights don't change the loss surface topology, just rescale gradients |
| No improvement | Medium | Low | This is a neutral-to-positive change; worst case is ±0.5pp |
| Convergence speed change | Low | Low | May need slightly more/fewer epochs; early stopping handles this |

**Overall risk: LOW.** Class weighting is one of the safest ablation changes. It cannot cause the kind of catastrophic failure seen in vR.1.2.

---

## Comparison: vR.1.2 (Rejected) vs vR.1.3 (Current)

| Aspect | vR.1.2 (Augmentation) | vR.1.3 (Class Weights) |
|--------|----------------------|----------------------|
| Change type | Data transformation | Loss reweighting |
| Modifies input data | Yes (random transforms) | No |
| Modifies loss function | No | Yes (per-class scaling) |
| Risk level | High (disruptive to ELA signal) | Low (conservative scaling) |
| Impact on training speed | 20–30% slower (generator) | No change |
| Theoretical basis | Generic CV technique | Directly targets identified imbalance |
| Result | REJECTED (-2.85pp) | Pending |
