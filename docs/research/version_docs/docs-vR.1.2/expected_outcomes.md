# Expected Outcomes: vR.1.2 — Data Augmentation

---

## Hypothesis

Adding real-time data augmentation (horizontal flip, vertical flip, rotation ±15°) to the training pipeline will reduce overfitting and improve test-set performance, compensating for the reduced training set in the 70/15/15 split.

---

## Predicted Outcomes

### Optimistic Scenario (augmentation works well)

| Metric | vR.1.1 | Predicted | Change |
|--------|--------|-----------|--------|
| Test Accuracy | 88.38% | 90–91% | +1.5–2.5pp |
| Tampered Recall | 0.8830 | 0.92–0.94 | Significant improvement |
| FN rate | 11.7% | 6–8% | Halved |
| Train-val gap | 4.25pp | 1.5–2.5pp | Substantially reduced |
| Best epoch | 8 | 12–18 | Longer training (good sign) |
| Val collapse | Yes (epochs 12–13) | None | Eliminated |

### Expected Scenario (moderate improvement)

| Metric | vR.1.1 | Predicted | Change |
|--------|--------|-----------|--------|
| Test Accuracy | 88.38% | 89–90% | +0.5–1.5pp |
| Tampered Recall | 0.8830 | 0.89–0.91 | Moderate improvement |
| FN rate | 11.7% | 8–10% | Reduced |
| Train-val gap | 4.25pp | 2.5–3.5pp | Reduced |
| Best epoch | 8 | 10–14 | Slightly longer |
| Val collapse | Yes (epochs 12–13) | Delayed or milder | Improved stability |

### Pessimistic Scenario (augmentation doesn't help much)

| Metric | vR.1.1 | Predicted | Change |
|--------|--------|-----------|--------|
| Test Accuracy | 88.38% | 88–89% | ~Neutral |
| Tampered Recall | 0.8830 | 0.88–0.89 | ~Neutral |
| FN rate | 11.7% | 10–12% | ~Neutral |
| Train-val gap | 4.25pp | 3.5–4.0pp | Slightly reduced |

This would happen if the overfitting is dominated by the Flatten→Dense(256) bottleneck (29.5M params) rather than insufficient data diversity. In this case, architectural changes (vR.1.6/1.7) would be the real fix.

---

## Success Criteria

### POSITIVE verdict (proceed to vR.1.3):
- Test accuracy ≥ 88.88% (≥ +0.5pp from vR.1.1)
- OR Macro F1 ≥ 0.8855 (≥ +0.5pp from vR.1.1)
- OR FN rate ≤ 10% (meaningful improvement)

### NEUTRAL verdict (still proceed, less confident):
- Test accuracy 87.88%–88.88% (within ±0.5pp)
- AND no metric regression > 1pp

### NEGATIVE verdict (investigate before proceeding):
- Test accuracy < 87.88% (> 0.5pp drop)
- OR FN rate > 13% (regression)

---

## Ablation Control

The only variable that changes is:
```
vR.1.1: No augmentation
vR.1.2: ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=15)
```

All other parameters frozen. Any performance difference is attributable to augmentation.

---

## Risks

1. **Rotation fill artifacts** — `fill_mode='nearest'` creates edge artifacts. At 128×128, a 15° rotation affects ~10% of pixels at the border. Mitigation: 'nearest' is better than 'constant' (black fill) which would introduce artificial patterns.

2. **Slower training** — Each batch requires augmentation computation. Expected overhead: 20–30% longer per epoch. Acceptable.

3. **Seed determinism** — `ImageDataGenerator.flow()` with `seed=42` should produce reproducible augmented batches, but there may be minor platform-dependent differences.
