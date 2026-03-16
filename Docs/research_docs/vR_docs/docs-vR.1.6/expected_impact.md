# Expected Impact — vR.1.6: Deeper CNN

| Field | Value |
|-------|-------|
| **Version** | vR.1.6 |
| **Parent** | vR.1.5 (LR Scheduler — NEUTRAL, 88.96%) |
| **Change** | Add 3rd Conv2D(64, 3×3, ReLU) + MaxPooling2D(2×2) |
| **Risk Level** | MODERATE |

---

## 1. Predictions

### Scenario A: POSITIVE (Most Likely — 55%)

| Metric | Predicted | Parent (vR.1.5) | Delta |
|--------|-----------|-----------------|-------|
| Test Accuracy | 89.5–91.5% | 88.96% | +0.5–2.5pp |
| Macro F1 | 0.89–0.91 | 0.8873 | +0.003–0.023 |
| ROC-AUC | 0.96–0.97 | 0.9560 | +0.004–0.014 |
| Epochs | 12–25 (best 8–15) | 10 (best 5) | Longer |

**Rationale:** The 3rd conv layer adds meaningful feature extraction and halves the Flatten→Dense parameter count. With fewer parameters to memorize, the model should overfit less, allowing longer training and better generalization. The LR scheduler (retained from vR.1.5) can now operate over more epochs. This is the first architecture change that addresses the root cause of the stalled accuracy.

### Scenario B: NEUTRAL (Possible — 30%)

| Metric | Predicted | Parent (vR.1.5) | Delta |
|--------|-----------|-----------------|-------|
| Test Accuracy | 88.5–89.5% | 88.96% | ±0.5pp |
| Macro F1 | 0.88–0.89 | 0.8873 | ±0.005 |

**Rationale:** The 3rd conv layer may not provide enough additional feature extraction to overcome the accuracy ceiling. The model may still be limited by the ELA signal quality at 128×128, not by architectural depth. The Dense layer still has 13.8M params — still overparameterized, just less so.

### Scenario C: NEGATIVE (Unlikely — 15%)

| Metric | Predicted | Parent (vR.1.5) | Delta |
|--------|-----------|-----------------|-------|
| Test Accuracy | <88.0% | 88.96% | >−1pp |

**Rationale:** The added depth could hurt if the ELA features at 128×128 are too simple for 3 conv layers — the 3rd layer might learn noise. The reduced Flatten size might lose spatial information the Dense layer was using. Also, no BN on the 3rd conv layer (to maintain single-variable ablation) could cause gradient issues at increased depth.

---

## 2. Success Criteria

| Verdict | Condition |
|---------|-----------|
| **POSITIVE** | Macro F1 ≥ 0.8923 (≥ +0.5pp) OR Test Acc ≥ 89.46% (≥ +0.5pp) |
| **NEUTRAL** | All metrics within ±0.5pp of vR.1.5 |
| **NEGATIVE** | Macro F1 < 0.8823 (>0.5pp drop) OR Test Acc < 88.46% |

---

## 3. Why Deeper CNN Should Help

### The Fundamental Problem

The ETASR CNN has been at 88-89% for 5 versions. The bottleneck is not training configuration (class weights, BN, LR scheduler all tried — marginal effect). The bottleneck is **architecture**: only 2 conv layers extracting features, then 29.5M params memorizing them.

### What the 3rd Conv Layer Adds

1. **More expressive features.** Two 5×5 conv layers capture low-level ELA edges and textures. A 3rd layer with 64 filters at 3×3 can learn higher-level combinations — e.g., boundaries between tampered and authentic regions, ELA intensity gradients.

2. **Reduced overfitting.** The Flatten→Dense layer drops from 29.5M to 13.8M params. Fewer parameters = less memorization capacity = better generalization.

3. **Better training dynamics.** With fewer Dense params, the loss landscape should be smoother. The LR scheduler and BN can operate more effectively. Training may extend beyond 10 epochs for the first time since vR.1.3.

4. **Hierarchical feature hierarchy.** Layer 1 (5×5) sees raw ELA pixels. Layer 2 (5×5) sees combinations of layer 1 features. Layer 3 (3×3, at reduced resolution after MaxPool) sees abstract patterns — this is the standard deep learning pyramid.

---

## 4. What to Watch For

1. **Training length:** Should be 12-25 epochs (vs 10). If shorter, the architecture is converging too fast (not gaining from extra depth).
2. **Train-val gap:** Should narrow. vR.1.5 had train_acc=0.95 vs val_acc=0.89 (6pp gap). With 53% fewer Dense params, expect <4pp gap.
3. **Epoch 1 BN spike:** May change in magnitude — different feature map sizes feed into the Dense layer.
4. **LR scheduler triggers:** Expect 2-3 LR reductions (vs 1 in vR.1.5) if training extends.
5. **ROC-AUC:** The critical metric. All 5 prior ablations degraded AUC from the baseline (0.9601). If deeper features improve discrimination, AUC should finally improve.

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 3rd conv layer learns noise | Low | NEGATIVE result | 64 filters at 3×3 is standard design |
| Gradient issues (no BN on layer 3) | Low | Slightly worse training | ReLU + existing BN on layers 1-2 provides stability |
| Information loss from extra MaxPool | Moderate | NEUTRAL result | 29×29×64 still has 53K features — more than sufficient |
| Training too short (early stopping) | Low | NEUTRAL result | LR scheduler provides buffer |

---

## 6. Comparison Across Series

| Version | Change | Test Acc | Macro F1 | Params | Epochs | Verdict |
|---------|--------|----------|----------|--------|--------|---------|
| vR.1.1 | Eval fix | 88.38% | 0.8805 | 29.52M | 13 (8) | Baseline |
| vR.1.2 | Augmentation | 85.53% | 0.8505 | 29.52M | 6 (1) | REJECTED |
| vR.1.3 | Class weights | 89.17% | 0.8889 | 29.52M | 14 (9) | POSITIVE |
| vR.1.4 | BatchNorm | 88.75% | 0.8852 | 29.52M | 8 (3) | NEUTRAL |
| vR.1.5 | LR Scheduler | 88.96% | 0.8873 | 29.52M | 10 (5) | NEUTRAL |
| **vR.1.6** | **Deeper CNN** | **?** | **?** | **13.83M** | **?** | **?** |

vR.1.6 is the first version to change the parameter count. Every prior version had ~29.52M params. The 53% reduction is the most dramatic structural change in the series.
