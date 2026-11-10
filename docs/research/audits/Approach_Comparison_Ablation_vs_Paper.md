# Approach Comparison: Ablation Study vs Research Paper Architecture

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Scope** | Strategic comparison of all experimental approaches to guide future direction |
| **Runs Covered** | ETASR (vR.1.0--1.7), Pretrained (vR.P.0--P.14), Standalone (3 runs) |

---

## 1. Approach Summary

### Approach A: ETASR Classification Track (vR.1.x)

- **Architecture:** Paper-faithful CNN (2--3 conv blocks + Dense classifier)
- **Task:** Binary classification (authentic vs tampered)
- **Best result:** vR.1.6 — 90.23% accuracy, 0.9004 Macro F1, 0.9657 ROC-AUC
- **Limitation:** Cannot produce pixel-level localization masks

### Approach B: Pretrained Localization Track (vR.P.x)

- **Architecture:** UNet + ResNet-34 encoder (ImageNet pretrained, frozen body + BN unfrozen)
- **Task:** Pixel-level semantic segmentation (localization masks)
- **Best result:** vR.P.10 — 0.7277 Pixel F1, 0.5719 IoU, 0.9573 Pixel AUC, 87.32% Image Acc
- **Strength:** Full localization pipeline; satisfies assignment requirements

### Approach C: Standalone Paper Architecture Runs

- **Architecture:** Paper's exact CNN (or deeper variant)
- **Best result:** Deeper CNN — 90.76% accuracy (classification only)
- **Limitation:** No localization; Sagnik dataset run INVALID due to data leak (99.95% accuracy)

---

## 2. Head-to-Head Comparison

| Dimension | ETASR (vR.1.x) | Pretrained (vR.P.x) | Standalone | Winner |
|-----------|-----------------|---------------------|------------|--------|
| **Image Accuracy** | **90.23%** (vR.1.6) | 88.48% (P.12) | **90.76%** (Deeper) | Standalone |
| **Image Macro F1** | **0.9004** (vR.1.6) | 0.8756 (P.12) | 0.9082 (Deeper) | Standalone |
| **Image ROC-AUC** | **0.9657** (vR.1.6) | 0.9633 (P.10) | N/A | ETASR (barely) |
| **Pixel F1** | N/A | **0.7277** (P.10) | N/A | Pretrained |
| **Pixel IoU** | N/A | **0.5719** (P.10) | N/A | Pretrained |
| **Pixel AUC** | N/A | **0.9573** (P.10) | N/A | Pretrained |
| **Localization masks** | No | **Yes** | No | Pretrained |
| **Assignment alignment** | Partial (5/10) | **Full (14/15)** | None (2/10) | Pretrained |
| **Training stability** | Stable | Mostly stable (P.12 had spikes) | Stable (or overfit) | Tie |
| **Reproducibility** | Not tested | **Perfect** (P.10 Run-01 = Run-02) | Not tested | Pretrained |
| **Param efficiency** | 64K (vR.1.7) | 3.17M (P.3) | 24--38M | ETASR |

---

## 3. New Run Analysis (This Audit Cycle)

### Pretrained Track Updates

| Run | Pixel F1 | Delta from P.3 | Verdict | Key Finding |
|-----|----------|-----------------|---------|-------------|
| **vR.P.7** (50ep extended) | 0.7154 | +2.34pp | POSITIVE | Confirmed P.3 undertrained; best epoch 36/50 |
| **vR.P.10 Run-02** | 0.7277 | +3.57pp | POSITIVE | Perfect reproducibility (bit-identical to Run-01) |
| **vR.P.12** (augmentation) | 0.6968 | +0.48pp | NEUTRAL | Marginal gain; training instability (2 val loss spikes) |
| **vR.P.14** (TTA) | 0.6388 | -5.32pp | NEGATIVE | TTA at threshold=0.5 hurts; code bug crashed 40% of evaluation |

### Standalone Updates

| Run | Accuracy | Verdict | Key Finding |
|-----|----------|---------|-------------|
| **ELA-CNN-Forgery-sagnik** | 99.95% | **INVALID** | Data leak (probable mask-as-input contamination) |

---

## 4. Updated Pretrained Leaderboard

| Rank | Version | Change | Pixel F1 | IoU | Verdict |
|------|---------|--------|----------|-----|---------|
| 1 | **vR.P.10** | CBAM attention + Focal+Dice | **0.7277** | **0.5719** | **Series leader** |
| 2 | vR.P.7 | Extended training (50ep) | 0.7154 | 0.5569 | POSITIVE |
| 3 | vR.P.4 | 4ch RGB+ELA | 0.7053 | 0.5447 | NEUTRAL |
| 4 | vR.P.8 | Progressive unfreeze | 0.6985 | 0.5367 | NEUTRAL |
| 5 | **vR.P.12** | Augmentation + Focal+Dice | **0.6968** | **0.5347** | **NEUTRAL (NEW)** |
| 6 | vR.P.9 | Focal+Dice loss | 0.6923 | 0.5294 | NEUTRAL |
| 7 | vR.P.3 | ELA input (baseline) | 0.6920 | 0.5291 | STRONG POSITIVE |
| 8 | **vR.P.14** | TTA (4 views) | **0.6388** | **0.4693** | **NEGATIVE (NEW)** |

---

## 5. Impact Hierarchy (Updated)

```
1. INPUT REPRESENTATION                          >>> +23.7pp Pixel F1
   ELA input (P.3) was the single biggest breakthrough

2. ATTENTION MECHANISM                            +3.57pp Pixel F1
   CBAM in decoder (P.10) — best architectural improvement

3. TRAINING BUDGET                                +2.34pp Pixel F1
   Extended training 50ep (P.7) — simple but effective

4. ARCHITECTURE (Encoder)                         +5.9--6.7pp Pixel F1 (from P.1)
   ResNet-50/EfficientNet-B0 vs baseline, but diminishing on ELA

5. DATA AUGMENTATION                              +0.48pp Pixel F1
   Albumentations (P.12) — marginal, introduces instability

6. LOSS FUNCTION                                  +0.03pp Pixel F1
   Focal+Dice (P.9) — negligible effect, AUC regressed

7. POST-PROCESSING (TTA)                          -5.32pp Pixel F1
   TTA at threshold=0.5 (P.14) — HARMFUL without threshold recalibration

8. PROGRESSIVE UNFREEZE                           +0.65pp Pixel F1
   3-stage unfreeze (P.8) — marginal, Stage 1 counterproductive
```

---

## 6. Decision: Should We Switch Approach?

### Recommendation: **NO — Continue the Pretrained Localization Track**

**Reasons:**

1. **Assignment requires localization.** The ETASR and standalone tracks produce classification only. No amount of accuracy improvement satisfies the pixel-level prediction requirement.

2. **Pretrained track is the only viable path.** vR.P.10 (Pixel F1=0.7277, IoU=0.5719) is the only approach that produces localization masks, GT mask comparison, and visual overlays.

3. **Pretrained track has untapped headroom:**
   - P.10 + extended training (50ep) has not been tried -> estimated 0.74--0.76 Pixel F1
   - Multi-quality ELA (P.15) as an input representation experiment could unlock another +2--5pp
   - P.13 (combined CBAM + augmentation + extended training) tests the combined best-of hypothesis

4. **Standalone paper architecture is a dead end.** Best classification accuracy (90.76%) cannot be improved on without fundamentally changing the architecture, and it cannot localize.

5. **Sagnik dataset experiments are INVALID.** Data leak confirmed; no further experiments on this dataset without decontamination.

### What to Deprecate

| Track | Action | Reason |
|-------|--------|--------|
| ETASR (vR.1.x) | **Archive** | Classification ceiling reached; no localization |
| Standalone CNN | **Archive** | Classification only; paper gap not closeable |
| Sagnik dataset | **Abandon** | Confirmed data leak |
| TTA (naive) | **Abandon** | Harmful at threshold=0.5; would need threshold optimization to revisit |

### What to Continue

| Track | Action | Next Steps |
|-------|--------|------------|
| Pretrained (vR.P.x) | **Continue** | P.13 (combined best), P.15 (multi-Q ELA), P.10+extended |

---

## 7. Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| P.13 overfits (too many tricks) | Medium | Compare each component's contribution; ablate if needed |
| Multi-Q ELA (P.15) degrades signal | Low-Medium | Each quality captures complementary artifacts; grayscale preserves signal |
| Training instability (from P.12 data) | Medium | Monitor val loss; reduce augmentation strength if spikes recur |
| Pixel F1 plateau at ~0.73 | High | May need to explore encoder-level changes (EfficientNet + ELA) or higher resolution |
| Assignment deadline pressure | Medium | P.10 is already submission-ready; new experiments are incremental improvements |

---

## 8. Bottom Line

**vR.P.10 is the submission candidate.** It produces localization masks, achieves 0.7277 Pixel F1, and has been reproduced. All future experiments aim to improve upon P.10, but P.10 alone satisfies the assignment requirements.

The ETASR track served its purpose as a methodological training ground. The standalone paper reproduction confirmed the published accuracy is not achievable (3.81pp gap). Both tracks are now archived, and all resources should focus on the pretrained localization pipeline.
