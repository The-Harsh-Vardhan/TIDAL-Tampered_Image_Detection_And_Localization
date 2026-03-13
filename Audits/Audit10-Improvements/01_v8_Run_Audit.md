# v8 Run Audit

**Notebook:** `v8-tampered-image-detection-localization-run-01.ipynb`
**Environment:** Kaggle, 2x Tesla T4 (15.6 GB VRAM each)
**Epochs:** 27 (early stopped from 50, patience=10)

---

## Final Test Metrics

| Metric | Value |
|--------|-------|
| Image-Level Accuracy | 0.7190 |
| Image-Level AUC-ROC | **0.8170** |
| Tampered-Only Pixel-F1 | 0.2949 +/- 0.3450 |
| Tampered-Only Pixel-IoU | 0.2321 +/- 0.2956 |
| Mixed-Set Pixel-F1 | 0.5181 |
| Mixed-Set Pixel-IoU | 0.4926 |
| Mixed-Set Precision | 0.5230 |
| Mixed-Set Recall | 0.7579 |
| Optimal Threshold | 0.7500 |

### Forgery-Type Breakdown

| Type | Count | F1 |
|------|-------|----|
| Splicing | 274 | 0.5758 |
| Copy-move | 495 | 0.1394 |

### Mask-Size Stratified Results

| Bucket | F1 |
|--------|----|
| Tiny (<2%) | 0.1432 |
| Small (2-5%) | 0.2429 |
| Medium (5-15%) | 0.4057 |
| Large (>15%) | 0.5573 |

---

## Training Observations

- Best model at epoch 17 (val F1=0.3585)
- Training loss steadily decreased (2.24 → ~1.45)
- **Val loss increased** (2.16 → 2.26) — overfitting signal
- Val F1 highly volatile (oscillating 0.08–0.36)
- ReduceLROnPlateau triggered 4 LR reductions
- Gradient norms grew from ~2.07 to ~4.0

---

## Strengths

1. **SMP U-Net with ResNet34** pretrained on ImageNet (24.4M params)
2. **BCE pos_weight=30.01** computed from mask pixel ratio — addresses class imbalance at pixel level
3. **Differential learning rates**: encoder=1e-4, decoder=1e-3
4. **Gradient accumulation** (4 steps, effective batch=256)
5. **ReduceLROnPlateau** monitoring val F1 (patience=3, factor=0.5)
6. **Expanded threshold sweep** (0.05–0.80, 15 candidates)
7. **Mask-size stratified evaluation** — reveals failure modes
8. **Forgery-type breakdown** — identifies copy-move weakness
9. **Grad-CAM explainability** on encoder layer4
10. **Failure case analysis** with mask-size annotation
11. **Shortcut learning checks**: mask randomization test (F1=0.077, passes) + boundary sensitivity
12. **Robustness testing**: 8 conditions (JPEG QF50/70, noise light/heavy, blur, resize 0.5x/0.75x)
13. **Comprehensive W&B integration** with artifacts
14. **Data leakage verification** (zero overlap assertion)
15. **Full artifact inventory** with file presence verification
16. **DataParallel** across 2 GPUs
17. **Detailed changelog** from v6.5 to v8

---

## Weaknesses

1. **Copy-move detection essentially non-functional** (F1=0.1394)
2. **Tampered-only F1 is mediocre** (0.2949) — high variance (std=0.345)
3. **Threshold pushed to 0.75** — notebook itself flags pos_weight may be too aggressive
4. **Val loss diverges** while train loss drops — classic overfitting
5. **Val F1 highly unstable** — model hasn't converged to stable localization
6. **Gaussian noise robustness** still poor (13% drop)
7. **Tiny mask performance** near random (F1=0.14)

---

## Verdict

The most scientifically rigorous notebook. Excellent evaluation methodology with robustness testing, explainability, stratified analysis, and shortcut checks. The engineering is top-tier. However, the actual localization performance is disappointing — the model fundamentally struggles with tampered-only pixel prediction, especially for copy-move and small regions. Classification (AUC=0.817) is solid.

**Score: 82/100** — Best evaluation methodology, mediocre localization results.
