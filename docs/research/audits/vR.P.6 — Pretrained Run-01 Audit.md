# Technical Audit: vR.P.6 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-6-efficientnet-b0-encoder-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 23 epochs (early stopped), best at epoch 16 |
| **Version** | vR.P.6 — EfficientNet-B0 Encoder |
| **Parent** | vR.P.1 (ResNet-34, RGB, frozen encoder, Pixel F1=0.4546) |
| **Change** | Replace ResNet-34 encoder with EfficientNet-B0 (frozen) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview & Experiment Goal

vR.P.6 tests whether a **different encoder architecture family** improves localization. EfficientNet-B0 replaces ResNet-34, bringing:
- **MBConv (Mobile Inverted Bottleneck)** blocks instead of BasicBlocks
- **Squeeze-and-Excite (SE) attention** per block — channel-wise attention
- **Compound scaling** — optimized depth/width/resolution ratios
- **Higher ImageNet accuracy** (77.1% vs 73.3%) with **fewer parameters** (5.3M vs 21.8M)
- **Narrower skip connections**: [16, 24, 40, 112, 320] vs [64, 64, 128, 256, 512]

The parent is vR.P.1 (not P.1.5) because P.6 does **not** include P.1.5's speed optimizations (no AMP, no TF32, no drop_last). This is a methodology inconsistency with P.5, which branches from P.1.5 and includes all speed opts.

**Result:** Pixel F1 reaches **0.5217** (+6.71pp from P.1's 0.4546), exceeding the +2pp POSITIVE threshold. The smaller encoder produces a smaller decoder (~2.24M vs 3.15M), resulting in the most parameter-efficient model in the series (6.25M total, 2.24M trainable). However, image accuracy is the lowest in the P.3-P.6 batch at 70.68%.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | **EfficientNet-B0** (ImageNet, **FULLY FROZEN**) |
| Input | RGB 384x384, ImageNet normalization |
| Loss | SoftBCEWithLogitsLoss + DiceLoss (mode=binary, from_logits=True) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5, decoder only) |
| Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | patience=7, val_loss |
| Seed | 42 |
| AMP | **NOT ENABLED** |
| TF32 | **NOT ENABLED** |
| num_workers | **0** (main process only) |
| drop_last | **NOT ENABLED** (defaults to False) |
| pin_memory | True |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### EfficientNet-B0 Architecture

| Block | Type | Output Shape | Channels | Status |
|-------|------|-------------|----------|--------|
| stem | 3x3, stride 2 | 192x192 | 32 | FROZEN |
| block1 | 1× MBConv1, k3x3 | 192x192 | **16** | FROZEN (skip 1) |
| block2 | 2× MBConv6, k3x3 | 96x96 | **24** | FROZEN (skip 2) |
| block3 | 2× MBConv6, k5x5 | 48x48 | **40** | FROZEN (skip 3) |
| block4 | 3× MBConv6, k3x3 | 24x24 | 80 | FROZEN |
| block5 | 3× MBConv6, k5x5 | 24x24 | **112** | FROZEN (skip 4) |
| block6 | 4× MBConv6, k5x5 | 12x12 | 192 | FROZEN |
| block7 | 1× MBConv6, k3x3 | 12x12 | **320** | FROZEN (skip 5) |

### Skip Connection Comparison

| Skip Level | ResNet-34 | EfficientNet-B0 | Ratio |
|------------|-----------|-----------------|-------|
| Skip 1 | 64 | 16 | 0.25x |
| Skip 2 | 64 | 24 | 0.38x |
| Skip 3 | 128 | 40 | 0.31x |
| Skip 4 | 256 | 112 | 0.44x |
| Skip 5 | 512 | 320 | 0.63x |

### Parameters

| Category | Count |
|----------|-------|
| Total | 6,251,469 |
| Trainable (decoder only) | 2,243,921 (35.9%) |
| Frozen (encoder) | 4,007,548 |
| Data:param ratio | 1:254 |

**Key advantage:** P.6 has the best data:param ratio in the series (1:254) — nearly 4x better than P.5 (1:1,021) and better than P.3 (1:359).

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **+6.71pp Pixel F1 over parent** | 0.5217 vs P.1's 0.4546 — clear POSITIVE |
| S2 | **Best data:param ratio in series: 1:254** | 2.24M trainable on 8,829 images — good balance |
| S3 | **Smallest total model: 6.25M params, 43.3 MB saved** | 3.9x smaller than ResNet-34 (24.4M), 5.2x smaller than ResNet-50 (32.5M) |
| S4 | **SE attention provides channel-wise feature recalibration** | Built into the encoder — no additional parameters needed |
| S5 | **Higher ImageNet accuracy from fewer params** | 77.1% top-1 (EffNet-B0) vs 73.3% (ResNet-34) with 4x fewer encoder params |
| S6 | **Clean early stopping** | Stopped at epoch 23, best at 16 — proper convergence, not hitting epoch ceiling |
| S7 | **Pixel precision highest among RGB models: 0.7034** | Higher than P.1 (0.6335), P.1.5 (0.6364), P.5 (0.6089) |
| S8 | **Clean execution** | All cells pass, model saved, no errors |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **MAJOR** | No AMP/TF32 — methodology inconsistent with P.5 (which has both) |
| W2 | **MAJOR** | Image accuracy 70.68% — worst in P.3-P.6 batch, barely above P.1 baseline (70.15%) |
| W3 | **MAJOR** | FN rate 37.2% (286/769) — worst in P.3-P.6 batch, misses over 1 in 3 tampered images |
| W4 | **MAJOR** | Pixel recall only 0.4146 — 30% lower than P.3 (0.5905), conservative to a fault |
| W5 | MODERATE | num_workers=0 — slower training vs P.5's num_workers=2 |
| W6 | MODERATE | FP rate 23.9% (269/1124) — second worst after P.5 (27.4%) |
| W7 | MODERATE | Growing overfitting: train_loss=0.42 vs val_loss=0.82 at epoch 23 (ratio 1.95x) |
| W8 | MINOR | Narrower skip channels may limit decoder's ability to reconstruct fine spatial details |

---

## 5. Major Issues

### 5.1 MAJOR: Methodology Inconsistency (W1)

P.6 omits AMP, TF32, drop_last, and uses num_workers=0, while P.5 (its sister encoder-swap experiment) includes all of these from P.1.5. The notebook states "Everything else is identical to vR.P.1" — meaning it branches from P.1, not P.1.5.

This creates a confounding variable: P.5 vs P.6 performance differences could be attributed to the speed optimizations as well as the encoder swap. A rigorous comparison would require both experiments to use the same infrastructure settings.

**Impact on results:** AMP typically introduces small numerical differences (~0.1-0.3pp on metrics). The more significant confound is drop_last=True (which P.5 uses and P.6 does not), as this changes the effective number of training batches.

### 5.2 MAJOR: Poor Image-Level Performance (W2, W3)

Image accuracy (70.68%) is barely above the P.1 baseline (70.15%) and far below P.3 (86.79%). The FN rate (37.2%) is the worst in the batch — over a third of tampered images are classified as authentic. This suggests EfficientNet-B0's frozen features, while good for ImageNet classification, do not produce mask predictions that cross the ≥100 pixel threshold reliably.

### 5.3 MAJOR: Low Pixel Recall (W4)

Pixel recall (0.4146) means the model detects less than half of all tampered pixels. While precision is decent (0.7034), the model is excessively conservative — it activates small, high-confidence regions while missing large tampered areas. The narrow skip channels (16-320 vs 64-512 for ResNet-34) may contribute: fewer channels carry less spatial information to the decoder.

---

## 6. Minor Issues

### 6.1 Narrow Skip Channels (W8)

EfficientNet-B0's skip connections are 2-4x narrower than ResNet-34's at every level. While this produces a smaller decoder (2.24M vs 3.15M), it limits the spatial information available for mask reconstruction. The decoder receives coarser features at each resolution, which may explain the lower recall.

### 6.2 num_workers=0 (W5)

Using the main process for data loading (num_workers=0) means GPU idles during data preprocessing. While this avoids Kaggle multiprocessing bugs, it slows training. P.5 demonstrates that num_workers=2 works on Kaggle with persistent_workers=True.

---

## 7. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR |
|-------|-----------|----------|----------|-----|-----|
| 1 | 1.0494 | 0.9812 | 0.2403 | 0.1366 | 1e-3 |
| 3 | 0.9221 | 0.9054 | 0.3285 | 0.1966 | 1e-3 |
| 5 | 0.8503 | 0.8635 | 0.3600 | 0.2195 | 1e-3 |
| 10 | 0.7270 | 0.8168 | 0.4176 | 0.2639 | 1e-3 |
| 15 | 0.5748 | 0.8056 | 0.4252 | 0.2700 | 5e-4 |
| **16** (best) | **0.5597** | **0.7882** | **0.4551** | **0.2946** | **5e-4** |
| 20 | 0.4920 | 0.7888 | 0.4481 | 0.2888 | 5e-4 |
| 23 (final) | 0.4166 | 0.8174 | 0.4370 | 0.2796 | 2.5e-4 |

**LR schedule:** 1e-3 (epochs 1-14) → 5e-4 (epochs 15-20) → 2.5e-4 (epochs 21-23)

**Key observations:**
- **Slowest convergence in P.3-P.6:** Pixel F1 reaches 0.40 only at epoch 10 (P.3 reached it at epoch 1)
- **Proper early stopping:** Best at epoch 16, stopped at 23 (patience=7 exhausted). Unlike P.3-P.5, P.6 shows clean convergence behavior.
- **Val loss plateaus above 0.78:** Never drops below P.1's level (0.83), suggesting the frozen EfficientNet features have a lower ceiling than frozen ResNet-34 features for this task.
- **Post-LR-reduction improvement:** First reduction (epoch 15) produces the best epoch (16) — standard LR scheduler behavior.

---

## 8. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.1 |
|--------|-------|----------------|
| **Pixel Precision** | **0.7034** | **+0.0699** |
| Pixel Recall | 0.4146 | +0.0601 |
| **Pixel F1** | **0.5217** | **+0.0671** |
| **Pixel IoU** | **0.3529** | **+0.0587** |
| **Pixel AUC** | **0.8708** | **+0.0199** |

### Image-Level (Classification)

| Metric | Value | Delta from P.1 |
|--------|-------|----------------|
| Test Accuracy | 70.68% | +0.53pp |
| Macro F1 | 0.6950 | +0.0083 |
| ROC-AUC | 0.7801 | +0.0016 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.7493 | 0.7607 | 0.7550 | 1,124 |
| Tampered | 0.6423 | 0.6281 | 0.6351 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 855 (TN) | 269 (FP) |
| **Tp** | 286 (FN) | 483 (TP) |

- **FP rate: 23.9%** (269/1124) — much higher than P.3's 2.7%
- **FN rate: 37.2%** (286/769) — worst in P.3-P.6 batch
- Net vs P.1: +15 more FPs recovered as TN, but 25 fewer FNs recovered as TP — marginal image-level improvement

### Cross-Encoder Comparison (RGB-only, frozen encoder)

| Metric | P.1 (R34) | P.1.5 (R34+speed) | P.5 (R50) | **P.6 (EffNet-B0)** |
|--------|-----------|--------------------|-----------|-----------------------|
| Pixel F1 | 0.4546 | 0.4227 | 0.5137 | **0.5217** |
| Pixel IoU | 0.2942 | 0.2680 | 0.3456 | **0.3529** |
| Image Acc | 70.15% | 71.05% | 72.00% | 70.68% |
| Trainable | 3.15M | 3.15M | 9.01M | **2.24M** |
| Data:param | 1:357 | 1:357 | 1:1,021 | **1:254** |

P.6 achieves the best pixel F1 among RGB-only models with the fewest trainable parameters, demonstrating EfficientNet-B0's feature quality.

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"The authors present an encoder swap experiment: EfficientNet-B0 replaces ResNet-34. Fair enough. The result is a +6.7pp pixel F1 improvement with the smallest model in the series. On paper, this is a clean win for efficient architectures.

But then the details emerge. AMP is missing. TF32 is missing. drop_last is missing. num_workers is 0. This notebook is running in what can only be described as 'economy class' while its sister experiment P.5 flies business. The methodology inconsistency means we can't cleanly compare P.5 and P.6: is EfficientNet-B0 better than ResNet-50, or does P.6 just benefit from a different training dynamic?

The FN rate of 37.2% is alarming. Over a third of tampered images escape detection entirely. At this rate, a forger's best strategy is to submit their work and hope — they've got a one-in-three chance of getting away with it.

The charitable view: EfficientNet-B0 extracts good features with 4x fewer encoder parameters, and the narrow skip channels create a naturally regularized decoder. The uncharitable view: we've learned the same lesson as P.5 — that RGB input is the bottleneck, not the encoder. P.6's 0.5217 is still 25% below P.3's 0.6920. Swap the encoder all you want; if you don't swap the input, you're rearranging deck chairs."

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction (pixel masks) | **PASS** | Model generates masks, saved (43.3 MB) |
| Train/val/test split | **PASS** | 70/15/15 stratified, seed=42 |
| Standard metrics (F1, IoU, AUC) | **PASS** | Full pixel-level + image-level suite |
| Visual results (Original/GT/Predicted/Overlay) | **PASS** | All visualization cells executed |
| Model weights (.pth file) | **PASS** | Saved: vR.P.6_unet_efficientnet_b0_model.pth (43.3 MB) |
| Architecture explanation | **PASS** | MBConv blocks, SE attention, skip channels documented |
| Single notebook execution | **PASS** | End-to-end, early stopping at epoch 23 |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 13 | 15 | Good encoder choice, SE attention, best param efficiency. Narrow skips may limit performance. |
| Dataset | 14 | 15 | Proper GT masks, standard pipeline, consistent with series |
| Methodology | 13 | 20 | No AMP/TF32 creates methodology inconsistency with P.5 (-4), num_workers=0 (-1), proper early stopping (+1) |
| Evaluation | 18 | 20 | Comprehensive metrics, all visualizations present, cross-encoder comparison in notebook |
| Documentation | 12 | 15 | Good architecture documentation, MBConv explained, but methodology differences not acknowledged |
| Assignment Alignment | 8 | 15 | All deliverables present, but low image accuracy (70.68%) and high FN rate weaken practical utility |
| **Total** | **78** | **100** | |

---

## 12. Final Verdict: **POSITIVE** — Score: 78/100

**Pixel F1: 0.5217 (+0.0671 from parent P.1 — exceeds +2pp POSITIVE threshold)**

vR.P.6 demonstrates that EfficientNet-B0 produces better frozen features than ResNet-34 for forensic localization, achieving the best pixel F1 among RGB-only models (0.5217) with the smallest model footprint (6.25M total, 2.24M trainable, 43.3 MB saved). The SE attention mechanism and compound scaling provide richer features from fewer parameters.

However, the methodology inconsistency (no AMP/TF32) compared to P.5 complicates cross-encoder comparisons, and the absolute performance remains far below P.3's ELA-based approach (0.6920). At 70.68% image accuracy and 37.2% FN rate, the model's practical utility for assignment submission is limited.

### Key Insight: Input > Encoder (Confirmed)

| Experiment | Input | Encoder | Trainable | Pixel F1 |
|-----------|-------|---------|-----------|----------|
| P.1 | RGB | ResNet-34 | 3.15M | 0.4546 |
| P.5 | RGB | ResNet-50 | 9.01M | 0.5137 |
| **P.6** | **RGB** | **EffNet-B0** | **2.24M** | **0.5217** |
| **P.3** | **ELA** | **ResNet-34** | **3.17M** | **0.6920** |

The best encoder swap (P.6, +0.0671) provides less than half the improvement of the input swap (P.3, +0.2374). Future work should prioritize input representation over encoder architecture.

### Recommended Next Steps

1. **Try EfficientNet-B0 with ELA input** to combine the best encoder efficiency with the best input modality
2. **Standardize methodology** — all experiments should use AMP/TF32/drop_last consistently
3. **Consider EfficientNet-B0 with BN unfreeze** — P.3 showed BN adaptation is critical for domain shift
