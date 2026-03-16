# Technical Audit: vR.P.4 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-4-4-channel-input-rgb-ela-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 25 epochs (all ran, no early stopping), best at epoch 24 |
| **Version** | vR.P.4 — 4-Channel Input (RGB + ELA) |
| **Parent** | vR.P.3 (ELA as input, Pixel F1=0.6920) |
| **Change** | 4-channel input: 3 RGB + 1 ELA grayscale, conv1 unfrozen, dual normalization |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview & Experiment Goal

vR.P.4 tests whether **combining RGB and ELA information** outperforms ELA-only input (P.3). The architecture concatenates 3 RGB channels with 1 ELA grayscale channel into a 4-channel input tensor. This requires:

1. **Modifying conv1** from 3→64 to 4→64 channels (SMP handles this automatically by averaging pretrained RGB weights to initialize the 4th channel)
2. **Unfreezing conv1** so the 4th channel filter can learn ELA-specific patterns
3. **Dual normalization** — ImageNet mean/std for RGB channels 0-2, ELA-specific mean/std for channel 3

The hypothesis: RGB provides natural scene context while ELA provides forensic artifacts. Together they should provide richer features than either alone.

**Result:** Pixel F1 reaches **0.7053** (+1.33pp from P.3), the best absolute pixel-level result in the pretrained series. However, the gain falls below the ±2pp POSITIVE threshold and image accuracy drops from 86.79% to 84.42% (-2.37pp). This is a **NEUTRAL** result — the 4ch approach shows marginal pixel benefit at the cost of image classification quality.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | ResNet-34 (ImageNet, **FROZEN body + conv1 UNFROZEN + BN UNFROZEN**) |
| Input | **4-channel: RGB(3) + ELA grayscale(1)**, 384x384 |
| Normalization | **Dual**: ch 0-2 ImageNet [0.485,0.456,0.406]/[0.229,0.224,0.225], ch 3 ELA [0.0461]/[0.0562] |
| Loss | SoftBCEWithLogitsLoss + DiceLoss (mode=binary, from_logits=True) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5, single param group) |
| Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | patience=7, val_loss |
| Seed | 42 |
| AMP | Enabled (autocast + GradScaler) |
| TF32 | Enabled (no effect on P100) |
| drop_last | True (train only) |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### Freeze Strategy

| Component | Status | Rationale |
|-----------|--------|-----------|
| **conv1** | **UNFROZEN** | Modified from 3→4 channels; 4th channel initialized by averaging RGB weights, needs training |
| bn1 | **UNFROZEN** | BN adaptation to mixed RGB+ELA input |
| layer1-4 conv weights | **FROZEN** | Preserve ImageNet features |
| layer1-4 BatchNorm | **UNFROZEN** | Adapt running statistics to 4ch input distribution |
| Decoder (all) | **TRAINABLE** | Learn to fuse RGB and ELA skip features |
| Segmentation head | **TRAINABLE** | Final 1x1 conv output |

### Parameters

| Category | Count |
|----------|-------|
| Total | 24,439,505 |
| Trainable | 3,181,265 (13.0%) |
| — conv1 | 12,544 |
| — Encoder BN | 17,024 |
| — Decoder | 3,151,552 |
| — Segmentation head | 145 |
| Frozen (encoder conv/fc except conv1) | 21,258,240 |

### 4-Channel Input Pipeline

```
Raw RGB Image (384x384x3)
    |
    +---> RGB: ToTensor → [3, H, W] (range [0, 1])
    |
    +---> ELA: JPEG Q=90 → pixel diff → brightness scale → Grayscale → [1, H, W]
    |
    v
Concatenate → [4, H, W]
    |
    v
Normalize: ch 0-2 (ImageNet) + ch 3 (ELA gray: mean=0.0461, std=0.0562)
    |
    v
ResNet-34 conv1 (4→64, 7x7, stride 2) — UNFROZEN
```

SMP initializes the 4th channel of conv1 by averaging the 3 pretrained RGB filter slices. This gives the ELA channel a reasonable starting point before fine-tuning.

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **Best absolute Pixel F1 in pretrained series: 0.7053** | Marginal improvement (+1.33pp) over P.3 but still the series best |
| S2 | **Best absolute Pixel IoU: 0.5447** | +0.0156 from P.3 (0.5291) |
| S3 | **Best pixel recall in pretrained series: 0.6051** | +0.0146 from P.3 (0.5905) — detects more tampered pixels |
| S4 | **Cleanest execution in P.3-P.6 batch** | All cells ran, model saved (123.5 MB), no errors |
| S5 | **Innovative dual normalization** | Correctly applies ImageNet stats to RGB and ELA-computed stats to channel 4 |
| S6 | **Conv1 unfreeze is minimal and targeted** | Only 12,544 additional params unfrozen — negligible impact on regularization |
| S7 | **Good convergence despite training instability** | Recovered from epoch 10 catastrophic spike (val_loss=0.7968) without intervention |
| S8 | **Per-image F1 distribution improved** | Tampered mean F1=0.5360 (P.4) — best in series |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **MAJOR** | Pixel F1 gain only +1.33pp from P.3 — below ±2pp POSITIVE threshold |
| W2 | **MAJOR** | Image accuracy DROPPED: 84.42% vs P.3's 86.79% (-2.37pp) |
| W3 | **MAJOR** | Image ROC-AUC dropped: 0.9229 vs P.3's 0.9502 (-0.0273) |
| W4 | **MAJOR** | Training instability at epoch 10: val_loss spiked to 0.7968 (+34% from epoch 9), pixel F1 crashed to 0.3567 |
| W5 | MODERATE | FP rate more than doubled: 6.4% (72/1124) vs P.3's 2.7% (30/1124) — RGB adds noise to classification boundary |
| W6 | MODERATE | Best epoch at 24 — model still improving, 25 epochs likely insufficient |
| W7 | MODERATE | Slower convergence than P.3: P.4 reaches F1=0.60 at epoch 12, P.3 at epoch 6 |
| W8 | MINOR | 4ch adds pipeline complexity (dual normalization, conv1 unfreeze) for marginal gain |

---

## 5. Major Issues

### 5.1 Marginal Pixel Improvement Does Not Justify Complexity (W1)

The +1.33pp pixel F1 gain (0.7053 vs 0.6920) falls below the ±2pp threshold for a POSITIVE verdict. Given that P.4 adds significant complexity — 4-channel input pipeline, dual normalization logic, conv1 unfreezing — the marginal pixel benefit does not justify the additional engineering overhead. The simpler P.3 (ELA-only) achieves 98.1% of P.4's pixel F1 with a cleaner pipeline.

### 5.2 Image-Level Regression (W2, W3)

Adding RGB channels paradoxically worsens image-level classification:
- Accuracy: 84.42% vs 86.79% (-2.37pp)
- ROC-AUC: 0.9229 vs 0.9502 (-0.0273)
- FP rate: 6.4% vs 2.7% (+3.7pp)

The likely cause: RGB channels add high-frequency natural image content that the decoder treats as potential tampering artifacts, increasing false positives. The mask-to-classification pipeline (≥100 tampered pixels) amplifies this noise into image-level errors.

### 5.3 Training Instability (W4)

Epoch 10 shows a catastrophic validation spike: val_loss jumps from 0.5953 to 0.7968 (+34%), and pixel F1 drops from 0.5761 to 0.3567 (-38%). The model recovers by epoch 12, but this suggests the 4-channel input creates a more complex loss landscape with sharp minima. The LR=1e-3 may be too aggressive for the unfrozen conv1.

---

## 6. Minor Issues

### 6.1 Pipeline Complexity (W8)

The 4-channel approach requires:
1. Parallel RGB and ELA preprocessing
2. Grayscale conversion for ELA (loses color forensic info)
3. Dual normalization with separate stats per channel group
4. Conv1 unfreezing and initialization

This complexity is justified only if the results clearly surpass the simpler ELA-only approach, which they do not.

### 6.2 Insufficient Training Duration (W6)

Like P.3, the best epoch (24) is near the maximum (25). The early stopping patience of 7 was never exhausted because the model kept finding marginal improvements. More epochs would likely help, but the training instability (W4) suggests the learning rate schedule may need adjustment first.

---

## 7. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR |
|-------|-----------|----------|----------|-----|-----|
| 1 | 0.9936 | 0.8486 | 0.3513 | 0.2131 | 1e-3 |
| 4 | 0.6612 | 0.6646 | 0.5215 | 0.3527 | 1e-3 |
| 8 | 0.5228 | 0.5955 | 0.5831 | 0.4115 | 1e-3 |
| **10** | **0.4789** | **0.7968** | **0.3567** | **0.2171** | **1e-3** |
| 12 | 0.4498 | 0.5569 | 0.6069 | 0.4356 | 1e-3 |
| 17 | 0.3676 | 0.5182 | 0.6359 | 0.4661 | 5e-4 |
| 18 | 0.3464 | 0.5178 | 0.6388 | 0.4693 | 5e-4 |
| **24** (best) | **0.2700** | **0.5090** | **0.6425** | **0.4733** | **2.5e-4** |
| 25 | 0.2687 | 0.5103 | 0.6436 | 0.4745 | 2.5e-4 |

**LR schedule:** 1e-3 (epochs 1-16) → 5e-4 (epochs 17-22) → 2.5e-4 (epochs 23-25)

**Key observations:**
- **Epoch 10 catastrophe:** Val_loss spikes to 0.7968 (+34%), pixel F1 crashes to 0.3567. Full recovery by epoch 12 — suggests a saddle point transition rather than divergence.
- **LR reduction at epoch 17** produced steady improvement: F1 climbed from 0.5975 (epoch 16) to 0.6425 (epoch 24).
- **P.4 converges ~50% slower than P.3:** P.4 reaches F1=0.60 at epoch 12; P.3 reached it at epoch 6. The 4th channel adds learning difficulty.

---

## 8. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.3 | Delta from P.1 |
|--------|-------|----------------|----------------|
| Pixel Precision | 0.8452 | +0.0096 | +0.2117 |
| **Pixel Recall** | **0.6051** | **+0.0146** | **+0.2506** |
| **Pixel F1** | **0.7053** | **+0.0133** | **+0.2507** |
| **Pixel IoU** | **0.5447** | **+0.0156** | **+0.2505** |
| Pixel AUC | 0.9433 | -0.0095 | +0.0924 |

### Image-Level (Classification)

| Metric | Value | Delta from P.3 | Delta from P.1 |
|--------|-------|----------------|----------------|
| **Test Accuracy** | **84.42%** | **-2.37pp** | **+14.27pp** |
| Macro F1 | 0.8322 | -0.0238 | +0.1455 |
| ROC-AUC | 0.9229 | -0.0273 | +0.1444 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.8251 | 0.9359 | 0.8770 | 1,124 |
| Tampered | 0.8835 | 0.7100 | 0.7873 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 1,052 (TN) | 72 (FP) |
| **Tp** | 223 (FN) | 546 (TP) |

- **FP rate: 6.4%** (72/1124) — 2.4x worse than P.3's 2.7%
- **FN rate: 29.0%** (223/769) — similar to P.3's 28.6%
- Net vs P.3: +42 FPs, +3 FNs — RGB channels increase false alarms without reducing misses

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"The authors propose the sensible idea of combining RGB and ELA into a 4-channel input. The dual normalization is correctly implemented, the conv1 initialization is properly handled by SMP, and the experiment is cleanly executed with all cells passing. This is, by far, the best-run notebook in the P.3-P.6 batch.

The problem is that it doesn't really work. The +1.33pp pixel F1 improvement over P.3 is within noise margin. The image accuracy drops 2.37pp. And the FP rate more than doubles. The authors have essentially shown that adding RGB back into an ELA-based pipeline introduces noise that confuses the classifier without meaningfully helping localization.

The epoch 10 training catastrophe is informative but unaddressed. When val_loss spikes 34% in a single epoch and pixel F1 halves, the response should not be 'it recovered, so it's fine.' The response should be to investigate why the 4-channel loss landscape has such sharp gradients and whether a lower initial LR or warmup schedule would help.

The most generous interpretation: P.4 proves that ELA contains most of the forensic signal, and RGB adds marginal spatial context. The least generous: P.4 proves that adding complexity to a winning formula produces diminishing returns and new failure modes."

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction (pixel masks) | **PASS** | Model generates and saves masks correctly |
| Train/val/test split | **PASS** | 70/15/15 stratified, seed=42 |
| Standard metrics (F1, IoU, AUC) | **PASS** | Full pixel-level + image-level suite |
| Visual results (Original/GT/Predicted/Overlay) | **PASS** | All visualization cells executed |
| Model weights (.pth file) | **PASS** | Saved: 123.5 MB |
| Architecture explanation | **PASS** | Detailed pipeline diagram, 4ch explanation |
| Single notebook execution | **PASS** | End-to-end execution with no errors |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 14 | 15 | Innovative 4ch design, proper conv1 handling, dual normalization |
| Dataset | 14 | 15 | Same proper pipeline as P.3, ELA gray stats correctly computed |
| Methodology | 17 | 20 | Clean execution, model saved, but training instability unaddressed (-2), marginal gain (-1) |
| Evaluation | 19 | 20 | Comprehensive metrics, visualizations present, per-image stats computed (-1 for no instability analysis) |
| Documentation | 12 | 15 | Good markdown cells, pipeline diagram, but could better explain epoch 10 anomaly |
| Assignment Alignment | 10 | 15 | All deliverables present, but marginal improvement over P.3 weakens scientific justification |
| **Total** | **86** | **100** | |

---

## 12. Final Verdict: **NEUTRAL** — Score: 86/100

**Pixel F1: 0.7053 (+0.0133 from parent P.3 — below ±2pp threshold)**

vR.P.4 achieves the highest absolute Pixel F1 (0.7053) and IoU (0.5447) in the pretrained series, but the improvement over P.3 is marginal (+1.33pp) and comes at the cost of image-level accuracy (-2.37pp), increased false positives (+42), and greater pipeline complexity. The training instability at epoch 10 suggests the 4-channel loss landscape is more challenging than P.3's 3-channel ELA.

**High score (86) despite NEUTRAL verdict** because execution quality is excellent: all cells pass, model is saved, documentation is thorough, and the experiment is scientifically well-motivated even though the result is inconclusive.

### Key Insight

**RGB adds noise, not signal.** P.3's ELA-only input captures the forensic artifacts that matter for localization. Adding RGB channels provides marginal pixel improvement (+1.33pp) while degrading classification (-2.37pp) and increasing FP rate (2.7% → 6.4%). The simpler P.3 approach is preferred.

### Recommended Next Steps

1. **Branch future experiments from P.3, not P.4** — ELA-only is the better foundation
2. **If revisiting 4ch:** try lower initial LR (5e-4) or LR warmup to address training instability
3. **Alternative fusion:** Instead of early fusion (4ch input), try late fusion (two encoders → fused decoder) — but this doubles compute
