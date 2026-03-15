# Technical Audit: vR.P.5 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-5-resnet-50-encoder-test-deeper-features-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 25 epochs (all ran, no early stopping), best at epoch 19 |
| **Version** | vR.P.5 — ResNet-50 Encoder (Test Deeper Features) |
| **Parent** | vR.P.1.5 (ResNet-34, RGB, frozen, speed-optimized, Pixel F1=0.4227) |
| **Change** | Swap encoder from ResNet-34 to ResNet-50 (frozen) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS (copy-paste bugs in output strings)** |

---

## 1. Notebook Overview & Experiment Goal

vR.P.5 tests whether a **deeper encoder architecture** improves localization with frozen pretrained features. ResNet-50 replaces ResNet-34, bringing:
- **Bottleneck blocks** (1x1→3x3→1x1) instead of BasicBlocks (3x3→3x3)
- **4x wider skip connections** at each resolution level: [256, 512, 1024, 2048] vs [64, 128, 256, 512]
- A **3x larger decoder** (~9M vs ~3.15M trainable params) to handle wider skip inputs

The parent is vR.P.1.5 (not P.1) because P.1.5 includes AMP/TF32 speed optimizations that P.5 also uses. The encoder swap is the only change — input (RGB), normalization (ImageNet), loss, optimizer, and evaluation are identical.

**Result:** Pixel F1 reaches **0.5137** (+9.1pp from P.1.5's 0.4227), exceeding the +2pp POSITIVE threshold. However, absolute performance remains far below P.3's ELA-based result (0.6920), demonstrating that **encoder depth matters less than input representation** for this task.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | **ResNet-50** (ImageNet, **FULLY FROZEN**) |
| Input | RGB 384x384, ImageNet normalization |
| Loss | SoftBCEWithLogitsLoss + DiceLoss (mode=binary, from_logits=True) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5, decoder only) |
| Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | 25 |
| Early stopping | patience=7, val_loss |
| Seed | 42 |
| AMP | Enabled (autocast + GradScaler) |
| TF32 | Enabled (no effect on P100) |
| num_workers | 2, persistent_workers=True |
| drop_last | True (train only) |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### ResNet-50 Architecture

| Block | Type | Output Shape | Channels | Status |
|-------|------|-------------|----------|--------|
| conv1 | 7x7, stride 2 | 192x192 | 64 | FROZEN |
| pool | 3x3, stride 2 | 96x96 | 64 | FROZEN |
| layer1 | 3× Bottleneck(64→256) | 96x96 | **256** | FROZEN |
| layer2 | 4× Bottleneck(128→512) | 48x48 | **512** | FROZEN |
| layer3 | 6× Bottleneck(256→1024) | 24x24 | **1,024** | FROZEN |
| layer4 | 3× Bottleneck(512→2048) | 12x12 | **2,048** | FROZEN |

### Skip Connection Comparison

| Skip Level | ResNet-34 | ResNet-50 | Ratio |
|------------|-----------|-----------|-------|
| Skip 0 | 64 | 64 | 1x |
| Skip 1 | 64 | 256 | 4x |
| Skip 2 | 128 | 512 | 4x |
| Skip 3 | 256 | 1,024 | 4x |
| Skip 4 | 512 | 2,048 | 4x |

### Parameters

| Category | Count |
|----------|-------|
| Total | 32,521,105 |
| Trainable (decoder only) | 9,013,073 (27.7%) |
| Frozen (encoder) | 23,508,032 |
| Data:param ratio | 1:1,021 |

**Key observation:** The decoder is **2.86x larger** than ResNet-34's decoder (9.0M vs 3.15M) because wider skip connections require larger conv blocks in the decoder to process 4x more channels at each resolution.

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **+9.1pp Pixel F1 over parent** | 0.5137 vs P.1.5's 0.4227 — clear POSITIVE by ±2pp threshold |
| S2 | **Pixel IoU improved +7.8pp** | 0.3456 vs P.1.5's 0.2680 |
| S3 | **Pixel AUC improved +2.7pp** | 0.8828 vs P.1.5's 0.8560 |
| S4 | **Best tampered recall (image-level) among RGB models** | 0.7113 — higher than P.1 (0.5956) and P.1.5 (0.6619) |
| S5 | **ImageNet top-1 accuracy is higher** | ResNet-50: 76.1% vs ResNet-34: 73.3% — richer features |
| S6 | **Clean execution** | All cells pass, model saved (202.6 MB) |
| S7 | **Proper methodology** | AMP, TF32, persistent workers, drop_last — same as P.1.5 |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **CRITICAL** | Copy-paste bug: model saved as `vR.P.5_unet_resnet34_model.pth` instead of `resnet50` |
| W2 | **CRITICAL** | Copy-paste bug: comparison table prints "ResNet-34" as encoder name for vR.P.5 |
| W3 | **MAJOR** | 9M trainable decoder with only 8,829 training images — data:param ratio 1:1,021 |
| W4 | **MAJOR** | Image accuracy only 72.00% — worst in the P.3-P.6 batch (P.3: 86.79%, P.4: 84.42%) |
| W5 | **MAJOR** | FP rate 27.4% (308/1124) — 10x worse than P.3's 2.7% |
| W6 | **MAJOR** | Severe training instability: val metrics oscillate wildly (F1 range 0.03-0.47 across epochs) |
| W7 | MODERATE | Slow convergence: best epoch at 19, and pixel F1 at epoch 19 (0.4734) differs from test F1 (0.5137) |
| W8 | MODERATE | 202.6 MB model file — largest in series, deployment concern |
| W9 | MINOR | Overfitting growing: train_loss=0.37 vs val_loss=0.77 at epoch 25 (ratio 2.1x) |

---

## 5. Major Issues

### 5.1 CRITICAL: Copy-Paste Bugs (W1, W2)

Two output strings were copied from the parent notebook (vR.P.1.5, which uses ResNet-34) without updating:

1. **Model save filename:** `model_filename = f'{VERSION}_unet_resnet34_model.pth'` → saves as `vR.P.5_unet_resnet34_model.pth`
2. **Comparison table:** `print(f'{VERSION:<10} {"Pretrained":<12} {"ResNet-34":<12} ...')` → displays "ResNet-34" as the encoder

The actual encoder IS ResNet-50 (confirmed by `ENCODER = 'resnet50'` and model creation output `Model: UNet + resnet50 (imagenet)`). The bugs are cosmetic — they affect output strings, not the model itself. But they indicate insufficient review before execution and would confuse anyone inspecting the notebook outputs.

### 5.2 MAJOR: Decoder Size Problem (W3)

The 4x wider skip connections from ResNet-50 force SMP to create a 9M-parameter decoder (vs 3.15M for ResNet-34). With 8,829 training images, the data:param ratio is 1:1,021 — roughly 3x worse than ResNet-34 models (1:359). This asymmetry means the encoder provides richer features but the decoder has too many parameters to learn from them effectively.

### 5.3 MAJOR: Training Instability (W6)

Validation metrics oscillate dramatically:
- Epoch 3: F1=0.3059, then epoch 5: F1=0.2253, then epoch 8: F1=0.3701
- Epoch 12: F1=0.4207, then epoch 13: F1=0.3192 — a 10pp drop in one epoch
- The pattern suggests the decoder is too large to converge smoothly on this dataset size

---

## 6. Minor Issues

### 6.1 Model File Size (W8)

At 202.6 MB, the saved model is the largest in the pretrained series (P.4: 123.5 MB, P.6: 43.3 MB). The 9M decoder parameters contribute most of this size. For assignment submission and deployment, this is a practical concern.

### 6.2 Growing Overfitting (W9)

Train loss reaches 0.37 at epoch 25 while val loss stabilizes around 0.77-0.84 — a 2.1x ratio. The large decoder memorizes training patterns that don't generalize. This is worse than P.3's 1.73x ratio but better than P.2's 2.9x ratio.

---

## 7. Training Summary

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR |
|-------|-----------|----------|----------|-----|-----|
| 1 | 1.0872 | 1.0634 | 0.0347 | 0.0177 | 1e-3 |
| 3 | 0.9590 | 0.9311 | 0.3059 | 0.1806 | 1e-3 |
| 6 | 0.8800 | 0.9056 | 0.3452 | 0.2086 | 1e-3 |
| 9 | 0.7987 | 0.8580 | 0.3770 | 0.2323 | 1e-3 |
| 12 | 0.7181 | 0.8207 | 0.4207 | 0.2664 | 1e-3 |
| 17 | 0.5372 | 0.8356 | 0.4221 | 0.2675 | 5e-4 |
| **19** (best) | **0.4983** | **0.7687** | **0.4734** | **0.3101** | **5e-4** |
| 24 | 0.3917 | 0.8056 | 0.4532 | 0.2930 | 2.5e-4 |
| 25 | 0.3714 | 0.8384 | 0.4094 | 0.2574 | 2.5e-4 |

**LR schedule:** 1e-3 (epochs 1-16) → 5e-4 (epochs 17-23) → 2.5e-4 (epochs 24-25)

**Key observations:**
- **Epoch 1 near-zero F1** (0.0347): Much worse cold start than P.3 (0.4051) or P.4 (0.3513) — the 9M decoder needs more epochs to produce meaningful outputs
- **Extremely volatile val metrics:** F1 at epochs 11-14: 0.3917 → 0.4207 → 0.3192 → 0.3209 — 10pp swings between epochs
- **LR reduction at epoch 17** stabilized training somewhat, enabling the best result at epoch 19
- **Post-best degradation:** After epoch 19, val metrics worsen despite continued training — suggests overfitting onset

---

## 8. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.1.5 | Delta from P.1 |
|--------|-------|-------------------|----------------|
| Pixel Precision | 0.6089 | -0.0275 | -0.0246 |
| **Pixel Recall** | **0.4442** | **+0.1277** | **+0.0897** |
| **Pixel F1** | **0.5137** | **+0.0910** | **+0.0591** |
| **Pixel IoU** | **0.3456** | **+0.0776** | **+0.0514** |
| **Pixel AUC** | **0.8828** | **+0.0268** | **+0.0319** |

### Image-Level (Classification)

| Metric | Value | Delta from P.1.5 | Delta from P.1 |
|--------|-------|-------------------|----------------|
| Test Accuracy | 72.00% | +0.95pp | +1.85pp |
| Macro F1 | 0.7143 | +0.0127 | +0.0276 |
| ROC-AUC | 0.8126 | +0.0146 | +0.0341 |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.7861 | 0.7260 | 0.7549 | 1,124 |
| Tampered | 0.6398 | 0.7113 | 0.6736 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 816 (TN) | 308 (FP) |
| **Tp** | 222 (FN) | 547 (TP) |

- **FP rate: 27.4%** (308/1124) — worst in P.3-P.6 batch
- **FN rate: 28.9%** (222/769) — similar to P.3 (28.6%) and P.4 (29.0%)
- The high FP rate suggests the wider decoder generates more false positive activations

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"The authors swap ResNet-34 for ResNet-50 and observe a +9pp improvement in pixel F1. Promising! But then you look at the decoder: 9 million trainable parameters on 8,829 images. The data:param ratio is 1:1,021. This is not an encoder ablation — it's a study in what happens when you triple your model capacity and hope the data doesn't notice.

The copy-paste bugs are particularly egregious. If you change the encoder from ResNet-34 to ResNet-50, you should probably change the strings that say 'ResNet-34.' The model is saved as `vR.P.5_unet_resnet34_model.pth`. Imagine a future researcher loading this file expecting 34-layer features and getting 50-layer features instead.

But the real story here is P.3 vs P.5. P.3 reaches Pixel F1=0.6920 with **3.17M** trainable parameters and ELA input. P.5 reaches 0.5137 with **9.0M** trainable parameters and RGB input. That's 2.86x more parameters for 74% of the performance. The deeper encoder buys you sophistication; the ELA input buys you signal. Choose signal.

The training curves read like a seismograph. Pixel F1 oscillates 10pp between consecutive epochs. The cold start is agonizing — epoch 1 pixel F1 is 0.0347. This model spent its first 6 epochs learning what the 3M-parameter ResNet-34 version learned in 1. More depth is not free."

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction (pixel masks) | **PASS** | Model generates masks, saved (202.6 MB) |
| Train/val/test split | **PASS** | 70/15/15 stratified, seed=42 |
| Standard metrics (F1, IoU, AUC) | **PASS** | Full pixel-level + image-level suite |
| Visual results (Original/GT/Predicted/Overlay) | **PASS** | All visualization cells executed |
| Model weights (.pth file) | **PASS** | Saved (filename incorrectly says "resnet34") |
| Architecture explanation | **PASS** | Detailed pipeline with ResNet-50 block descriptions |
| Single notebook execution | **PASS** | End-to-end, no errors |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 12 | 15 | ResNet-50 is a valid choice, but decoder bloat (9M params) is a design flaw not addressed |
| Dataset | 14 | 15 | Proper GT masks, standard pipeline, consistent with series |
| Methodology | 14 | 20 | AMP/TF32 present, but copy-paste bugs (-3), training instability not analyzed (-2), epoch 1 cold start (-1) |
| Evaluation | 17 | 20 | All metrics computed, visualizations present, but no decoder size analysis, no skip channel discussion |
| Documentation | 10 | 15 | Copy-paste bugs reduce trust (-3), otherwise adequate markdown documentation |
| Assignment Alignment | 10 | 15 | All deliverables present, but copy-paste errors in output could confuse reviewers |
| **Total** | **77** | **100** | |

---

## 12. Final Verdict: **POSITIVE** — Score: 77/100

**Pixel F1: 0.5137 (+0.0910 from parent P.1.5 — exceeds +2pp POSITIVE threshold)**

vR.P.5 demonstrates that a deeper encoder (ResNet-50) improves localization over ResNet-34 when using frozen RGB features (+9.1pp Pixel F1). However, the improvement comes at the cost of a 3x larger decoder (9M params), 10x worse FP rate vs P.3, and training instability. The copy-paste bugs in output strings undermine confidence in the notebook's integrity.

**Critical context:** P.5's best result (0.5137) is **25.8% below** P.3's ELA-based result (0.6920) despite using 2.86x more trainable parameters. This conclusively demonstrates that **input representation (ELA) matters more than encoder depth (ResNet-50)** for forensic localization.

### Recommended Next Steps

1. **Do not pursue deeper encoders as the primary improvement path** — input modality is the bottleneck
2. **If re-running P.5:** fix copy-paste bugs, try ResNet-50 with ELA input to isolate encoder vs input effects
3. **Consider ResNet-50 + BN unfreeze** — the fully frozen BN may be limiting adaptation (P.3 showed BN unfreeze helps)
