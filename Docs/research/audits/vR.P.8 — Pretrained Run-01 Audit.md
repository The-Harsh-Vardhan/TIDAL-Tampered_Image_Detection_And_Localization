# Technical Audit: vR.P.8 — Pretrained Run-01

| Field | Value |
|-------|-------|
| **Auditor** | Principal AI Engineer (Automated Deep Audit) |
| **Date** | 2026-03-15 |
| **Notebook** | `vr-p-8-ela-gradual-encoder-unfreeze-run-01.ipynb` |
| **Platform** | Kaggle, Tesla P100-PCIE-16GB |
| **Training** | 32 epochs (3 stages), best at epoch 23 (Stage 0) |
| **Version** | vR.P.8 — ELA + Progressive Encoder Unfreeze |
| **Parent** | vR.P.3 (ELA, frozen+BN, Pixel F1=0.6920) |
| **Change** | 3-stage progressive encoder unfreeze (frozen → layer4 → layer3+layer4) |
| **Status** | **FULLY EXECUTED — ALL CELLS PASS** |

---

## 1. Notebook Overview & Experiment Goal

vR.P.8 tests whether **progressively unfreezing encoder layers** improves localization beyond P.3's frozen-encoder approach. The hypothesis is that the deepest encoder layers (layer3, layer4) contain task-generic features that benefit from fine-tuning on forensic data, while early layers (layer1, layer2) should remain frozen to preserve low-level feature extraction.

**3-Stage Protocol:**
- **Stage 0 (epochs 1-25):** All encoder frozen except BatchNorm (identical to P.3)
- **Stage 1 (epochs 26-32):** Unfreeze layer4 (deepest ResNet block), dual LR: encoder 1e-5, decoder 1e-3
- **Stage 2 (epochs 33+):** Unfreeze layer3+layer4, dual LR maintained

**Result:** Pixel F1 reaches **0.6985** (+0.65pp from P.3), with the best metrics at epoch 23 during Stage 0. Stage 1 (layer4 unfreeze) did not improve beyond Stage 0's peak. The progressive unfreeze hypothesis is **not confirmed** — the best performance came from the frozen training phase.

---

## 2. Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | UNet (SMP) |
| Encoder | ResNet-34 (ImageNet, **progressive unfreeze**) |
| Input | ELA 384x384, ImageNet normalization |
| Loss | SoftBCEWithLogitsLoss + DiceLoss (mode=binary, from_logits=True) |
| Optimizer | Adam (decoder: lr=1e-3, encoder: lr=1e-5 when unfrozen) |
| Scheduler | ReduceLROnPlateau (mode=min, factor=0.5, patience=3) |
| Batch size | 16 |
| Max epochs | Stage 0: 25, Stage 1: +10, Stage 2: +10 |
| Early stopping | patience=7 per stage, val_loss |
| Seed | 42 |
| Dataset | CASIA v2.0 (sagnikkayalcse52) — 12,614 images, 100% GT masks |

### Unfreeze Schedule

| Stage | Epochs | Unfrozen Layers | Encoder LR | Decoder LR | Trainable Params |
|-------|--------|----------------|------------|------------|-----------------|
| 0 | 1-25 | BN only | N/A (frozen) | 1e-3 | 3.17M (13%) |
| 1 | 26-32 | BN + layer4 | 1e-5 | 1e-3 | 10.46M (43%) |
| 2 | 33+ | BN + layer3 + layer4 | 1e-5 | 1e-3 | 14.13M (58%) |

### Parameters (at max unfreeze)

| Category | Count |
|----------|-------|
| Total | 24,439,617 |
| Trainable (Stage 2 max) | 14,130,257 (57.8%) |
| Frozen | 10,309,360 |
| Model file size | 282.8 MB (saves all params) |

---

## 3. Strengths

| # | Strength | Evidence |
|---|----------|----------|
| S1 | **Best Pixel F1 in ELA series: 0.6985** | +0.65pp over P.3's 0.6920 |
| S2 | **Highest Pixel Precision: 0.8857** | Best in entire pretrained series |
| S3 | **Best Image Accuracy: 87.59%** | Highest in pretrained track |
| S4 | **Best image-level ROC-AUC: 0.9578** | Strongest discrimination |
| S5 | **Best Pixel AUC: 0.9541** | Marginal improvement over P.3's 0.9528 |
| S6 | **Dual learning rate** | 100x lower LR for encoder prevents catastrophic forgetting |
| S7 | **Clean stage transitions** | Proper optimizer reset, LR re-warm, checkpoint loading |
| S8 | **Comprehensive evaluation** | Full pixel + image metrics, all visualizations present |

---

## 4. Weaknesses

| # | Severity | Weakness |
|---|----------|----------|
| W1 | **MAJOR** | Best epoch (23) is in Stage 0, not Stage 1/2 — progressive unfreeze didn't help |
| W2 | **MAJOR** | Stage 1 early-stopped at epoch 32 (best at epoch 23 in Stage 0) — training regressed |
| W3 | MODERATE | Model file 282.8 MB (2.3x larger than P.3's 123.4 MB) |
| W4 | MODERATE | Pixel recall 0.5766 — lower than P.3's 0.5905 |
| W5 | MODERATE | Stage 2 never meaningfully executed — early stopping cut it short |
| W6 | MINOR | Training time ~45 min vs P.3's ~25 min — 80% longer for +0.65pp |

---

## 5. Major Issues

### 5.1 MAJOR: Stage 0 Outperforms Stage 1 (W1, W2)

The progressive unfreeze hypothesis predicts that unfreezing deeper layers should improve performance. Instead, the best metrics come from Stage 0 (fully frozen encoder). Stage 1 (layer4 unfrozen) saw validation loss increase and Pixel F1 decrease from the Stage 0 peak.

**Interpretation:** The frozen ResNet-34 features, when combined with ELA input, are already well-suited for the task. Unfreezing layer4 introduces instability without meaningful benefit.

### 5.2 MAJOR: Stage 2 Not Meaningfully Tested (W5)

Stage 2 (layer3+layer4 unfreeze) was cut short by early stopping. With the model already degrading in Stage 1, Stage 2 had no chance to demonstrate value.

---

## 6. Minor Issues

### 6.1 Precision-Recall Trade-off (W4)

P.8 achieved the highest pixel precision (0.8857) but at the cost of lower recall (0.5766 vs P.3's 0.5905). The model became more conservative.

### 6.2 Model File Size (W3)

The model file is 282.8 MB because it saves all parameters (including frozen encoder weights). P.3's 123.4 MB file only contains trainable parameters.

---

## 7. Training Summary

### Stage 0 (Frozen + BN, epochs 1-25)

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR |
|-------|-----------|----------|----------|-----|-----|
| 1 | 0.9841 | 0.8682 | 0.4243 | 0.2688 | 1e-3 |
| 5 | 0.5776 | 0.5601 | 0.5862 | 0.4148 | 1e-3 |
| 10 | 0.4173 | 0.4648 | 0.6371 | 0.4675 | 1e-3 |
| 15 | 0.3419 | 0.4257 | 0.6638 | 0.4966 | 5e-4 |
| 20 | 0.2915 | 0.4103 | 0.6818 | 0.5172 | 2.5e-4 |
| **23** (best) | **0.2712** | **0.3986** | **0.6985** | **0.5367** | **1.25e-4** |
| 25 | 0.2621 | 0.4012 | 0.6952 | 0.5330 | 1.25e-4 |

### Stage 1 (Layer4 unfrozen, epochs 26-32)

| Epoch | Train Loss | Val Loss | Pixel F1 | IoU | LR (enc/dec) |
|-------|-----------|----------|----------|-----|--------------|
| 26 | 0.2804 | 0.4089 | 0.6921 | 0.5293 | 1e-5 / 1e-3 |
| 28 | 0.2667 | 0.4156 | 0.6878 | 0.5246 | 1e-5 / 1e-3 |
| 32 (final) | 0.2543 | 0.4234 | 0.6812 | 0.5168 | 5e-6 / 5e-4 |

**Key observations:**
- **Stage 0 peak at epoch 23** — natural convergence
- **Stage 1 regressed:** Val loss increased, Pixel F1 decreased
- **Early stopping triggered at epoch 32** — 7 epochs without improvement
- **Stage 2 never reached meaningful training**

---

## 8. Test Results

### Pixel-Level (Localization)

| Metric | Value | Delta from P.3 |
|--------|-------|----------------|
| **Pixel Precision** | **0.8857** | **+0.0501** |
| Pixel Recall | 0.5766 | -0.0139 |
| **Pixel F1** | **0.6985** | **+0.0065** |
| **Pixel IoU** | **0.5367** | **+0.0076** |
| **Pixel AUC** | **0.9541** | **+0.0013** |

### Image-Level (Classification)

| Metric | Value | Delta from P.3 |
|--------|-------|----------------|
| **Test Accuracy** | **87.59%** | **+0.80pp** |
| **Macro F1** | **0.8650** | **+0.0090** |
| **ROC-AUC** | **0.9578** | **+0.0076** |

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Authentic | 0.8478 | 0.9769 | 0.9077 | 1,124 |
| Tampered | 0.9524 | 0.7282 | 0.8253 | 769 |

### Confusion Matrix

| | Pred Au | Pred Tp |
|-|---------|---------|
| **Au** | 1,098 (TN) | 26 (FP) |
| **Tp** | 209 (FN) | 560 (TP) |

- **FP rate: 2.3%** (26/1,124) — best in entire series
- **FN rate: 27.2%** (209/769) — improved over P.3's 28.6%

---

## 9. Roast (Conference Reviewer Style)

**Reviewer 2 writes:**

"The authors propose a 3-stage progressive unfreeze protocol with dual learning rates. Ambitious. Methodical. The kind of experiment that makes you nod approvingly during the introduction.

Then you reach the results. The best performance comes from Stage 0. The FROZEN stage. The part of the experiment that is literally identical to the parent model. Stage 1 — the entire point of this paper — made things worse. Stage 2 never even got to play.

So what we have is: train P.3 for 25 epochs instead of 25 epochs, and get a marginally better result at epoch 23 because stochastic gradient descent smiled upon us. The +0.65pp improvement is within the noise band of the reproducibility run. The progressive unfreeze protocol adds complexity, training time, and a larger model file, delivering nothing that 2 more epochs of frozen training wouldn't provide.

The pixel precision of 0.8857 is genuinely impressive — the model has become extremely confident about the pixels it marks as tampered. But it achieves this by becoming more conservative (recall drops to 0.5766).

+0.65pp for 80% more training time and 2.3x the model size. The cost-benefit analysis writes itself."

---

## 10. Assignment Alignment

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tampered region prediction (pixel masks) | **PASS** | Model generates 384x384 binary masks |
| Train/val/test split | **PASS** | 70/15/15 stratified, seed=42 |
| Standard metrics (F1, IoU, AUC) | **PASS** | Full pixel-level + image-level suite |
| Visual results (Original/GT/Predicted/Overlay) | **PASS** | All visualization cells executed |
| Model weights (.pth file) | **PASS** | Saved: 282.8 MB |
| Architecture explanation | **PASS** | Progressive unfreeze protocol documented |
| Single notebook execution | **PASS** | End-to-end on Kaggle |

---

## 11. Score Breakdown

| Category | Score | Max | Notes |
|----------|-------|-----|-------|
| Architecture | 14 | 15 | Progressive unfreeze is well-designed. Dual LR, proper checkpointing. |
| Dataset | 14 | 15 | Same proven pipeline as P.3. Standard splits, GT masks. |
| Methodology | 14 | 20 | 3-stage protocol rigorous but Stage 1/2 counterproductive (-4), large model file (-2) |
| Evaluation | 18 | 20 | Comprehensive metrics, per-stage analysis, all visualizations |
| Documentation | 12 | 15 | Good stage documentation, training curves per stage |
| Assignment Alignment | 12 | 15 | All deliverables present, best overall metrics in pretrained track |
| **Total** | **84** | **100** | |

---

## 12. Final Verdict: **NEUTRAL** (+0.65pp from P.3) — Score: 84/100

**Pixel F1: 0.6985 (+0.0065 from P.3's 0.6920 — within NEUTRAL threshold of ±2pp)**

vR.P.8 achieves the best absolute metrics in the pretrained track series (Pixel F1: 0.6985, Image Accuracy: 87.59%), but the improvement over P.3 is marginal (+0.65pp). The progressive unfreeze hypothesis is not confirmed — the best results come from Stage 0 (frozen encoder), not from the unfreezing stages.

### Key Insight: Frozen Features Are Near-Optimal for ELA Input

| Stage | Training | Pixel F1 | Verdict |
|-------|----------|----------|---------|
| Stage 0 | Frozen encoder + BN | **0.6985** | Best result |
| Stage 1 | + Layer4 unfrozen | 0.6812 | Regressed |
| Stage 2 | + Layer3 unfrozen | N/A | Cut short |

The pretrained ResNet-34 features, when applied to ELA input, require minimal adaptation. Future experiments should focus on decoder improvements (P.10: CBAM attention) or input representation changes (P.11: higher resolution) rather than encoder fine-tuning.
