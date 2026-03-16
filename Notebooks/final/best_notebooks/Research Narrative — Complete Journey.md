# Research Narrative — Complete Journey

## From Paper Reproduction to Pixel-Level Forgery Localization on CASIA v2.0

**Project:** Image Tampering Detection and Localization
**Dataset:** CASIA v2.0 (12,614 images: 7,491 Authentic + 5,123 Tampered)
**Reference Paper:** "Enhanced Image Tampering Detection using ELA and a CNN" (Nagm et al. 2024, PeerJ CS / ETASR 9593)
**Total Experiments:** 26+ across 3 tracks
**Best Result:** Pixel F1 = 0.7277, Image Accuracy = 87.32% (vR.P.10 — CBAM Attention UNet)

---

## Phase 1: Literature Review and Reference Code Audit

The project began with a literature survey of the image forgery detection landscape. Five IEEE/PeerJ papers were reviewed, with Nagm et al. 2024 selected as the primary reference — it proposed a lightweight CNN trained on Error Level Analysis (ELA) images to classify CASIA v2.0 images as authentic or tampered, reporting 96.21% accuracy.

Eighteen reference notebooks were collected from Kaggle covering a range of approaches: ELA+CNN classification, copy-move detection, UNet segmentation, and various dataset variants. Two standalone Python implementations were audited in detail, revealing 16 bugs including fatal issues (incorrect loss functions, missing data splits, evaluation on training data).

### The "image-detection-with-mask" Reference Notebook

Among the collected references, one notebook proved particularly influential: `image-detection-with-mask (1).ipynb` — a PyTorch dual-head notebook that attempted both classification and segmentation on the CASIA Splicing dataset. It introduced the `UNetWithClassifier` architecture that would become the foundation for the entire vK track: a vanilla UNet encoder (64-128-256-512-1024 channels) with skip connections, a segmentation head (Conv2d 1x1), and a classification head (AdaptiveAvgPool + Linear layers). No pretrained encoder — all 31.6M parameters trained from scratch on RGB input at 256x256.

The notebook ran two training configurations:
- **Run 1** (BCE loss, 30 epochs): 71.88% accuracy, Dice = 0.5949
- **Run 2** (FocalLoss + Dice, 50 epochs): **89.75% accuracy**, Dice = 0.5673

The classification improvement from Run 1 to Run 2 was genuine — FocalLoss with class weights addressed the class imbalance effectively. But the segmentation Dice of ~0.57 was deceptive: on a dataset where ~40% of images are authentic (all-zero masks), a model predicting all-zero masks for everything scores roughly Dice = 0.58 because empty masks contribute perfect scores. **The segmentation head was essentially not learning.**

The audit uncovered a critical data leak: `TEST_CSV` pointed to `val_metadata.csv`, not the test set. The reported 89.75% "test" accuracy was actually validation accuracy wearing a test name tag. The real test performance was never evaluated.

This notebook served as both a template and a cautionary tale. It demonstrated that dual-head UNet could achieve strong classification, but that training 31.6M parameters from scratch on ~8,800 RGB images was insufficient for meaningful segmentation. The diagnosis — **no pretrained features, no ELA signal, insufficient data:param ratio** — directly motivated the pretrained localization track (vR.P.x) that would ultimately solve the problem.

### Early Notebook Iterations

Early notebook versions (v1 through v9) were created manually, iterating on the `UNetWithClassifier` architecture from `image-detection-with-mask`. These form the "legacy vK track" — useful as stepping stones but ultimately outperformed by later systematic work.

---

## Phase 2: The "New Research Approach" Pivot

A critical inflection point came when the project was restructured from ad-hoc notebook iteration to a systematic, paper-grounded research approach. This introduced:

- **Two parallel tracks:** ETASR (classification ablation on the paper's CNN) and Pretrained (UNet localization with ImageNet encoders)
- **Master documentation:** 6 reference files covering architecture analysis, dataset analysis, training methodology, code audit, and implementation roadmap
- **Per-experiment protocol:** Every experiment gets a versioned notebook, documentation (experiment description, implementation plan, expected outcomes), and a post-run audit report
- **Ablation methodology:** One variable changes per experiment, with verdicts of POSITIVE (>+2pp), NEUTRAL (+/-2pp), or REJECTED (regression)

This structure transformed the project from "trying things" to "systematic scientific investigation."

---

## Phase 3: ETASR Classification Track (vR.ETASR to vR.1.7)

The ETASR track reproduced and ablated the paper's CNN architecture: 2x Conv2D(32, 5x5) + MaxPool + Flatten + Dense(256) + Softmax. All experiments used ELA preprocessing at quality 90, 128x128 resolution.

### The Prototypes (vR.ETASR, vR.0, vR.1)

The first three versions established the pipeline but had flawed evaluation — they used validation-set metrics as the headline number. vR.ETASR reported 89.89% accuracy, but this was evaluated on the same data used for early stopping.

### vR.1.1 — The Honest Baseline

The turning point for the ETASR track. Added a proper 70/15/15 train/val/test split and per-class metrics. The honest test accuracy landed at **88.38%** — a 7.83pp gap from the paper's claimed 96.21%. This became the baseline against which all ablations were measured.

Key metrics: 88.38% accuracy, 0.8805 Macro F1, 0.9601 ROC-AUC.

### vR.1.2 — Data Augmentation (REJECTED)

Added horizontal/vertical flips and +/-15 degree rotation. The result was catastrophic: accuracy dropped to **85.53%** (-2.85pp), and the model's best epoch was epoch 1 — it learned nothing from augmented data.

Root cause: the Flatten-to-Dense(256) bottleneck memorizes exact pixel positions. Augmented images activate completely different neurons. This was the most instructive failure in the entire project — **you cannot augment your way out of an architecture problem.**

### vR.1.3 to vR.1.5 — Training Tricks

Three incremental improvements, each branching from vR.1.1 (since vR.1.2 was rejected):

- **vR.1.3 (Class Weights):** 89.17% (+0.79pp). Improved tampered recall from 0.8830 to 0.9012. Verdict: POSITIVE.
- **vR.1.4 (BatchNorm):** 88.75% (-0.42pp). BN caused the worst single-epoch catastrophe in the series (val_loss=16.13 at epoch 1) but recovered by epoch 3. Tampered recall improved to 0.9194. Verdict: NEUTRAL.
- **vR.1.5 (LR Scheduler):** 88.96% (+0.21pp). ReduceLROnPlateau fired once, buying 2 extra epochs. Every metric within noise of vR.1.4. Verdict: NEUTRAL.

Combined contribution of all three training tricks: **+0.58pp** accuracy.

### vR.1.6 — Deeper CNN (The Breakthrough)

Added a 3rd convolutional layer (Conv2D(64, 3x3) + MaxPool) before the Flatten layer. Paradoxically, adding a layer **reduced** parameters from 29.5M to 13.8M (-53%) because the additional pooling shrank the spatial dimensions before Flatten.

Result: **90.23%** accuracy (+1.85pp from baseline), the first time the ETASR series crossed 90%. Macro F1 = 0.9004, ROC-AUC = 0.9657. Trained for 18 epochs — the longest in the series.

This was definitive proof that **architecture changes matter more than training tricks.** One structural change (+1.85pp) outperformed all three training changes combined (+0.58pp) by 3.2x.

### vR.1.7 — Global Average Pooling

Replaced Flatten with GlobalAveragePooling2D, collapsing parameters from 13.8M to just **64K** (-99.5%). Accuracy came in at 89.17% — only 1.06pp below vR.1.6 despite having 215x fewer parameters.

The tradeoff was informative: GAP averages away all spatial information, which matters for forensics where the *location* of anomalies is critical. But 89.17% with a 250KB model is remarkable efficiency.

### ETASR Track Summary

| Version | Test Acc | Macro F1 | ROC-AUC | Params | Verdict |
|---------|----------|----------|---------|--------|---------|
| vR.1.1 (baseline) | 88.38% | 0.8805 | 0.9601 | 29.5M | Honest Baseline |
| vR.1.2 (augmentation) | 85.53% | 0.8505 | 0.9011 | 29.5M | REJECTED |
| vR.1.3 (class weights) | 89.17% | 0.8889 | 0.9580 | 29.5M | POSITIVE |
| vR.1.4 (BatchNorm) | 88.75% | 0.8852 | 0.9536 | 29.5M | NEUTRAL |
| vR.1.5 (LR scheduler) | 88.96% | 0.8873 | 0.9560 | 29.5M | NEUTRAL |
| **vR.1.6 (deeper CNN)** | **90.23%** | **0.9004** | **0.9657** | **13.8M** | **POSITIVE** |
| vR.1.7 (GAP) | 89.17% | 0.8901 | 0.9495 | 64K | NEUTRAL |

**Critical limitation:** Every ETASR experiment is classification-only. None can produce pixel-level localization masks. The assignment requires localization.

---

## Phase 4: Standalone Paper Reproduction

Three standalone runs attempted to reproduce the reference paper's exact architecture on different datasets:

### Paper CNN on divg07 Dataset
Test accuracy: **90.33%** vs the paper's claimed 94.14% — a 3.81pp gap never closed. The gap is likely explained by the paper's JPEG-only filtering (9,501 images) vs our use of all formats (12,614 images).

### Paper CNN on sagnik Dataset (DATA LEAK)
Test accuracy: **100.00%** — a result that is always suspicious on real-world data. Investigation revealed a critical data leak: the sagnik dataset contained ground truth mask images mixed in as input data. The CNN trivially distinguished masks from photos. This result is scientifically invalid and must never be cited.

### Deeper CNN on divg07 Dataset
A deeper 3-block architecture (Conv64 + Conv128 + Conv256 + BN + Dense(512)) achieved **90.76%** — the best classification accuracy across all experiments. Tampered recall reached 96.27%. But with 38.3M parameters and no localization capability, it confirmed: **better classification does not equal better localization.**

---

## Phase 5: Pretrained Localization Track (vR.P.0 to vR.P.10)

This track addressed the assignment's core requirement: pixel-level forgery localization. All experiments used UNet from Segmentation Models PyTorch (SMP) with pretrained ImageNet encoders. Input resolution: 384x384.

### P.0 and P.1 — Establishing the Baseline

**vR.P.0** was the first localization experiment: frozen ResNet-34 encoder, RGB input, BCE+Dice loss. Without proper ground truth masks (142 images used ELA pseudo-masks), it achieved Pixel F1 = 0.3749.

**vR.P.1** fixed the dataset by switching to sagnikkayalcse52 with real GT masks for all 5,123 tampered images. Pixel F1 improved to **0.4546** (+0.0797). This became the proper baseline.

### P.1.5 — Speed Infrastructure

Added AMP, TF32, pin_memory, persistent workers. No model changes. Metrics stayed within noise (Pixel F1 = 0.4227). All speed optimizations carried forward to future experiments.

### P.2 — Gradual Encoder Unfreeze

Unfroze encoder layer3 + layer4 with differential learning rates. Pixel F1 improved to **0.5117** (+0.0571), but image accuracy dropped and the model severely overfit (23M trainable params on 8,829 training images = 1:2,615 data:param ratio). A cell crash also prevented model saving.

### P.3 — ELA as Input (The Second Major Breakthrough)

The most impactful single change in the entire project. Replaced RGB input with ELA (Error Level Analysis) images — JPEG compressed at quality 90, then the absolute difference between original and recompressed images is computed.

The results were dramatic:
- Pixel F1: **0.6920** (+0.2374 from P.1, a **+23.74pp jump**)
- Image accuracy: **86.79%** (+16.64pp from P.1)
- Pixel AUC: 0.9528
- FP rate dropped to just 2.7%

Why? ELA highlights compression inconsistencies. Tampered regions that were saved at different JPEG quality levels show as bright spots in ELA images. This gives the UNet a direct signal pointing to manipulated areas, rather than having to infer tampering from subtle RGB artifacts.

The model reverted to frozen encoder + BN unfreeze only (3.17M trainable) yet massively outperformed P.2's 23M-parameter unfrozen model. **Input representation matters more than model capacity.**

A critical bug (`NameError: denormalize`) crashed the visualization cells and prevented model saving in Run-01. Run-02 confirmed exact reproducibility (identical metrics) and saved the model.

### P.4 — 4-Channel RGB+ELA

Concatenated RGB (3 channels) with ELA grayscale (1 channel) for 4-channel input. Pixel F1 = **0.7053** (+0.0133 from P.3), but image accuracy dropped to 84.42% and FP rate increased 2.4x. An epoch 10 catastrophe (val_loss spiked 34%) showed that RGB adds noise to the ELA signal rather than useful complementary information.

### P.5 and P.6 — Encoder Swaps

**vR.P.5 (ResNet-50):** Pixel F1 = 0.5137 with 9M trainable params. Despite 2.86x more parameters than P.3, it scored 25.8% lower on Pixel F1.

**vR.P.6 (EfficientNet-B0):** Pixel F1 = 0.5217, smallest model at 2.24M trainable. Best parameter efficiency but still far below ELA-based results.

Both experiments used RGB input, confirming the hierarchy: **ELA input (P.3, 3.17M params, F1=0.6920) >> ResNet-50 (P.5, 9M params, F1=0.5137) >> EfficientNet-B0 (P.6, 2.24M params, F1=0.5217)**. Input representation dominates encoder architecture.

### P.7 — Extended Training (50 Epochs)

Same architecture as P.3, but doubled max_epochs from 25 to 50. P.3's best epoch was 25/25 (the last epoch), suggesting premature stopping. With more runway, the model found its optimum at **epoch 36** and early-stopped at epoch 46.

Pixel F1 = **0.7154** (+0.0234 from P.3). Image accuracy = 87.37%. This was essentially free performance — no architecture change, just more training time.

### P.8 — Progressive Encoder Unfreeze

A 3-stage progressive unfreeze strategy: frozen+BN for epochs 1-25, then layer4 for 26-32, then layer3+4 for 33+. Hypothesis: gradually adapting encoder features to ELA should help.

Result: Pixel F1 = 0.6985 and Image accuracy = **87.59%** (best pretrained). But the model's best epoch was 23 — in Stage 0 (fully frozen), before any unfreezing occurred. The unfreeze stages provided no benefit.

This confirmed that **frozen ImageNet features + BN adaptation is the sweet spot for ELA input.** The encoder doesn't need fine-tuning.

### P.9 — Focal + Dice Loss

Replaced BCE with Focal Loss (alpha=0.25, gamma=2.0). Pixel F1 = 0.6923 (essentially unchanged from P.3's 0.6920). Pixel AUC actually regressed (-0.0205) because Focal Loss concentrates predictions near 0/1 extremes, destroying probability calibration.

The existing BCE+Dice combination already handles class imbalance through the Dice component. Focal Loss is counterproductive here.

### P.10 — CBAM Attention (New Series Best)

Added CBAM (Convolutional Block Attention Module) to all 5 UNet decoder blocks. CBAM applies sequential channel attention then spatial attention, teaching the decoder to focus on relevant features and spatial locations. Cost: just 11,402 parameters (0.36% of trainable).

Result: Pixel F1 = **0.7277** (new series best, +3.57pp from P.3), Pixel IoU = 0.5719, Pixel AUC = 0.9573, Image ROC-AUC = 0.9633. FP rate = 2.0% (best ever).

CBAM is the second most efficient improvement in the project: 11,402 parameters for +3.57pp, compared to ELA's +23.74pp from a preprocessing change and extended training's +2.34pp from zero parameter changes.

The model's best epoch was 24 of 25 — it hit the epoch ceiling again, strongly suggesting more training would help.

### Pretrained Track Summary

| Version | Change | Pixel F1 | Pixel IoU | Img Acc | Verdict |
|---------|--------|----------|-----------|---------|---------|
| P.0 (no GT masks) | First pretrained | 0.3749 | 0.2307 | 70.63% | Baseline |
| P.1 (GT masks) | Dataset fix | 0.4546 | 0.2942 | 70.15% | Proper Baseline |
| P.2 (unfreeze) | Layer3+4 unfrozen | 0.5117 | 0.3439 | 69.04% | POSITIVE (pixel) |
| **P.3 (ELA input)** | **RGB -> ELA** | **0.6920** | **0.5291** | **86.79%** | **STRONG POSITIVE** |
| P.4 (4ch RGB+ELA) | 4-channel input | 0.7053 | 0.5447 | 84.42% | NEUTRAL |
| P.5 (ResNet-50) | Deeper encoder | 0.5137 | 0.3456 | 72.00% | POSITIVE |
| P.6 (EffNet-B0) | Efficient encoder | 0.5217 | 0.3529 | 70.68% | POSITIVE |
| P.7 (50 epochs) | Extended training | 0.7154 | 0.5569 | 87.37% | POSITIVE |
| P.8 (progressive) | Staged unfreeze | 0.6985 | 0.5367 | 87.59% | NEUTRAL |
| P.9 (Focal+Dice) | Loss change | 0.6923 | 0.5294 | 87.16% | NEUTRAL |
| **P.10 (CBAM)** | **Attention modules** | **0.7277** | **0.5719** | **87.32%** | **POSITIVE** |

---

## Phase 6: The Legacy vK Track

The older notebook series (v1 through vK.12.0) from `Notebooks/Runs/` was evaluated for completeness. These notebooks all descended from the `image-detection-with-mask` reference notebook's `UNetWithClassifier` architecture — a vanilla UNet (no pretrained encoder, 31.6M params) with a classification head, trained from scratch on RGB at 256x256. The reference notebook's segmentation failure (Dice ~0.57, essentially predicting all-zeros) and data leak (`TEST_CSV` pointing to val set) were the starting conditions that needed to be overcome.

**Best results:**
- **v6.5:** Tampered F1 = 0.4101, Image accuracy = 82.46%. Best localization from this track but far below vR.P.x.
- **vk-7-1:** Overall Dice = 0.5780, Image accuracy = 89.91%. Competitive classification but the Dice metric includes authentic images (which score trivially high).
- **vK.10.3b (100 epochs):** Tampered Dice = 0.2205, Image accuracy = 83.04%. Training duration was critical — patience=10 versions early-stopped at epoch 11 with near-zero Dice.
- **vK.10.6:** Tampered Dice = 0.2213 (optimized at threshold 0.15), Image accuracy = 83.57%. Most comprehensive evaluation suite.
- **vK.11.x and vK.12.x:** Model collapse. Tampered Dice ~0.13, accuracy as low as 40%. These experiments failed to learn meaningful segmentation.

**Key takeaway:** The vK track's best tampered F1 (0.4101 from v6.5) is 43.6% lower than vR.P.10's 0.7277. The pretrained ImageNet encoder + ELA input combination proved decisively superior to training from scratch on RGB.

---

## Key Findings and Lessons Learned

### 1. Input representation is king
The switch from RGB to ELA (P.3) produced a +23.74pp Pixel F1 improvement — larger than all other gains combined across both tracks. Every subsequent best result uses ELA. ELA directly encodes compression inconsistencies that signal tampering, giving the model a dramatically better signal-to-noise ratio.

### 2. Architecture changes outperform training tricks
In the ETASR track: the deeper CNN (vR.1.6, +1.85pp) outperformed class weights + BatchNorm + LR scheduler combined (+0.58pp) by 3.2x. In the pretrained track: CBAM attention (P.10, +3.57pp) was the largest single improvement after ELA. Structural improvements move the ceiling; training tricks approach it.

### 3. The paper's claimed 96.21% accuracy is not reproducible
Best honest reproduction: 90.23% (vR.1.6), with a persistent 5.98pp gap. The standalone paper CNN reproduced at 90.33%. The gap is likely due to JPEG-only dataset filtering in the original paper (9,501 images) vs our all-format approach (12,614 images).

### 4. The Flatten-Dense bottleneck is architecturally toxic
This bottleneck caused: augmentation failure (vR.1.2), training instability in every ETASR run, and 99%+ parameter concentration. Reducing it (vR.1.6, -53% params) improved results. Eliminating it (vR.1.7, -99.5% params) showed the model can nearly match with 64K parameters.

### 5. Frozen encoder features + BN adaptation is optimal for ELA
P.3 (3.17M trainable, frozen body + BN) massively outperformed P.2 (23M trainable, aggressive unfreeze). P.8's progressive unfreeze confirmed that the best epoch occurred before any unfreezing. ImageNet features, when adapted only through BatchNorm statistics, are near-optimal for ELA input.

### 6. Training budget matters but has diminishing returns
P.3 at 25 epochs: F1 = 0.6920. P.7 at 50 epochs: F1 = 0.7154 (+2.34pp). That's a 4.2x longer training for +2.34pp. Worth it, but not as impactful as architecture or input changes.

### 7. CBAM attention is the most parameter-efficient improvement
11,402 parameters (0.36% of trainable) produced +3.57pp Pixel F1. That's 0.313pp per thousand parameters — by far the highest improvement density of any change.

### 8. Always validate your data
The sagnik dataset's 100% accuracy revealed a data leak (GT masks loaded as input). The vK.10-12.x series' model collapse revealed instability in training-from-scratch approaches. Suspicious results always warrant investigation.

### 9. Classification and localization are fundamentally different problems
The best classifier (Deeper CNN, 90.76%) cannot localize. The best localizer (P.10, 87.32% image accuracy) trails by 3.44pp on classification. The assignment requires localization — only the pretrained UNet track is viable.

---

## Final Leaderboard — Top 5 Selected Experiments

| # | Notebook | Pixel F1 | Pixel IoU | Pixel AUC | Img Acc | Img ROC-AUC | Why Selected |
|---|----------|----------|-----------|-----------|---------|-------------|--------------|
| 1 | **vR.P.10** — CBAM Attention | **0.7277** | **0.5719** | **0.9573** | 87.32% | **0.9633** | Best localization overall |
| 2 | **vR.P.7** — Extended Training | 0.7154 | 0.5569 | 0.9504 | 87.37% | 0.9433 | 2nd best Pixel F1; best FN rate |
| 3 | **vR.P.8** — Progressive Unfreeze | 0.6985 | 0.5367 | 0.9541 | **87.59%** | 0.9578 | Best image accuracy among localizers |
| 4 | **vR.P.3** — ELA Breakthrough | 0.6920 | 0.5291 | 0.9528 | 86.79% | 0.9502 | Most impactful single innovation |
| 5 | **vR.1.6** — Deeper CNN | N/A | N/A | N/A | **90.23%** | **0.9657** | Best classification (ETASR champion) |

---

## What's Next

The single most promising next experiment is combining **vR.P.10's CBAM architecture** with **vR.P.7's extended training budget (50 epochs)**. Evidence:

- P.10 hit the epoch ceiling: best epoch was 24 out of 25
- P.7 showed extended training adds +2.34pp Pixel F1
- Projected result: Pixel F1 in the range of 0.74-0.75

Beyond that, potential directions include:
- **Multi-scale ELA** (multiple quality levels as input channels)
- **Attention-gated skip connections** (learnable gate on each skip instead of simple concatenation)
- **Post-processing** (CRF or morphological operations on predicted masks)
- **Larger input resolution** (512x512 or higher, if GPU memory allows)

---

*This document was compiled from 30+ audit reports (including reference notebook audits), 2 experiment tracking documents, and 26+ experiment runs across 3 research tracks.*
