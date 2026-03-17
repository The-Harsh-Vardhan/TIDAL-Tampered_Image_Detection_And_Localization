# Docs11: Weaknesses and Limitations

A systematic catalog of every identified weakness in vK.10.5, organized by severity and category. Each weakness cites its source audit or research evidence.

---

## Severity Definitions

| Level | Meaning |
|---|---|
| CRITICAL | Directly limits grade or produces misleading results |
| HIGH | Significant gap relative to assignment requirements or research best practices |
| MEDIUM | Reduces quality but does not block submission |
| LOW | Nice-to-have improvement |

---

## 1. Architectural Weaknesses

### W1: No Pretrained Encoder — CRITICAL

The vK.10.5 UNet trains all ~31M parameters from scratch on ~5,500 training images. The v8 run used SMP UNet with ImageNet-pretrained ResNet34 and achieved AUC=0.817. Training from scratch means the encoder must simultaneously learn generic visual features (edges, textures, shapes) AND forensic features (compression artifacts, noise patterns), which is data-inefficient.

Every surveyed research paper (P1-P21) uses pretrained encoders. The custom DoubleConv encoder has non-standard channel progression (64→128→256→512→1024) that prevents loading standard pretrained weights.

**Source:** Audit10/04_Head_to_Head_Comparison.md, Audit10/06_Roast.md

---

### W2: RGB-Only Input — No Forensic Preprocessing — CRITICAL

The model receives only 3-channel RGB input. Research consensus is that RGB alone contains insufficient information for robust tamper detection:

- **ELA (Error Level Analysis):** JPEG re-save amplifies compression inconsistencies between authentic and tampered regions. Papers P1/P7 achieved 96.21% accuracy on CASIA using ELA + lightweight CNN.
- **SRM noise maps:** High-pass filter residuals expose manipulation traces invisible in RGB. Papers P13 (EMT-Net, AUC=0.987) and P17 (ME-Net, F1=0.905) rely on SRM as a core input.
- **Chrominance (YCbCr Cb/Cr):** Paper P16 demonstrated 96.52% accuracy using chrominance channels alone.

The single most impactful improvement identified across all research papers is adding forensic preprocessing as additional input channels.

**Source:** Research_Paper_Analysis_Report.md, Paper_Analyses/Resource_07, Resource_08

---

### W3: No Edge Supervision — HIGH

Both top-performing research methods use explicit edge supervision:

| Method | Edge Technique | Result |
|---|---|---|
| EMT-Net (P13) | Edge Artifact Enhancement (EAE) | AUC=0.987 on NIST |
| ME-Net (P17) | Edge Enhancement Path Aggregation (EEPA) | F1=0.905 on NIST16 |

Manipulation boundaries are the most informative forensic signal. The current loss function (BCE + Dice) provides no explicit incentive to predict accurate boundaries — it optimizes for region overlap, which allows blurry, imprecise edges.

**Source:** Paper_Analyses/Resource_03, Resource_16

---

### W4: No Attention Mechanisms — MEDIUM

The encoder uses simple MaxPool downsampling and the decoder uses TransposedConv + concatenation skip connections. There is no channel attention (SE/CBAM) or spatial attention to weight feature importance.

TransU2-Net (P21) demonstrated +14.2% F-measure improvement from adding self-attention in the encoder and cross-attention in skip connections. This is a lightweight upgrade that does not require architectural redesign.

**Source:** Paper_Analyses/Resource_23

---

## 2. Evaluation Weaknesses

### W5: Fixed Segmentation Threshold — CRITICAL

The segmentation threshold is hardcoded at 0.5 with no optimization. The v8 run found the optimal threshold at 0.75 — a massive 0.25 shift that is only discoverable through sweeping. This is a free metric improvement requiring zero retraining (~30 lines of code).

Models are rarely calibrated at exactly 0.5. Without threshold optimization, reported metrics are artificially suppressed relative to the model's true capability.

**Source:** Audit10/05_Best_Features_Missing_in_vK103.md (item #2)

---

### W6: No Robustness Testing — CRITICAL

This is an explicit assignment bonus requirement (B1: "Testing robustness against distortions such as JPEG compression, resizing, cropping, and noise").

v8 implemented a full 8-condition robustness suite and revealed actionable insights: only 0.9% metric drop for JPEG compression but 13% drop for Gaussian noise. This kind of analysis directly earns bonus points.

vK.10.5 has zero robustness testing.

**Source:** Assignment.md (Bonus B1), Audit10/05_Best_Features_Missing_in_vK103.md (item #1)

---

### W7: No Grad-CAM Explainability — HIGH

No visualization of what the model attends to internally. Without explainability:

- Cannot verify whether the model learned meaningful forensic features or dataset shortcuts
- Cannot demonstrate "thoughtful architecture choices" to the reviewer (Assignment Section 2)
- Cannot identify failure modes at the feature level

v8 implemented Grad-CAM with diagnostic coloring (TP=green, FP=red, FN=blue).

**Source:** Audit10/05_Best_Features_Missing_in_vK103.md (item #3)

---

### W8: No Forgery-Type Breakdown — MEDIUM

Cannot distinguish model performance on splicing vs copy-move forgeries. v8 revealed a critical asymmetry: splicing F1=0.58 vs copy-move F1=0.14 (near random). This insight is invisible without per-type evaluation and directly relates to bonus B2 ("Successfully detecting subtle tampering such as copy-move manipulation").

**Source:** Audit10/05_Best_Features_Missing_in_vK103.md (item #6), v08_kaggle_run_01/03_Model_Architecture_and_Training_Review.md

---

### W9: No Mask-Size Stratification — MEDIUM

Cannot identify performance variation by tampered region size. v8 showed dramatic stratification:

| Mask Size | F1 |
|---|---|
| Tiny (<2%) | 0.14 |
| Small (2-5%) | 0.24 |
| Medium (5-15%) | 0.41 |
| Large (>15%) | 0.56 |

Tiny-mask detection being near random is a critical finding that drives architectural decisions (e.g., multi-scale processing, attention mechanisms).

**Source:** Audit10/05_Best_Features_Missing_in_vK103.md (item #5)

---

### W10: No Pixel-Level AUC-ROC — MEDIUM

Image-level AUC-ROC is implemented but pixel-level AUC is not. Pixel-level AUC is a threshold-independent localization metric used by both EMT-Net (P13) and ME-Net (P17). It avoids the arbitrary threshold problem entirely and is trivial to compute with `sklearn.metrics.roc_auc_score(gt.flatten(), pred.flatten())`.

**Source:** Paper_Analyses/Resource_03, Resource_16

---

## 3. Training Weaknesses

### W11: Aspect Ratio Destruction — MEDIUM

All images are squashed to 256×256 regardless of original aspect ratio. This destroys spatial proportions and can:

- Compress tampered regions in one dimension, making them harder to detect
- Introduce artificial geometric artifacts that confuse boundary detection
- Change the effective scale of small tampered regions

The recommended approach is resize-with-padding to preserve aspect ratio.

**Source:** v08_kaggle_run_01/02_Data_Pipeline_Review.md

---

### W12: No Gradient Accumulation — LOW-MEDIUM

Effective batch size is determined solely by VRAM (8-32). v8 used 4-step gradient accumulation for effective batch=256, which stabilizes training significantly by reducing gradient noise. On T4 with batch_size=16, gradient accumulation of 4 steps would give effective batch=64 — a meaningful improvement in gradient quality.

**Source:** Audit10/05_Best_Features_Missing_in_vK103.md (item #12)

---

### W13: No Differential Learning Rates — LOW

A single learning rate (1e-4) is used for all parameters. This is acceptable when training from scratch but becomes important when a pretrained encoder is introduced. Pretrained encoder layers should use lower LR (1e-4) while randomly-initialized decoder and head layers benefit from higher LR (1e-3).

**Source:** v08_kaggle_run_01/03_Model_Architecture_and_Training_Review.md

---

## 4. Data Pipeline Weaknesses

### W14: Data Leakage Not Verified — MEDIUM

No explicit assertion cell verifies zero overlap between train/val/test image paths. Given the history:

- vK.3 and vK.7.5 had a data leakage CSV bug (`TRAIN_CSV = "test_metadata.csv"`)
- v8's leakage check was path-level only (no near-duplicate or source-family checks)

vK.10.5 uses a different splitting approach but includes no verification. A simple path overlap check + pHash near-duplicate detection would add credibility with ~15 lines of code.

**Source:** Audit_vK.3_run_01, Audit10/03_vK3_Run_Audit.md

---

### W15: CASIA Dataset Limitations — MEDIUM

- CASIA ground truth masks have known annotation noise (imprecise boundaries)
- Class imbalance between splicing and copy-move samples is not documented
- No cross-dataset evaluation to verify generalization (CASIA-only training and testing)
- Authentic mask handling is assumed (all-zero masks fabricated without verifying actual Au/ mask files)
- Mask binarization (`mask > 0`) is applied without inspecting raw mask value distributions

**Source:** v08_kaggle_run_01/02_Data_Pipeline_Review.md, Audit10/06_Roast.md

---

## 5. Weakness Summary Table

| ID | Weakness | Severity | Category | Fix Effort |
|---|---|---|---|---|
| W1 | No pretrained encoder | CRITICAL | Architecture | Medium |
| W2 | RGB-only input | CRITICAL | Architecture | Medium |
| W3 | No edge supervision | HIGH | Architecture | Easy |
| W4 | No attention mechanisms | MEDIUM | Architecture | Medium |
| W5 | Fixed threshold at 0.5 | CRITICAL | Evaluation | Very Low |
| W6 | No robustness testing | CRITICAL | Evaluation | Medium |
| W7 | No Grad-CAM | HIGH | Evaluation | Medium |
| W8 | No forgery-type breakdown | MEDIUM | Evaluation | Low |
| W9 | No mask-size stratification | MEDIUM | Evaluation | Low |
| W10 | No pixel-level AUC | MEDIUM | Evaluation | Low |
| W11 | Aspect ratio destruction | MEDIUM | Training | Low |
| W12 | No gradient accumulation | LOW-MEDIUM | Training | Medium |
| W13 | No differential LR | LOW | Training | Low |
| W14 | Data leakage unverified | MEDIUM | Data | Very Low |
| W15 | CASIA dataset limitations | MEDIUM | Data | N/A |
