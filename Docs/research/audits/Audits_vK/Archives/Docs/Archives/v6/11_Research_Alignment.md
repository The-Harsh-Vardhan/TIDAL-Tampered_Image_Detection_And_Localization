# Research Alignment

This document ties the project's design decisions to specific research papers and explains where the system sits relative to the research frontier. The project is a **baseline aligned with assignment constraints**, not a frontier-research system.

---

## Evidence Tiering

Research papers in the repository are tiered by relevance:

| Tier | Meaning | Papers |
|---|---|---|
| **A** | Directly relevant tamper-detection/localization papers or strong surveys | Comprehensive DL Review (P15), Comprehensive Evaluation Survey (P14), EMT-Net (P13), ME-Net (P17), TransU²-Net (P21) |
| **B** | Adjacent but useful — classification-only, category-specific, older reviews | Multistream ID Networks (P6), Enhanced ELA+CNN (P7), Copy-move Circular Domains (P19), Multi-scale Weber (P16) |
| **C** | Weakly relevant — off-domain, duplicates, active authentication | QFT Medical Auth (P10), Semi-Fragile Watermarking (P18), Tempered Glass Defects (information-17-00122) |

---

## Design Decisions Supported by Research

### 1. Pixel-Level Forgery Localization

**Decision:** Treat tamper detection as a dense prediction (segmentation) task rather than image-level classification only.

**Research support:** Tier A surveys (P14, P15) consistently frame forgery localization as a dense prediction task. Direct localization papers (P13, P17, P21) all output pixel-level probability maps. The `image-detection-with-mask.ipynb` reference notebook confirms this approach by implementing a UNet that outputs both classification scores and segmentation masks.

### 2. Segmentation-Based Architecture (U-Net + Pretrained Encoder)

**Decision:** Use `smp.Unet` with ResNet34 pretrained on ImageNet.

**Research support:** Transfer learning from ImageNet-pretrained encoders is a well-established pattern in the survey literature (P14, P15). U-Net is a proven architecture for dense prediction tasks. Research paper P4 (U-Net Mixed Tampering Localization) achieves ~95% accuracy on CASIA v2 using a U-Net variant. The reference notebook `image-detection-with-mask.ipynb` also implements a U-Net architecture, further validating this choice.

**Honest positioning:** The `smp.Unet + ResNet34` baseline is simpler than frontier models in the paper repository. Tier A papers demonstrate stronger architectures:
- **P13 (EMT-Net):** Multi-trace fusion with Swin Transformer + ResNet + CNN + Edge Enhancement (AUC=0.987 on NIST)
- **P17 (ME-Net):** Dual-branch ConvNeXt + ResNet-50 with PSDA fusion (F1=0.905 on NIST16)
- **P21 (TransU²-Net):** U2-Net with self-attention and cross-attention (14.2% improvement on CASIA)

These models are more complex and are documented as future work. The MVP stays with the simpler baseline because it is appropriate for Kaggle T4 constraints and the assignment scope.

### 3. Overlap Metrics (F1, IoU)

**Decision:** Use Pixel-F1 as the primary metric and Pixel-IoU as the secondary metric.

**Research support:** F1 and IoU are the standard overlap metrics across all Tier A papers. P13 and P17 report AUC-ROC, F1, and IoU. P14 (survey) compiles results tables using F1 and IoU. These metrics directly measure the spatial overlap between predicted and ground-truth tampered regions, which is the core objective of this project.

### 4. BCE + Dice Loss

**Decision:** Combine binary cross-entropy with Dice loss for training (equal weight, smooth=1.0).

**Research support:** Class imbalance is a well-known challenge in forgery localization (tampered pixels are typically <5% of image area). Dice loss directly optimizes F1-like overlap and provides stronger gradients for small regions. The `image-detection-with-mask.ipynb` reference notebook uses `0.5 * BCE + 0.5 * DiceLoss`, confirming this as a standard combination.

### 5. Robustness Against Post-Processing

**Decision:** Test model robustness under 8 degradation conditions (JPEG compression, Gaussian noise, blur, resize, brightness, contrast, saturation, combined).

**Research support:** Post-processing robustness is a recurring evaluation criterion in research papers:
- **P13 (EMT-Net):** Robust to JPEG quality ≥50 and Gaussian blur up to kernel 15
- **P4 (FENet):** Evaluates under JPEG, blur, and noise
- **P17 (ME-Net):** Includes systematic degradation testing

The assignment also specifically lists robustness testing as a bonus criterion. The degradation suite in `06_Robustness_Testing.md` aligns with the research literature.

### 6. Optional ELA Preprocessing

**Decision:** ELA as an optional 4th input channel in Phase 2.

**Research support:** P1 (ELA-CNN Hybrid) and P7 (Enhanced ELA+CNN, 96.21% accuracy on CASIA v2.0) demonstrate that ELA can amplify compression artifacts invisible in RGB. The `document-forensics-using-ela-and-rpa.ipynb` reference notebook validates ELA as a practical forensic preprocessing technique with grid-searchable parameters.

---

## Design Decisions Not Strongly Supported by Research

### 1. Top-k Mean Probability for Image-Level Detection

The image-level detection score uses the **mean of the top-k pixel probabilities** (top 1% of pixels). This is a pragmatic engineering shortcut. Research papers focus more on localization quality or use dedicated classification heads. P2 (Dual-task Classification + Segmentation) and the `image-detection-with-mask.ipynb` reference notebook both implement separate classification heads for image-level decisions.

**Mitigation:** This limitation is documented. A dual-task classification head is listed as a Phase 2 optional enhancement.

### 2. No Edge Supervision

Edge-aware models are a significant trend in Tier A papers:
- **P13 (EMT-Net):** Edge Artifact Enhancement prevents loss of boundary clues
- **P17 (ME-Net):** EEPA edge enhancement module

The MVP does not include edge supervision. This is a deliberate simplification for the assignment scope.

### 3. No Multi-Domain Feature Fusion

The MVP uses RGB-only input. Research papers show that multi-domain fusion (RGB + noise + frequency + edge) improves performance:
- **P6:** Texture + frequency + noise branches
- **P13:** Swin Transformer (global noise) + ResNet (local noise) + CNN (RGB)
- **P17:** ConvNeXt (RGB edge) + ResNet-50 (noise)

Multi-domain fusion is future work, with ELA (Phase 2) and SRM (Phase 3) as initial steps.

---

## Research Ideas for Future Work

These techniques are supported by Tier A/B papers but are out of scope for the MVP:

| Technique | Source | Complexity | Potential Impact |
|---|---|---|---|
| Edge supervision loss | P13, P17 | Medium | Sharper boundary detection |
| Self-attention in encoder | P21 | Medium | Better global context |
| Cross-attention in skip connections | P21 | Medium | Filtered decoder features |
| SRM noise maps | P13, P15, P17 | Medium | Noise-domain forensic cues |
| Multi-stream architecture | P6, P13, P17 | High | Multi-domain feature fusion |
| Transformer hybrids | P13, P21 | High | State-of-the-art performance |
| Dual-task classification head | P2, reference notebook | Low | Better image-level detection |
| Focal Loss for classification | Reference notebook | Low | Better handling of class imbalance |
| CLAHE preprocessing | P20 | Low | Enhanced subtle artifacts |
| Chrominance (YCbCr) analysis | P16 | Low | Alternative feature domain |

---

## Research Papers Not Applicable

The following papers in the repository should **not** be used to justify design decisions for this project:

| Paper | Reason |
|---|---|
| `information-17-00122.pdf` (Tempered Glass Defects) | Industrial defect detection, not image forgery |
| `Optimal_Semi-Fragile_Watermarking` (P18) | Active watermarking, different problem setting |
| `Tamper_Localisation_Using_Quantum_Fourier` (P10) | Medical image authentication, different domain |
| `Image Tempering Doc1.md`/`.pdf` | Local synthesis artifacts, not primary research |
| `s11042-022-13808-w9.pdf` | Duplicate of the comprehensive survey |

---

## Summary

The project implements a credible segmentation-based baseline for image tamper localization. It is:

- **Supported** by the research literature for its core formulation (pixel-level localization, U-Net, overlap metrics, robustness testing)
- **Simpler** than frontier models (no edge supervision, no multi-trace fusion, no transformer attention)
- **Appropriate** for the assignment constraints (single Kaggle notebook, T4 GPU, CASIA dataset)
- **Extensible** toward stronger research-aligned designs through documented Phase 2/3 enhancements
