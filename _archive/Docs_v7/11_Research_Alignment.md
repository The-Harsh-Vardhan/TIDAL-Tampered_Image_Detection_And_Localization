# Research Alignment

This document ties the project's design decisions to specific research papers and explains where the system sits relative to the research frontier. The project is a **baseline aligned with assignment constraints**, not a frontier-research system.

---

## Evidence Tiering

| Tier | Meaning | Papers |
|---|---|---|
| **A** | Directly relevant tamper-detection/localization papers or strong surveys | Comprehensive DL Review (P15), Comprehensive Evaluation Survey (P14), EMT-Net (P13), ME-Net (P17), TransU²-Net (P21) |
| **B** | Adjacent but useful — classification-only, category-specific, older reviews | Multistream ID Networks (P6), Enhanced ELA+CNN (P7), Copy-move Circular Domains (P19), Multi-scale Weber (P16), Hybrid Deep Forgery Model (P10h) |
| **C** | Weakly relevant — off-domain, duplicates, active authentication | QFT Medical Auth (P10), Semi-Fragile Watermarking (P18), Tempered Glass Defects (off-domain) |

---

## Design Decisions Supported by Research

### 1. Pixel-Level Forgery Localization

**Decision:** Treat tamper detection as a dense prediction (segmentation) task.

**Research support:** Tier A surveys (P14, P15) consistently frame forgery localization as dense prediction. Direct localization papers (P13, P17, P21) all output pixel-level probability maps. The reference notebook (`image-detection-with-mask.ipynb`) confirms this by outputting segmentation masks.

### 2. U-Net + Pretrained Encoder

**Decision:** Use `smp.Unet` with ResNet34 pretrained on ImageNet.

**Research support:** Transfer learning from ImageNet-pretrained encoders is well-established (P14, P15). U-Net is proven for dense prediction. P4 (U-Net Mixed Tampering) achieves ~95% accuracy on CASIA v2 using a U-Net variant. The reference notebook also implements a U-Net.

**Honest positioning:** The `smp.Unet + ResNet34` baseline is simpler than frontier models:
- **P13 (EMT-Net):** Multi-trace fusion with Swin Transformer + ResNet + CNN + Edge Enhancement (AUC=0.987 on NIST)
- **P17 (ME-Net):** Dual-branch ConvNeXt + ResNet-50 with PSDA fusion (F1=0.905 on NIST16)
- **P21 (TransU²-Net):** U2-Net with self-attention and cross-attention (14.2% improvement on CASIA)

The baseline stays simpler because it is appropriate for T4 constraints and the assignment scope.

### 3. Overlap Metrics (F1, IoU)

**Decision:** Pixel-F1 as primary metric, Pixel-IoU as secondary.

**Research support:** F1 and IoU are standard across all Tier A papers. P13 and P17 report AUC-ROC, F1, and IoU. P14 compiles results tables using these metrics. They directly measure spatial overlap between predicted and ground-truth tampered regions.

### 4. BCE + Dice Loss

**Decision:** Combined BCE + Dice loss (equal weight, smooth=1.0).

**Research support:** Class imbalance is well-known in forgery localization (tampered pixels often < 5% of image area). Dice loss provides stronger gradients for small regions. The reference notebook uses `0.5 * BCE + 0.5 * DiceLoss`.

### 5. Robustness Against Post-Processing

**Decision:** Test under 8 degradation conditions with no retraining or threshold adaptation.

**Research support:**
- **P13 (EMT-Net):** Robust to JPEG quality ≥50 and Gaussian blur up to kernel 15
- **P17 (ME-Net):** Includes systematic degradation testing
- The assignment lists robustness testing as a bonus criterion

### 6. Optional ELA Preprocessing

**Decision:** ELA as optional 4th input channel (Phase 2).

**Research support:** P7 (Enhanced ELA+CNN, 96.21% accuracy on CASIA v2.0) demonstrates that ELA amplifies compression artifacts invisible in RGB. The `document-forensics-using-ela-and-rpa.ipynb` reference notebook validates ELA as a practical technique.

---

## Design Decisions Not Strongly Supported by Research

### 1. Top-k Mean for Image-Level Detection

A pragmatic engineering heuristic. Research papers use dedicated classification heads (P2, reference notebook). This limitation is documented; a dual-task head is Phase 2 future work.

### 2. No Edge Supervision

Edge-aware models are a significant trend:
- **P13 (EMT-Net):** Edge Artifact Enhancement module
- **P17 (ME-Net):** EEPA edge enhancement module

The baseline omits edge supervision as a scope decision.

### 3. No Multi-Domain Feature Fusion

The baseline uses RGB-only input. Research shows that multi-domain fusion (RGB + noise + frequency + edge) improves performance:
- **P6:** Texture + frequency + noise branches
- **P13:** Swin Transformer + ResNet + CNN (multiple trace types)
- **P17:** ConvNeXt (RGB edge) + ResNet-50 (noise)

Multi-domain fusion is future work, with ELA (Phase 2) and SRM (Phase 3) as stepping stones.

---

## Citation Map: Design Choices → Research Papers

| Design Choice | Supporting Papers | Tier |
|---|---|---|
| Segmentation-based localization | P14, P15, P13, P17, P21 | A |
| U-Net architecture | P4, P14, P15, reference notebook | A/B |
| ImageNet transfer learning | P14, P15 | A |
| BCE + Dice loss | Reference notebook, P14 | A |
| Pixel-F1 / IoU metrics | P13, P14, P17 | A |
| JPEG robustness testing | P13, P17 | A |
| ELA preprocessing (optional) | P7, ELA reference notebook | B |
| Top-k mean image-level (heuristic) | — (pragmatic, not research-supported) | — |
| Edge supervision (not implemented) | P13, P17, P21 | A |
| SRM noise features (not implemented) | P13, P15, P17 | A |

---

## Research Ideas for Future Work

| Technique | Source | Complexity | Potential Impact |
|---|---|---|---|
| Edge supervision loss | P13, P17 | Medium | Sharper boundary detection |
| Self-attention in encoder | P21 | Medium | Better global context |
| Cross-attention in skip connections | P21 | Medium | Filtered decoder features |
| SRM noise maps | P13, P15, P17 | Medium | Noise-domain forensic cues |
| Multi-stream architecture | P6, P13, P17 | High | Multi-domain feature fusion |
| Transformer hybrids | P13, P21 | High | State-of-the-art performance |
| Dual-task classification head | P2, reference notebook | Low | Better image-level detection |
| Focal Loss for classification | Reference notebook | Low | Better class imbalance handling |
| CLAHE preprocessing | P20 | Low | Enhanced subtle artifacts |

---

## Research Papers Not Applicable

| Paper | Reason |
|---|---|
| `information-17-00122.pdf` (Tempered Glass Defects) | Industrial defect detection, not image forgery |
| `Optimal_Semi-Fragile_Watermarking` (P18) | Active watermarking, different problem setting |
| `Tamper_Localisation_Using_Quantum_Fourier` (P10) | Medical image authentication, different domain |
| `Image Tempering Doc1.md`/`.pdf` | Synthesis narrative, not primary research |
| `s11042-022-13808-w9.pdf` | Duplicate of the comprehensive evaluation survey |
| `A_Review_on_Video_Image_Authentication_a.pdf` | Historical authentication review |

---

## Summary

The project implements a credible segmentation-based baseline for image tamper localization:

- **Supported** by research for core formulation (pixel-level localization, U-Net, overlap metrics, robustness testing)
- **Simpler** than frontier models (no edge supervision, no multi-trace fusion, no transformer attention)
- **Appropriate** for assignment constraints (Kaggle/Colab T4, CASIA dataset, single-notebook delivery)
- **Extensible** toward stronger designs through documented Phase 2/3 enhancements
