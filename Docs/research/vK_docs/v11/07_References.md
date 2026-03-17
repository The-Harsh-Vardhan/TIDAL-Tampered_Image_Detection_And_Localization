# Docs11: References

Consolidated references for all sources cited in the Docs11 technical review.

---

## Project Internal References

| Document | Location | Content |
|---|---|---|
| Assignment Spec | `Assignment.md` | Big Vision Internship requirements |
| Docs9 Improvement Plan | `Docs/v9/` | Prior improvement planning for v9 notebook |
| Audit10-Improvements | `Audits/Audit10-Improvements/` | Cross-notebook comparison and feature gap analysis |
| v08_kaggle_run_01 Audit | `Audits/v08_kaggle_run_01/` | Detailed v8 run analysis |
| Audit v7.1 | `Audits/Audit v7.1/` | vK.7.1 notebook audit |
| Audit vK.3 run 01 | `Audits/Audit vK.3 run 01/` | vK.3 run analysis |
| Research Paper Analysis | `Research/Research_Paper_Analysis_Report.md` | Master synthesis of 21 papers |
| Paper Analyses | `Research/Paper_Analyses/` | Individual per-paper analyses |

---

## Notebooks

| Notebook | Version | Key Contribution |
|---|---|---|
| `vK.10.5 Image Detection and Localisation.ipynb` | vK.10.5 | Current best engineering (baseline for Docs11) |
| `v8-tampered-image-detection-localization-run-01.ipynb` | v8 | Best evaluation methodology and trained results |
| `v9-tampered-image-detection-localization-kaggle.ipynb` | v9 | Intermediate iteration |
| `vK.7.1 Image Detection and Localisation.ipynb` | vK.7.1 | Best documentation style |

---

## Datasets

| Dataset | Usage | Reference |
|---|---|---|
| CASIA v2.0 | Primary training/evaluation dataset | Dong, J. et al. "CASIA Image Tampering Detection Evaluation Database." IEEE ChinaSIP 2013. |
| Coverage | Referenced alternative | Wen, B. et al. "Coverage — A Novel Database for Copy-Move Forgery Detection." IEEE ICIP 2016. |
| CoMoFoD | Referenced alternative | Tralic, D. et al. "CoMoFoD — New Database for Copy-Move Forgery Detection." ELMAR 2013. |
| NIST'16 | Referenced benchmark (used by EMT-Net, ME-Net) | NIST Nimble Challenge 2016. |

---

## Research Papers (Cited in Docs11)

### Forensic Preprocessing

| ID | Paper | Key Technique | Result |
|---|---|---|---|
| P1 | "Tempered Image Detection Using ELA and CNNs" (IEEE 10444440) | ELA preprocessing + CNN | 87.75% accuracy |
| P7 | "Enhanced ELA + CNN" (ETASR_9593) | ELA + optimized CNN | 96.21% on CASIA v2.0 |
| P16 | "Evaluation of Image Forgery Detection Using Multi-scale Weber Local Descriptors" | Chrominance (YCbCr Cb/Cr) analysis | 96.52% on CASIA v2.0 |

### Architectures

| ID | Paper | Key Technique | Result |
|---|---|---|---|
| P2 | "Real or Fake?" (IEEE 10052973) | Dual-task classification + segmentation | Validated dual-head approach |
| P4 | "Deep Localization on Mixed Image Tampering" (IEEE 10652417) | U-Net + FENet + edge attention | ~95% on CASIA v2 |
| P13 | "EMT-Net" (Pattern Recognition, S0031320322005064) | Multi-trace: Swin Transformer + ResNet (SRM) + CNN + Edge | AUC=0.987 on NIST |
| P17 | "ME-Net" (3647701) | ConvNeXt + ResNet-50 + PSDA + EEPA | F1=0.905 on NIST16 |
| P21 | "TransU2-Net" (Hybrid Transformer) | U2-Net + self/cross-attention | F-measure=0.735 on CASIA |

### Surveys and Reviews

| ID | Paper | Scope |
|---|---|---|
| P14 | "Comprehensive Analyses of Image Forgery Detection" (2022, 11042_2022_Article_13808) | 34-page survey across 10+ methods and datasets |
| P15 | "A Comprehensive Review of DL-Based Methods for Image Forensics" | 39-page review of 180+ methods |

### Classical / Specialized Methods

| ID | Paper | Key Technique |
|---|---|---|
| P3 | "Robust Algorithm for Copy-Rotate-Move Detection" (IEEE 10168896) | Zernike moments + ACO |
| P6 | "ID Document Tampering Detection via Multistream Networks" (043018_1) | Texture + frequency + noise branches |

---

## Libraries and Frameworks

| Library | Version | Purpose |
|---|---|---|
| PyTorch | ≥1.12 | Deep learning framework |
| segmentation_models_pytorch (SMP) | ≥0.3.0 | Pretrained encoder-decoder architectures |
| Albumentations | ≥1.3.0 | Image augmentation with mask alignment |
| OpenCV (cv2) | ≥4.5 | Image I/O, ELA computation, color space conversion |
| kornia | ≥0.6 | Differentiable Sobel filters for edge loss |
| scikit-learn | ≥1.0 | Metrics (ROC-AUC, confusion matrix, classification report) |
| matplotlib | ≥3.5 | Visualization |
| seaborn | ≥0.12 | Confusion matrix heatmaps |
| pandas | ≥1.4 | Metadata handling and CSV export |
| wandb | ≥0.13 | Experiment tracking (optional) |
| imagehash | ≥4.3 | Perceptual hashing for near-duplicate detection |
| numpy | ≥1.21 | Numerical operations |

---

## Kaggle Resources

| Resource | ID | Description |
|---|---|---|
| CASIA v2.0 Dataset | `harshv777/casia2-0-upgraded-dataset` | Primary dataset on Kaggle |
| Kaggle Notebook (Resource 14) | `image-detection-with-mask.ipynb` | Reference dual-head UNet implementation |
