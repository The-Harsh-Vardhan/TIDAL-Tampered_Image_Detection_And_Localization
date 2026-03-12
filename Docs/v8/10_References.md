# 10 — References

## Purpose

Consolidated reference list for all sources cited or consulted across the Docs8 document set.

---

## Project Internal References

### Documentation

| Document Set | Content | Role in Docs8 |
|---|---|---|
| Docs7/ (15 files) | System design documentation for v6.5 pipeline | Phase 1 baseline — what was designed |
| Audit6 Pro/ (5 files) | Critical review of design assumptions and gaps | Phase 2 critique — what was challenged |
| Audit 6.5 Notebook/ (9 files) | Run01 training run audit | Phase 3 evidence — what actually happened |
| Docs8/ (this set) | Evolution documentation and v8 blueprint | Synthesis and plan |

### Notebooks

| Notebook | Role |
|---|---|
| v6-5-tampered-image-detection-localization-run-01.ipynb | Run01 — first training run, source of all empirical evidence |
| tamper_detection_v6_kaggle.ipynb | Previous version (v6), referenced in Audit6 Pro consistency notes |
| tamper_detection_v6_colab.ipynb | Colab variant of v6 |

---

## Datasets

| Dataset | Citation | Use |
|---|---|---|
| CASIA v2.0 | Dong, J., Wang, W., & Tan, T. (2013). CASIA Image Tampering Detection Evaluation Database. IEEE China Summit & International Conference on Signal and Information Processing. | Primary training/evaluation dataset |
| Coverage | Wen, B., Zhu, Y., Subramanian, R., Ng, T.T., Shen, X., & Winkler, S. (2016). COVERAGE - A Novel Database for Copy-Move Forgery Detection. IEEE ICIP. | Potential cross-dataset evaluation (Tier 4 experiment) |
| CoMoFoD | Tralic, D., Zupancic, I., Grgic, S., & Grgic, M. (2013). CoMoFoD - New Database for Copy-Move Forgery Detection. ELMAR. | Potential cross-dataset evaluation |

---

## Architectures

| Architecture | Citation | Relevance |
|---|---|---|
| U-Net | Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. | Current architecture |
| ResNet | He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR. | Current encoder |
| DeepLabV3+ | Chen, L.C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV. | Planned comparison (Tier 2 experiment) |
| EfficientNet | Tan, M., & Le, Q.V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML. | Planned encoder comparison |
| SegFormer | Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., & Luo, P. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS. | Future architecture consideration (Tier 3) |

---

## Loss Functions & Training

| Topic | Citation | Relevance |
|---|---|---|
| Dice Loss | Milletari, F., Navab, N., & Ahmadi, S.A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV. | Current loss component |
| Focal Loss | Lin, T.Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal Loss for Dense Object Detection. ICCV. | Potential loss replacement (P2 experiment) |
| Tversky Loss | Salehi, S.S.M., Erdogmus, D., & Ghasemi, A. (2017). Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks. MLMI Workshop. | Potential loss variant |
| AdamW | Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR. | Current optimizer |
| Cosine Annealing | Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR. | Scheduler option |

---

## Image Forensics

| Topic | Citation | Relevance |
|---|---|---|
| SRM Filters | Fridrich, J., & Kodovsky, J. (2012). Rich Models for Steganalysis of Digital Images. IEEE TIFS. | Forensic input stream (Tier 3 experiment) |
| ELA | Krawetz, N. (2007). A Picture's Worth... Digital Image Analysis and Forensics. Black Hat. | Forensic input stream |
| ManTraNet | Wu, Y., AbdAlmageed, W., & Natarajan, P. (2019). ManTra-Net: Manipulation Tracing Network for Detection and Localization of Image Forgeries with Anomalous Features. CVPR. | Research alignment reference |
| MVSS-Net | Chen, X., Dong, C., Ji, J., Cao, J., & Li, X. (2021). Image Manipulation Detection by Multi-View Multi-Scale Supervision. ICCV. | Multi-stream architecture reference |
| ObjectFormer | Wang, J., Wu, Z., Chen, J., Han, X., Shrivastava, A., Lim, S.N., & Jiang, Y.G. (2022). ObjectFormer for Image Manipulation Detection and Localization. CVPR. | Transformer forensics reference |

---

## Libraries & Tools

| Library | Version (Run01) | Use |
|---|---|---|
| PyTorch | 2.x | Deep learning framework |
| Segmentation Models PyTorch (SMP) | Latest | Model architecture |
| Albumentations | Latest | Data augmentation |
| OpenCV | Latest | Image I/O and preprocessing |
| Weights & Biases (W&B) | Latest | Experiment tracking |
| scikit-learn | Latest | Metrics, stratified splitting |
| NumPy, Pandas | Latest | Data handling |

---

## Metrics & Evaluation

| Topic | Citation | Relevance |
|---|---|---|
| Boundary F1 | Perazzi, F., Pont-Tuset, J., McWilliams, B., Van Gool, L., Gross, M., & Sorkine-Hornung, A. (2016). A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation. CVPR. | Planned boundary metric (P1) |
| Grad-CAM | Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV. | Current XAI method |
