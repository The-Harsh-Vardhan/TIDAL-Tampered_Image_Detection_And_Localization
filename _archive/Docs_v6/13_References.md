# References

---

## Dataset

- **CASIA v2.0 — Splicing Detection + Localization**
  Dong, J., Wang, W., & Tan, T. (2013). CASIA Image Tampering Detection Evaluation Database.
  Kaggle: `sagnikkayalcse52/casia-spicing-detection-localization` (preferred source)
  Includes tampered images (Au_, Tp_) with corresponding pixel-level masks.

- **CASIA 2.0 Corrected Groundtruth**
  Kaggle: `divg07/casia-20-image-tampering-detection-dataset`
  GitHub: [CASIAv2-Groundtruth](https://github.com/namtpham/casia2groundtruth) — corrected mask generation scripts.

---

## Tier A — Primary Research Papers

These papers directly inform the project's problem framing, architecture choice, evaluation metrics, and robustness testing.

1. **Comprehensive Review of Deep-Learning-Based Methods for Image Forensics**
   `A_Comprehensive_Review_of_Deep-Learning-Based_Meth.pdf`
   Strong survey supporting localization framing and transfer-learning baselines.

2. **Comprehensive Evaluation of Image Forgery Detection Methods**
   `11042_2022_Article_13808.pdf`
   Survey compiling datasets, metrics (F1, IoU, AUC-ROC), and cautious evaluation methodology.

3. **EMT-Net: Image Manipulation Detection by Multiple Tampering Traces and Edge Artifact Enhancement**
   `1-s2.0-S0031320322005064-main.pdf`
   Multi-trace fusion with Swin Transformer + ResNet + CNN + edge enhancement. AUC=0.987 on NIST.

4. **ME-Net: Multi-Task Edge-Enhanced Image Forgery Localization**
   `ME - Multi-Task Edge-Enhanced for Image Forgery Localization 3647701.pdf`
   Dual-branch ConvNeXt + ResNet-50 with PSDA fusion. F1=0.905 on NIST16.

5. **TransU²-Net: Hybrid Transformer Architecture for Image Forgery Localization**
   `TransU_2_-Net_A_hybrid_Transformer_Architecture_fo.pdf`
   U2-Net with self-attention and cross-attention. 14.2% improvement on CASIA.

---

## Tier B — Secondary Research Papers

Adjacent papers useful for optional extensions (ELA, multi-stream, copy-move).

6. **Multistream Identity-Document Image Tampering Detection**
   `043018_1.pdf`
   Multi-stream fusion; useful future-work context.

7. **Enhanced Image Tampering Detection Using ELA and CNN**
   `ETASR_9593.pdf`
   ELA-based classification; supports optional ELA channel (Phase 2).

8. **Multi-Scale Weber Local Descriptor Evaluation**
   `evaluation-of-image-forgery-detection-using-multi-scale-weber-local-descriptors.pdf`
   Classical feature approach; historical context.

9. **Copy-Move Detection with Evolving Circular Domains**
   `s11042-022-12755-w.pdf`
   Category-specific; useful background for copy-move subset.

10. **Hybrid Deep Forgery Detection Model**
    `s11042-023-15475-x.pdf`
    Heavier hybrid architecture; shows stronger models exist beyond MVP.

---

## Tier C — Excluded or Weakly Relevant

Not used as primary evidence. Listed for completeness.

- `information-17-00122.pdf` — Tempered glass defect detection (industrial, off-domain)
- `Optimal_Semi-Fragile_Watermarking` — Active watermarking (different problem setting)
- `Tamper_Localisation_Using_Quantum_Fourier` — Medical image authentication (different domain)
- `Image Tempering Doc1.md`/`.pdf` — Local synthesis narrative (not primary research)
- `s11042-022-13808-w9.pdf` — Duplicate of the comprehensive evaluation survey
- `IJCRT24A5072.pdf` — Basic CNN classification (too shallow to justify segmentation baseline)
- `deep-learning-based-image-tamper-detection-techniques-a-study-IJERTV13IS020023.pdf` — Generic review (limited design detail)
- `A_Review_on_Video_Image_Authentication_a.pdf` — Older authentication review (historical only)
- `IMAGE_TAMPERING_DETECTION_A_REVIEW_OF_MULTI-TECHNI.pdf` — Broad review (background only)

---

## Reference Kaggle Notebooks

- **image-detection-with-mask.ipynb** — UNet baseline for tamper detection with segmentation masks. Validates the U-Net + BCE + Dice approach and pixel-level localization framing.
- **document-forensics-using-ela-and-rpa.ipynb** — ELA-based forensic analysis. Validates ELA preprocessing as a practical technique with grid-searchable quality parameters.

---

## Tools and Libraries

| Tool | Version | Purpose |
|---|---|---|
| PyTorch | ≥ 2.0 | Training framework |
| Segmentation Models PyTorch (SMP) | ≥ 0.3.3 | U-Net + ResNet34 encoder |
| Albumentations | ≥ 1.3 | Training augmentations |
| OpenCV (`cv2`) | ≥ 4.x | Image I/O and preprocessing |
| Weights & Biases (`wandb`) | ≥ 0.15 | Optional experiment tracking |
| Kaggle Secrets API | — | Secure W&B API key retrieval |
