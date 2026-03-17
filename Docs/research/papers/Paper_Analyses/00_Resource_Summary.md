# External Resource Summary

This folder documents every exact file currently present in `Research Papers/`. The point is not to flatter the collection. The point is to separate resources that actually help a tampered-image detection and localization assignment from resources that merely look academic when dropped into a bibliography.

## Comparison Table

| Resource | Type | Core method | Localization relevance | Hardware fit | Alignment | Recommendation | Most useful idea |
|---|---|---|---|---|---|---|---|
| `043018_1.pdf` | Paper | Multistream texture/frequency/noise detection | Partial | Medium | Medium | Use partially for inspiration | Multi-domain forensic fusion |
| `11042_2022_Article_13808.pdf` | Survey | Evaluation survey across traditional and DL methods | Indirect | High | High | Use partially for inspiration | Metric and benchmark discipline |
| `1-s2.0-S0031320322005064-main.pdf` | Paper | EMT-Net multi-trace localization with edge enhancement | Direct | Low | High | Use partially for inspiration | Edge-aware multi-trace localization |
| `A_Comprehensive_Review_of_Deep-Learning-Based_Meth.pdf` | Survey | Deep-learning forensics review | Indirect | High | High | Use partially for inspiration | Good architecture taxonomy |
| `A_Review_on_Video_Image_Authentication_a.pdf` | Review | Historical authentication taxonomy | Low | High | Low | Do not use | Active versus passive framing only |
| `deep-learning-based-image-tamper-detection-techniques-a-study-IJERTV13IS020023.pdf` | Review | Generic tamper detection study | Low | High | Low | Do not use | Quick orientation to method families |
| `document-forensics-using-ela-and-rpa.ipynb` | Notebook | ELA plus threshold heuristic | Low | High | Medium | Use partially for inspiration | Cheap ELA baseline |
| `ETASR_9593.pdf` | Paper | ELA plus CNN classification | Partial | High | Medium | Use partially for inspiration | ELA can help small models |
| `evaluation-of-image-forgery-detection-using-multi-scale-weber-local-descriptors.pdf` | Paper | Chrominance descriptors plus SVM | Partial | High | Medium | Use partially for inspiration | CbCr channels can carry forensic signal |
| `IJCRT24A5072.pdf` | Paper | Basic CNN classification | Low | High | Low | Do not use | Only a toy lower bound |
| `Image Tempering Doc1.md` | Local note | Secondary synthesis note | Low | High | Low | Do not use | Pointer map to primary papers |
| `Image Tempering Doc1.pdf` | Local note | Duplicate PDF synthesis note | Low | High | Low | Do not use | None beyond convenience |
| `IMAGE_TAMPERING_DETECTION_A_REVIEW_OF_MULTI-TECHNI.pdf` | Review | Broad traditional-to-DL survey | Low | High | Low | Do not use | High-level taxonomy only |
| `image-detection-with-mask.ipynb` | Notebook | Dual-head U-Net for classification and masks | Direct | High | High | Use partially for inspiration | Kaggle-scale detection plus localization baseline |
| `information-17-00122.pdf` | Paper | Tempered-glass defect detection | None | Medium | Low | Do not use | Generic sample-scarce training ideas only |
| `ME - Multi-Task Edge-Enhanced for Image Forgery Localization 3647701.pdf` | Paper | Dual-branch edge-enhanced localization | Direct | Low | High | Use partially for inspiration | Edge supervision and RGB/noise fusion |
| `Optimal_Semi-Fragile_Watermarking_based_on_Maximum.pdf` | Paper | Active watermark-based tamper localization | None | Medium | Low | Do not use | None for passive forensics |
| `s11042-022-12755-w.pdf` | Paper | Copy-move specific feature matching | Partial | High | Medium | Use partially for inspiration | Per-forgery-type evaluation mindset |
| `s11042-022-13808-w9.pdf` | Duplicate paper | Duplicate of 2022 evaluation survey | Low | High | Low | Do not use | Deduplication hygiene |
| `s11042-023-15475-x.pdf` | Paper | Hybrid DCCAE and ADFC detection pipeline | Partial | Low | Medium | Use partially for inspiration | Controlled preprocessing ablations |
| `Tamper_Localisation_Using_Quantum_Fourier_Transfor.pdf` | Paper | Medical authentication with QFT signatures | None | Medium | Low | Do not use | None for passive natural-image tampering |
| `Towards Effective Image Forensics via A Novel 2401.06998v1.pdf` | Preprint-style PDF | Opaque CASIA and ELA-related document | Unknown | Unknown | Low | Do not use | Review the full text before trusting it |
| `TransU_2_-Net_A_hybrid_Transformer_Architecture_fo.pdf` | Paper | Attention-augmented U2-Net localization | Direct | Medium | High | Use partially for inspiration | Lightweight path from U-Net to attention |

## Most Useful Resources

Ranked shortlist for this assignment:

1. `image-detection-with-mask.ipynb`
   It is the closest practical implementation match for detection plus localization on Kaggle-scale hardware.
2. `1-s2.0-S0031320322005064-main.pdf`
   It is the strongest direct pointer toward robust forensic localization ideas such as multi-trace fusion and edge preservation.
3. `ME - Multi-Task Edge-Enhanced for Image Forgery Localization 3647701.pdf`
   It provides a clean example of why edge-aware and noise-aware localization matters.
4. `TransU_2_-Net_A_hybrid_Transformer_Architecture_fo.pdf`
   It shows a plausible upgrade path from U-Net without going fully off the deep end.
5. `11042_2022_Article_13808.pdf`
   It is the best source for evaluation discipline and benchmark framing.
6. `A_Comprehensive_Review_of_Deep-Learning-Based_Meth.pdf`
   It broadens architecture awareness and keeps the project from confusing generic vision with forensic vision.

## Ideas That Should Influence Project Design

### Preprocessing ideas
- ELA is worth testing, but only as an auxiliary signal. `document-forensics-using-ela-and-rpa.ipynb` and `ETASR_9593.pdf` justify an RGB-plus-ELA ablation, not an ELA-only worldview.
- Chrominance channels deserve one controlled experiment. The Weber descriptor paper is old, but its CbCr argument is still testable.
- Heavy preprocessing stacks such as the hybrid DCCAE paper should be treated as warning labels. Borrow one cheap idea at a time, not the full mess.

### Model architecture ideas
- The cleanest baseline-aligned reference is the dual-head U-Net notebook with mask prediction.
- The most credible upgrade directions are edge supervision, residual/noise side channels, and lightweight attention.
- EMT-Net and ME-Net are valuable because they explain why boundaries and multiple forensic traces matter. They are not excuses to build a giant paper replica in a notebook.
- TransU2-Net is the sane transformer-adjacent option in this set because it stays within segmentation logic instead of turning the whole project into architecture cosplay.

### Training and loss ideas
- BCE plus Dice remains a solid baseline and is already reflected in the mask notebook.
- Focal loss for the image-level head is reasonable when tampered versus authentic balance is imperfect.
- Edge-aware auxiliary losses are the most defensible next upgrade from the direct localization papers.
- Avoid cargo-cult loss stacking. Every extra term should be tied to a visible failure mode such as blurry boundaries or missed small tampered regions.

### Evaluation and robustness ideas
- The 2022 evaluation survey should influence metric reporting more than any model paper does.
- Separate image-level detection claims from pixel-level localization claims. Mixing them is sloppy and easy to abuse.
- Add degradation tests for JPEG compression and blur if the project wants to sound remotely serious about robustness.
- Where possible, report per-forgery-type behavior instead of one blended headline metric. The copy-move paper is useful mainly because it reminds you that manipulation types behave differently.

## Ignore or Low-Value Resources

These resources should not meaningfully influence the project design:

- `A_Review_on_Video_Image_Authentication_a.pdf`
  Too old and too broad. Historical taxonomy only.
- `deep-learning-based-image-tamper-detection-techniques-a-study-IJERTV13IS020023.pdf`
  Generic review with weaker signal than better surveys already present.
- `IJCRT24A5072.pdf`
  Basic CNN classification with no localization value.
- `Image Tempering Doc1.md` and `Image Tempering Doc1.pdf`
  Local derivative notes, not primary evidence.
- `IMAGE_TAMPERING_DETECTION_A_REVIEW_OF_MULTI-TECHNI.pdf`
  Broad but low-leverage compared with stronger surveys.
- `information-17-00122.pdf`
  Industrial defect detection, wrong domain.
- `Optimal_Semi-Fragile_Watermarking_based_on_Maximum.pdf`
  Active watermarking, wrong problem setting.
- `Tamper_Localisation_Using_Quantum_Fourier_Transfor.pdf`
  Medical authentication, wrong domain and wrong assumptions.
- `s11042-022-13808-w9.pdf`
  Duplicate survey file, not new evidence.
- `Towards Effective Image Forensics via A Novel 2401.06998v1.pdf`
  Too opaque locally to trust as a design influence without proper extraction and review.

## Bottom Line

The resource folder is not useless, but it is noisy. The project should be shaped primarily by:

- `image-detection-with-mask.ipynb` for a runnable Kaggle-scale baseline
- `1-s2.0-S0031320322005064-main.pdf`, `ME - Multi-Task Edge-Enhanced for Image Forgery Localization 3647701.pdf`, and `TransU_2_-Net_A_hybrid_Transformer_Architecture_fo.pdf` for upgrade directions
- `11042_2022_Article_13808.pdf` and `A_Comprehensive_Review_of_Deep-Learning-Based_Meth.pdf` for evaluation and literature framing

Everything else is either secondary inspiration, task-specific background, or dead weight.
