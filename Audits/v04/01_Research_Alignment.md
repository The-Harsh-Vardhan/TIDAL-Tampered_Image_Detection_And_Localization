# Research Alignment

This report evaluates how well `Docs4/` aligns with the research materials stored in `Research Papers/` and `Research Papers/More Research Papers/`. It reviews all 19 PDFs plus `Research Papers/Image Tempering Doc1.md`.

## Evidence-Tiering

| Tier | Meaning | Repository examples | How to use it |
|---|---|---|---|
| Tier A | Directly relevant tamper-detection or tamper-localization papers, or strong surveys | `A_Comprehensive_Review_of_Deep-Learning-Based_Meth.pdf`, `11042_2022_Article_13808.pdf`, `1-s2.0-S0031320322005064-main.pdf`, `ME - Multi-Task Edge-Enhanced for Image Forgery Localization 3647701.pdf`, `TransU_2_-Net_A_hybrid_Transformer_Architecture_fo.pdf` | Primary evidence for architecture, evaluation, and future-work comparisons |
| Tier B | Adjacent but still useful papers: classification-only work, category-specific papers, older reviews, domain-specific tamper studies | `043018_1.pdf`, `ETASR_9593.pdf`, `s11042-022-12755-w.pdf`, `evaluation-of-image-forgery-detection-using-multi-scale-weber-local-descriptors.pdf` | Secondary support and context only |
| Tier C | Weakly relevant, off-target, duplicate, active-authentication, or unrelated papers | `information-17-00122.pdf`, `Optimal_Semi-Fragile_Watermarking_based_on_Maximum.pdf`, `Tamper_Localisation_Using_Quantum_Fourier_Transfor.pdf`, `IJCRT24A5072.pdf`, `Image Tempering Doc1.pdf`, `Image Tempering Doc1.md` | Do not let these override stronger evidence |

Additional caveats:
- `11042_2022_Article_13808.pdf` and `s11042-022-13808-w9.pdf` are effectively duplicate copies of the same survey.
- `Image Tempering Doc1.md` and `Image Tempering Doc1.pdf` are local synthesis artifacts, not primary source papers.

## Design Decisions Supported by Research

- Treating the problem as pixel-level localization is well supported. The strongest surveys and direct localization papers in the repository consistently frame forgery localization as a dense prediction task.
- Using `CASIA v2.0` for splicing and copy-move analysis is supported by the survey literature and by papers that still benchmark on CASIA-like forgery corpora.
- Reporting overlap-based metrics such as F1 and IoU is well aligned with the survey papers and direct localization work.
- Testing robustness against post-processing is supported by both surveys and stronger direct papers. Compression, noise, blur, and resize are all reasonable degradations for a bonus evaluation path.
- Keeping forensic-feature add-ons optional is supported. The paper set contains ELA-based classification work, edge-enhanced localization work, residual- or frequency-oriented methods, and transformer hybrids. That makes `Docs4` reasonable in treating ELA and SRM as optional extensions rather than baseline obligations.
- The engineering choice to stay with a relatively lightweight pretrained segmentation model is practical for Colab and is consistent with the broader transfer-learning pattern described in the review papers, even when the papers themselves often explore more complex networks.

## Design Decisions Not Strongly Supported

- The baseline `smp.Unet + ResNet34` is credible, but it is not especially research-ambitious relative to the strongest papers in the repository. Tier A papers trend toward multi-trace fusion, edge enhancement, or transformer hybrids.
- `max(prob_map)` as the image-level score is a pragmatic engineering shortcut. It is not strongly grounded in the research materials that focus more on localization quality or on dedicated detection/classification designs.
- Explainability remains weak. The paper set does not push Docs4 toward a mandatory explainability method, but the current feature-map inspection does not amount to a strong interpretability story.
- Generated tampering coverage is not part of the current dataset or notebook plan. Some survey material discusses GAN or synthetic manipulations, but Docs4 remains focused on classical CASIA categories.

## Docs4 vs Research Contradictions

- There is no hard research contradiction against the MVP baseline itself. A simple pretrained U-Net is a defensible engineering choice for the assignment.
- The strongest contradiction risk is evidentiary overreach: the repository contains some active-authentication, watermarking, medical, and off-domain papers that do **not** directly support a passive CASIA segmentation pipeline. Audit conclusions should not treat the entire `Research Papers/` folder as uniform support.
- Tier A papers show that edge-aware and multi-trace models are stronger research baselines than plain U-Net variants. This does not make Docs4 wrong, but it does mean the project should be described as a credible baseline rather than a frontier-aligned design.
- The local synthesis note `Image Tempering Doc1.md` contains useful narrative framing, but it should not be used as primary evidence because it mixes interpretation, citations, and unrelated claims.

## Research Ideas Worth Keeping Out of MVP

- Edge-enhanced multitask localization heads
- Multi-trace fusion using RGB, residual, edge, and frequency cues
- Transformer hybrids such as TransU2-Net
- ELA or SRM alternate-input experiments that break the cheap pretrained RGB path
- Generated/deepfake tampering coverage, which would require different data
- Active watermarking or authentication pipelines, which are a different problem setting

These are valuable future-work directions, but pushing them into the MVP would add complexity without improving assignment fit.

## Research Relevance Caveats

- The paper set is mixed in venue quality, scope, and rigor. Some entries are high-signal, but others are narrow, generic, or weakly related.
- Several papers are classification-only, while this project is primarily localization-driven.
- Some papers are domain-specific: identity documents, medical authentication, or copy-move-only detection. Those results should not be generalized directly to CASIA v2.0 natural-image localization.
- `information-17-00122.pdf` is about tempered-glass defect detection and should be excluded from image forgery reasoning.
- Active watermarking/authentication papers are adjacent at best and should not be used to justify the passive deep-learning MVP.

See [04_Research_Paper_Inventory.md](04_Research_Paper_Inventory.md) for the full per-paper inventory.
