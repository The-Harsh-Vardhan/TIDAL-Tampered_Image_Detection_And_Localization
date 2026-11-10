# Research Alignment

This report evaluates how well `Docs5/` and `notebooks/tamper_detection_v5.ipynb` align with the repository's research material. The conclusion is that the project is a research-informed baseline, not a frontier research implementation.

## Evidence Tiering

| Tier | Meaning | Repository examples | How to use it |
|---|---|---|---|
| A | Direct tamper-localization papers or strong surveys | `A_Comprehensive_Review_of_Deep-Learning-Based_Meth.pdf`, `11042_2022_Article_13808.pdf`, `1-s2.0-S0031320322005064-main.pdf`, `ME - Multi-Task Edge-Enhanced for Image Forgery Localization 3647701.pdf`, `TransU_2_-Net_A_hybrid_Transformer_Architecture_fo.pdf` | Primary support for architecture, evaluation, and future-work positioning |
| B | Adjacent but useful papers | `043018_1.pdf`, `ETASR_9593.pdf`, `s11042-022-12755-w.pdf`, `evaluation-of-image-forgery-detection-using-multi-scale-weber-local-descriptors.pdf`, `s11042-023-15475-x.pdf` | Secondary support only |
| C | Weakly relevant, off-target, duplicate, or active-authentication papers | `information-17-00122.pdf`, `Optimal_Semi-Fragile_Watermarking_based_on_Maximum.pdf`, `Tamper_Localisation_Using_Quantum_Fourier_Transfor.pdf`, `IJCRT24A5072.pdf`, `Image Tempering Doc1.pdf`, `Image Tempering Doc1.md`, `s11042-022-13808-w9.pdf` | Do not use as primary design evidence |

## Design Decisions Supported by Research

- Treating the problem as pixel-level localization is strongly supported by the Tier A surveys and direct localization papers.
- A segmentation baseline with a pretrained encoder is consistent with the transfer-learning pattern described in the survey literature and is appropriate for a Colab-constrained assignment.
- Overlap metrics such as Pixel-F1 and IoU are standard and are correctly emphasized in the project docs.
- BCE + Dice is a defensible loss choice for highly imbalanced tamper masks.
- Robustness testing against compression, blur, noise, and resize is supported by both the literature and the assignment framing.
- Optional forensic feature extensions such as ELA and SRM are research-motivated and correctly kept out of the MVP path.

## Design Decisions Not Strongly Supported

- The image-level decision path uses a top-k mean over pixel probabilities. That is a pragmatic engineering heuristic, but it is not a strongly research-backed substitute for a dedicated classification head.
- The baseline does not include edge supervision, multi-trace fusion, or transformer-style global context, all of which appear in the strongest papers in the repository.
- Explainability remains lightweight. Grad-CAM and overlays support model inspection, but the research set does not justify treating this as a strong explainability story.

## Research Positioning

The correct way to position the current system is:

- technically sound baseline
- appropriate for a single Colab notebook and T4 GPU
- aligned with the assignment scope
- below the research frontier represented by multi-trace, edge-enhanced, and transformer-based models

That positioning is now reflected well in `Docs5/11_Research_Alignment.md`.

## Research Caveats

- The repository paper set is mixed in quality and relevance. It should not be treated as uniform support for every design choice.
- Some papers are classification-only, domain-specific, active-authentication-oriented, or unrelated to passive natural-image tamper localization.
- `Image Tempering Doc1.md` and `Image Tempering Doc1.pdf` are local synthesis artifacts and should not be treated as primary evidence.
- `s11042-022-13808-w9.pdf` appears to duplicate `11042_2022_Article_13808.pdf`.

See [05_Research_Paper_Inventory.md](05_Research_Paper_Inventory.md) for the full file-by-file inventory.
