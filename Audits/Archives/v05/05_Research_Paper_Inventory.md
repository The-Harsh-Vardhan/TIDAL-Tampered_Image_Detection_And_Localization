# Research Paper Inventory

This inventory covers every PDF under `Research Papers/` and `Research Papers/More Research Papers/`, plus the local synthesis note `Research Papers/Image Tempering Doc1.md`.

| File | Inferred title | Evidence tier | Topic | Relevance to this project | Key takeaway | Supports current design? | Contradicts current design? | Use in final audit? |
|---|---|---|---|---|---|---|---|---|
| `Research Papers/043018_1.pdf` | Identity-document image tampering detection with multistream networks | B | Domain-specific tamper detection | Adjacent only | Useful future-work context for multi-stream fusion | Partial | No | Secondary |
| `Research Papers/A_Review_on_Video_Image_Authentication_a.pdf` | Review of video and image authentication techniques | C | Older authentication review | Historical background only | Useful taxonomy, weak implementation guidance | Partial | No | Secondary |
| `Research Papers/deep-learning-based-image-tamper-detection-techniques-a-study-IJERTV13IS020023.pdf` | Deep-learning image tamper detection study | C | Generic review | Limited support | Confirms DL trend, adds little design detail | Partial | No | Secondary |
| `Research Papers/ETASR_9593.pdf` | Enhanced image tampering detection using ELA and CNN | B | ELA-based classification | Supports optional ELA only | ELA can be useful, but this is not a localization baseline | Partial | No | Secondary |
| `Research Papers/IJCRT24A5072.pdf` | CNN-based image tampering detection model | C | Basic classification model | Weak evidence | Too shallow to justify the segmentation baseline | No | No | Exclude |
| `Research Papers/Image Tempering Doc1.pdf` | Local synthesis PDF | C | Local narrative artifact | Secondary only | Not a primary source paper | No | No | Exclude |
| `Research Papers/IMAGE_TAMPERING_DETECTION_A_REVIEW_OF_MULTI-TECHNI.pdf` | Review of multi-technique image tampering detection | C | Broad review | Limited support | Background only | Partial | No | Secondary |
| `Research Papers/information-17-00122.pdf` | Tempered-glass defect detection | C | Industrial defect detection | Not relevant | Off-domain and should not support forgery-localization claims | No | Yes | Exclude |
| `Research Papers/Tamper_Localisation_Using_Quantum_Fourier_Transfor.pdf` | Medical image authentication with QFT signatures | C | Medical authentication | Adjacent only | Different domain and problem setting | No | Yes | Exclude |
| `Research Papers/More Research Papers/1-s2.0-S0031320322005064-main.pdf` | Image manipulation detection by multiple tampering traces and edge artifact enhancement | A | Direct localization | Highly relevant | Frontier systems benefit from multiple traces and edge enhancement | Partial | Partial | Primary |
| `Research Papers/More Research Papers/11042_2022_Article_13808.pdf` | Comprehensive evaluation of image forgery detection methods | A | Strong survey | Highly relevant | Supports datasets, metrics, and cautious evaluation | Yes | No | Primary |
| `Research Papers/More Research Papers/A_Comprehensive_Review_of_Deep-Learning-Based_Meth.pdf` | Comprehensive review of deep-learning image forensics | A | Strong survey | Highly relevant | Supports localization framing and transfer-learning baselines | Yes | No | Primary |
| `Research Papers/More Research Papers/evaluation-of-image-forgery-detection-using-multi-scale-weber-local-descriptors.pdf` | Multi-scale Weber descriptor evaluation | B | Classical feature approach | Adjacent | Historical context, not direct support for the notebook baseline | Partial | No | Secondary |
| `Research Papers/More Research Papers/ME - Multi-Task Edge-Enhanced for Image Forgery Localization 3647701.pdf` | ME: multi-task edge-enhanced image forgery localization | A | Direct localization | Highly relevant | Strong support for edge-aware future work | Partial | Partial | Primary |
| `Research Papers/More Research Papers/Optimal_Semi-Fragile_Watermarking_based_on_Maximum.pdf` | Semi-fragile watermarking for tamper localization | C | Active authentication | Adjacent only | Different problem setting from passive CASIA localization | No | Yes | Exclude |
| `Research Papers/More Research Papers/s11042-022-12755-w.pdf` | Copy-move detection with evolving circular domains | B | Copy-move-specific detection | Category-specific | Useful background, not a general segmentation baseline | Partial | No | Secondary |
| `Research Papers/More Research Papers/s11042-022-13808-w9.pdf` | Duplicate of the comprehensive evaluation survey | C | Duplicate survey copy | Duplicate | Use the non-duplicate file instead | Yes | No | Duplicate only |
| `Research Papers/More Research Papers/s11042-023-15475-x.pdf` | Hybrid deep forgery detection model | B | Heavier hybrid architecture | Adjacent | Shows stronger hybrid models exist beyond the MVP | Partial | Partial | Secondary |
| `Research Papers/More Research Papers/TransU_2_-Net_A_hybrid_Transformer_Architecture_fo.pdf` | TransU2-Net hybrid transformer architecture | A | Advanced direct localization | Highly relevant | Transformer hybrids are viable future work, not MVP requirements | Partial | Partial | Primary |
| `Research Papers/Image Tempering Doc1.md` | Local synthesis note | C | Local narrative artifact | Secondary only | Useful for local context, not as primary evidence | No | No | Exclude |

## Inventory Summary

- **Primary evidence:** Tier A papers support the overall problem framing, localization-first formulation, overlap metrics, and robustness testing.
- **Secondary evidence:** Tier B papers are useful for optional extensions such as ELA or stronger hybrid architectures.
- **Exclude as primary evidence:** Tier C items are off-domain, duplicated, too weak, or not primary research.
