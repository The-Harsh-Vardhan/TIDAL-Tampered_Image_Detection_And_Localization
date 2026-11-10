# 05 — External Resource Integration

## Purpose

Evaluate each resource in `Docs_External_Resources/` for alignment with the assignment, technical usefulness, and whether it should influence the v9 implementation. Only ideas that are technically justified, feasible, and beneficial are adopted.

---

## Resource Assessment

### Tier 1: Directly Useful — Adopt Ideas

#### Resource 14: image-detection-with-mask.ipynb

**What it is:** A Kaggle notebook implementing a dual-head U-Net for both image classification and mask prediction on CASIA-style data.

**Alignment with assignment:** High. Directly solves both detection and localization in a single model.

**Key technique adopted:** **Dual-task architecture with learned classification head.** This notebook demonstrates that a single U-Net can predict both `cls_logits` and `seg_logits` in one forward pass, exactly what Audit8 Pro demanded as a replacement for `max(prob_map)`.

**How it influences v9:**
- The dual-task head design is adopted for v9 (see 03_Feasible_Improvements.md §1.1)
- The multi-task loss pattern (seg_loss + λ × cls_loss) is used as the starting point
- The focal loss for classification branch is noted as an option but deferred

**What is NOT adopted:** The notebook's messy engineering (duplicated sections, inconsistent code). v9 takes the architectural idea, not the implementation.

---

#### Resource 03: EMT-Net (1-s2.0-S0031320322005064-main.pdf)

**What it is:** A research paper on multi-trace image manipulation detection using RGB artifacts, noise cues, and edge enhancement.

**Alignment with assignment:** High for localization. The architecture is too complex for Colab but the ideas are directly relevant.

**Key techniques adopted:**
1. **Edge-aware supervision.** EMT-Net emphasizes that manipulation boundaries are the most informative forensic signal. v9 adopts an auxiliary edge loss to strengthen boundary prediction.
2. **Robustness evaluation under degradation.** EMT-Net evaluates under JPEG, blur, and noise. v9's robustness suite already does this.

**What is NOT adopted:**
- Full multi-branch architecture (RGB + global noise + local noise branches). Too complex for Colab and beyond assignment scope.
- Transformer-style global modeling. Deferred to future research.

---

#### Resource 16: ME-Net (Multi-Task Edge-Enhanced)

**What it is:** Dual-branch architecture with ConvNeXt for RGB and ResNet-50 for noise, fused with attention, with edge-enhanced decoding.

**Alignment with assignment:** High conceptually. The architecture is too heavy for direct implementation.

**Key techniques adopted:**
1. **Edge enhancement in decoding.** Reinforces the decision to add auxiliary edge loss in v9.
2. **Dual-domain evidence (RGB + noise).** Supports the decision to add ELA as a lightweight forensic channel instead of a full noise branch.

**What is NOT adopted:**
- ConvNeXt + ResNet-50 dual backbone (exceeds Colab memory constraints)
- PSDA attention fusion module (implementation complexity too high for marginal benefit)

---

#### Resource 23: TransU2-Net

**What it is:** Hybrid Transformer-U2Net architecture for splicing forgery detection with self-attention in encoder and cross-attention in skip connections.

**Alignment with assignment:** High conceptually. Demonstrates that attention can help localization.

**Key insight recorded for future work:**
- Attention in skip connections is a cleaner upgrade path than replacing the entire encoder with a transformer.
- This approach could be explored in v10+ if v9 results suggest that long-range context is the bottleneck.

**What is NOT adopted for v9:**
- Full TransU2-Net architecture (implementation complexity, hyperparameter sensitivity)
- Self-attention in encoder (changes training dynamics significantly)

**Decision:** Deferred to future research. Recorded in [08_Future_Research_Directions.md](08_Future_Research_Directions.md).

---

#### Resource 02: Evaluation Survey (11042_2022_Article_13808.pdf)

**What it is:** Comprehensive survey on evaluation methodologies for image tampering detection.

**Alignment with assignment:** High for evaluation discipline.

**Key influence on v9:**
- Reinforces the importance of separate pixel-level and image-level evaluation
- Supports the decision to add Boundary F1 and PR curves
- Validates the approach of reporting per-forgery-type metrics as standard practice

**Direct adoption:** No code adopted, but evaluation methodology improved based on survey best practices.

---

#### Resource 04: Comprehensive DL Review (A_Comprehensive_Review_of_Deep-Learning-Based_Meth.pdf)

**What it is:** Deep learning forensics review with architecture taxonomy.

**Alignment with assignment:** High for literature framing.

**Key influence on v9:**
- Provides vocabulary and framing for architecture justification in notebook documentation
- Confirms that U-Net-based approaches are a recognized baseline in the field
- Distinguishes between generic segmentation and forensic-specific architectures

**Direct adoption:** No code adopted. Influences notebook documentation narrative.

---

### Tier 2: Partially Useful — Specific Ideas Extracted

#### Resource 07: ELA Notebook (document-forensics-using-ela-and-rpa.ipynb)

**What it is:** Simple Kaggle notebook implementing ELA with threshold-based classification.

**Alignment with assignment:** Medium. No localization, but the ELA technique is relevant.

**Key technique extracted:** The ELA computation method (re-save JPEG, compute absolute difference). This informs the ELA auxiliary channel implementation in v9.

**What is NOT adopted:** The threshold-based classification approach. ELA is used as an input feature, not as a standalone detector.

---

#### Resource 08: ELA + CNN (ETASR_9593.pdf)

**What it is:** Paper combining ELA with CNN for tamper classification.

**Alignment with assignment:** Medium. Confirms ELA can help small models.

**Key insight:** ELA provides meaningful forensic signal even for simple architectures, supporting the decision to add it as a 4th input channel in v9.

---

#### Resource 09: Weber Local Descriptors

**What it is:** Paper on chrominance-based (CbCr) forgery detection using multi-scale Weber descriptors.

**Alignment with assignment:** Medium. The CbCr idea is interesting but lower priority.

**Key insight:** Chrominance channels can carry forensic signal invisible in RGB. This is recorded as a deferred improvement (CbCr channels, see 03_Feasible_Improvements.md §2.3).

---

#### Resource 01: Multistream Texture/Frequency/Noise (043018_1.pdf)

**What it is:** Multi-domain forensic fusion approach combining texture, frequency, and noise features.

**Alignment with assignment:** Medium. Too complex for direct adoption but the multi-domain concept is relevant.

**Key insight:** Confirms the theoretical value of combining multiple forensic evidence domains. Supports the ELA decision as a first step toward multi-domain input.

---

#### Resource 18: Copy-Move Feature Matching (s11042-022-12755-w.pdf)

**What it is:** Copy-move specific detection using feature matching techniques.

**Alignment with assignment:** Medium. Copy-move is the project's weakest area.

**Key insight:** Per-forgery-type evaluation is essential because manipulation types behave fundamentally differently. Reinforces the v9 decision to track copy-move metrics separately.

---

#### Resource 20: Hybrid DCCAE Pipeline (s11042-023-15475-x.pdf)

**What it is:** Complex detection pipeline with DCCAE and ADFC components.

**Alignment with assignment:** Medium. The pipeline itself is too heavy but controlled ablation methodology is useful.

**Key insight:** The value of controlled preprocessing ablations. Supports the v9 augmentation ablation experiment.

---

### Tier 3: Not Useful — Do Not Adopt

| Resource | Reason for Exclusion |
|---|---|
| Resource 05: Video Image Authentication Review | Historical taxonomy, wrong domain (video authentication) |
| Resource 06: Generic Tamper Detection Study (IJERT) | Too generic, weaker than better surveys already present |
| Resource 10: Basic CNN Classification (IJCRT) | Toy classification model, no localization value |
| Resource 11: Image Tempering Doc1.md | Local derivative note, not primary evidence |
| Resource 12: Image Tempering Doc1.pdf | Duplicate PDF synthesis note |
| Resource 13: Multi-Technique Review | Broad but low-leverage compared to better surveys |
| Resource 15: Tempered-Glass Defect Detection | Wrong domain (industrial defect detection) |
| Resource 17: Semi-Fragile Watermarking | Active watermarking, wrong problem setting (passive forensics needed) |
| Resource 19: Duplicate Survey (s11042-022-13808-w9) | Duplicate of Resource 02 |
| Resource 21: Quantum Fourier Transform Localization | Medical authentication, wrong domain |
| Resource 22: Towards Effective Image Forensics (2401.06998v1) | Too opaque to trust; needs full extraction before use |

---

## Integration Summary

### Ideas Adopted for v9

| Source | Idea | Where in v9 |
|---|---|---|
| Resource 14 (mask notebook) | Dual-task classification + segmentation head | Model architecture |
| Resource 03 (EMT-Net) | Auxiliary edge loss for boundary quality | Loss function |
| Resource 07/08 (ELA notebooks/papers) | ELA as auxiliary input channel | Data pipeline |
| Resource 16 (ME-Net) | Edge supervision reinforcement | Loss function (confirms edge loss decision) |
| Resource 02 (Evaluation survey) | Separate pixel/image evaluation, Boundary F1 | Evaluation methodology |
| Resource 04 (DL Review) | Architecture taxonomy for documentation framing | Notebook narrative |
| Resource 18 (Copy-move paper) | Per-forgery-type evaluation as standard | Evaluation methodology |

### Ideas Recorded for Future Work

| Source | Idea | Timeline |
|---|---|---|
| Resource 23 (TransU2-Net) | Attention in skip connections | v10+ |
| Resource 03 (EMT-Net) | Multi-branch forensic architecture | v10+ |
| Resource 16 (ME-Net) | Full noise branch with dual backbone | v10+ |
| Resource 09 (Weber) | CbCr chrominance channels | v10 (after ELA evaluation) |

### Ideas Rejected

| Source | Idea | Reason |
|---|---|---|
| Resource 21 (QFT) | Quantum Fourier Transform | Wrong domain entirely |
| Resource 17 (Watermarking) | Active watermarking | Wrong problem setting |
| Resource 15 (Glass defect) | Industrial defect methods | Wrong domain |
| Resource 22 (2401.06998) | Unclear methodology | Cannot be trusted without full review |

---

## Bottom Line

The external resources provide three concrete contributions to v9:

1. **Dual-task head** — from the Kaggle mask notebook (Resource 14)
2. **ELA auxiliary channel** — from ELA-focused notebooks and papers (Resources 07, 08)
3. **Edge supervision** — from forensic localization papers (Resources 03, 16)

Everything else is either deferred to future work (transformers, full multi-branch architectures) or excluded (wrong domain, too generic, duplicates).

The selection criterion is strict: only adopt ideas that are (a) technically justified by the project's specific failure modes, (b) feasible within Colab/Kaggle constraints, and (c) expected to measurably improve performance or evaluation credibility.
