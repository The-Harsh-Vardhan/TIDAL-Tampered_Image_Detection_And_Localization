# 08 — Future Research Directions

## Purpose

Catalogue ideas that are interesting and technically valid but not feasible within the scope of this assignment. These are documented as evidence of research awareness and as a roadmap for post-submission exploration. This document also contains the final summary of all Docs9 decisions.

---

## Deferred Improvements (Feasible but Deprioritized)

These items were evaluated in `03_Feasible_Improvements.md` and judged technically sound but deferred due to time, complexity, or marginal expected benefit relative to approved changes.

### Loss Function Alternatives

**Focal Loss**
- Addresses class imbalance by down-weighting easy negatives
- Deferred because pos_weight + per-sample Dice already handle imbalance in v8
- If v9 results show persistent false-positive issues, Focal Loss is the next loss to try
- Implementation: Replace BCE component with `-(1-p)^γ * log(p)`, γ=2

**Tversky Loss**
- Generalizes Dice with separate α/β for false positives and false negatives
- Useful if the model systematically over-segments or under-segments
- Deferred: Dice + BCE is well-established; adding another loss variant introduces tuning cost
- Revisit if v9 shows a consistent P/R imbalance in tampered-only metrics

### Input Enhancements

**SRM Noise Residuals**
- Steganalysis-derived high-pass filters that suppress image content and expose noise patterns
- Particularly promising for copy-move detection (highlights noise inconsistencies between duplicated and original regions)
- Deferred: Requires custom SRM filter implementation and 30-channel preprocessing (3 filters × 5 kernels × RGB), significantly increasing input complexity
- **Priority for v10:** This is the single most impactful deferred item for improving copy-move F1

**CbCr Color Channels**
- YCbCr chrominance channels may reveal compression artifacts invisible in RGB
- Deferred: Adding 2 more channels (CbCr) on top of ELA makes the input 6-channel, increasing model complexity and memory usage
- Modest expected gain compared to ELA alone

### Training Strategies

**Multi-Scale Training**
- Train with randomly resized crops (e.g., 256/384/512) to improve scale invariance
- Deferred: Increases training time ~40% and requires careful batch composition
- v9 already uses 384×384 fixed size, which is adequate for CASIA v2.0

**Multi-Scale Inference (TTA)**
- Run inference at multiple scales and average predictions
- Deferred: Increases inference time 3–5× and complicates threshold calibration
- Worth trying as a post-hoc enhancement if v9 results are close to targets

**Cosine Annealing Scheduler**
- More aggressive LR decay than ReduceLROnPlateau
- Deferred: ReduceLROnPlateau is already working as intended (reactive to validation loss)
- Low-cost experiment — could be tested alongside approved changes if time permits

**Gradient Accumulation Tuning**
- Current: 4 steps. Effective batch size: 16
- Increasing to 8 steps (effective 32) may stabilize training but slows convergence
- Deferred: Not a high-priority change

### Architecture Alternatives

**EfficientNet Encoder**
- Replace ResNet34 with EfficientNet-B3/B4 for better parameter efficiency
- Deferred: ResNet34 is well-supported in SMP and has reliable ImageNet weights
- EfficientNet requires careful channel adaptation for 4-channel input (fewer community examples)

---

## Rejected Ideas (Infeasible for This Assignment)

These items were rejected due to fundamental incompatibility with the assignment scope, hardware constraints, or risk/benefit ratio.

### Transformer-Based Encoders

**SegFormer, TransU2-Net**
- Strong performance on semantic segmentation benchmarks
- Rejected because: (1) T4 GPU memory insufficient for full transformer training at 384×384, (2) limited SMP integration, (3) assignment scope is better served by well-understood CNN architectures
- **Research value:** High. Modern forensic papers (2023+) increasingly use transformer or hybrid architectures. Worth exploring in a post-assignment research context with better hardware.

### Multi-Branch Forensic Architectures

**EMT-Net (multi-stream RGB + noise + edge)**
**ME-Net (mutual enhancement between edge detection and localization)**
- State-of-the-art architectures purpose-built for image forgery detection
- Rejected because: (1) Custom implementations, no SMP support, (2) significant engineering effort (2–4 weeks), (3) risk of bugs in custom architecture, (4) not justifiable for an internship assignment
- **Research value:** Very high. These architectures address the fundamental limitations of single-stream U-Net for forensic tasks.

### Full Multi-Branch Forensic Pipeline

**SRM + RGB + ELA in parallel branches with attention fusion**
- The "ideal" forensic detection architecture: frequency, spatial, and compression features processed in parallel branches with learned fusion
- Rejected because: builds on EMT-Net/ME-Net concepts above — same engineering/hardware barriers

### Large-Scale Dataset Replacement

**NIST MFC, DEFACTO, IMD2020**
- CASIA v2.0 is small (12,614 pairs) and has known quality issues
- Rejected because: (1) Some datasets require licensing (NIST), (2) dataset integration is a project in itself, (3) the assignment instructed "choose the dataset," and CASIA was chosen — changing the dataset changes the entire experimental comparison
- **Future direction:** Cross-dataset evaluation (train on CASIA, test on NIST) would be a strong post-assignment experiment

### Stronger Geometric Augmentation

**Elastic transforms, grid distortion, perspective transforms**
- Could improve generalization to varied forgery geometries
- Rejected for v9 because: risk of destroying forensic artifacts (compression boundaries, ELA patterns) that the model needs to learn. The augmentation could teach the model to ignore the very signals we want it to detect.
- May be revisited with careful testing in a controlled setting

---

## Cross-Dataset Evaluation (Future Research Priority)

**What:** Train on CASIA v2.0, evaluate zero-shot on:
- NIST MFC 2018/2019 (if accessible)
- Columbia Image Splicing Dataset
- Coverage (copy-move specific)

**Why:** Cross-dataset evaluation is the strongest test of genuine forgery detection capability. If the model's performance transfers, it has learned forensic features rather than dataset-specific shortcuts.

**Prerequisite:** Complete v9 with strong CASIA results and confirmed mask randomization test. Cross-dataset evaluation is meaningless if the model hasn't first demonstrated real learning on CASIA.

---

## Research Paper Connections

The following research directions connect Docs9 decisions to the broader forensic image analysis literature:

| Research Direction | Relevant Papers/Resources | Connection to v9 |
|---|---|---|
| Dual-task detection + localization | ManTra-Net, IF-OSN | v9 implements a simplified version of this paradigm |
| Compression-based features | Resource 07, Resource 08 | v9's ELA channel is a first step; full JPEG coefficient analysis is future work |
| Boundary-aware loss | Resource 03, Resource 16 | v9's edge loss draws from boundary detection literature |
| Noise residual analysis | SRM (Fridrich & Kodovsky), SPAM features | Deferred to v10 — most impactful for copy-move |
| Attention mechanisms | TransU2-Net, MVSS-Net | Deferred — requires transformer architecture |
| Multi-task learning | MTL survey literature | v9's dual-task head is a standard MTL approach |

---

## Implementation Roadmap (Post-Assignment)

If this work continues beyond the assignment submission:

**v10 (Next iteration — 2-3 weeks):**
1. SRM noise residual channels (highest priority deferred item)
2. Focal Loss or Tversky Loss (whichever addresses observed P/R imbalance)
3. Multi-scale inference TTA
4. Cross-dataset evaluation on Columbia Splicing Dataset

**v11 (Research extension — 1-2 months):**
1. EfficientNet encoder comparison
2. Attention-guided feature fusion (simplified MVSS-Net approach)
3. NIST MFC evaluation (if accessible)

**v12+ (Publication-grade — 3+ months):**
1. Full multi-branch architecture (RGB + SRM + ELA)
2. Transformer-based encoder comparison
3. Large-scale dataset training
4. Comprehensive cross-dataset benchmark

---

## Final Summary — Docs9 Decisions

### Key Improvements Approved for v9

| # | Improvement | Category | Expected Benefit |
|---|---|---|---|
| 1 | Learned classification head (dual-task) | Architecture | Replace heuristic detection; image AUC 0.90+ |
| 2 | ELA auxiliary channel (4-channel input) | Preprocessing | Compression artifact signal; copy-move F1 +0.05-0.10 |
| 3 | Auxiliary edge loss | Loss | Better boundary localization; Boundary F1 improvement |
| 4 | DataLoader optimization | Training | ~20% faster epochs (persistent_workers, prefetch) |
| 5 | DeepLabV3+ comparison | Architecture | Architecture-level evidence for decoder choice |
| 6 | Boundary F1 metric | Evaluation | Measures localization quality at boundaries |
| 7 | Precision-Recall curves | Evaluation | Full threshold-agnostic performance picture |
| 8 | Multi-seed validation (3 seeds) | Evaluation | Statistical confidence: mean ± std |
| 9 | Mask randomization test | Evaluation | Falsification of shortcut learning |
| 10 | pHash near-duplicate check | Data validation | Detect cross-split information leakage |
| 11 | Augmentation ablation | Experiment | Confirm augmentation contribution |
| 12 | Per-forgery-type loss tracking | Training | Diagnose copy-move vs splicing learning dynamics |
| 13 | Corrected dataset framing | Documentation | CASIA as "chosen baseline" — honest academic posture |
| 14 | Colab verification | Delivery | Mandatory pre-submission gate |

### Key Improvements Rejected and Why

| Improvement | Reason |
|---|---|
| Transformer encoder (SegFormer, TransU2-Net) | T4 memory insufficient; limited SMP support; engineering risk too high |
| Multi-branch forensic architecture (EMT-Net, ME-Net) | Custom implementation required; 2-4 weeks engineering; out of assignment scope |
| Full multi-branch pipeline (SRM + RGB + ELA) | Builds on rejected multi-branch approach |
| Large-scale dataset replacement | Changes experimental baseline; licensing issues; dataset integration scope |
| Stronger geometric augmentation | Risk of destroying forensic artifacts (compression boundaries, ELA patterns) |

### Key Improvements Deferred to v10+

| Improvement | Priority | Reason for Deferral |
|---|---|---|
| SRM noise residuals | HIGH | Most impactful for copy-move, but complex (30-channel preprocessing) |
| Focal / Tversky Loss | MEDIUM | pos_weight + Dice already adequate; revisit based on v9 P/R analysis |
| CbCr channels | LOW | Marginal gain over ELA alone |
| Multi-scale training | LOW | 40% time increase, limited expected benefit at 384×384 |
| Multi-scale inference (TTA) | MEDIUM | Post-hoc enhancement, not architectural |
| EfficientNet encoder | LOW | ResNet34 is reliable and well-supported |
| Cosine scheduler | LOW | ReduceLROnPlateau is working |
| Cross-dataset evaluation | HIGH | Requires completed v9 as baseline |

### Expected Impact on Model Performance

| Metric | Run01 (v8) | v9 Target | Confidence |
|---|---|---|---|
| Tampered-only Pixel F1 | 0.41 | 0.55–0.65 | Medium-High |
| Copy-move F1 | 0.31 | 0.38–0.45 | Medium |
| Splicing F1 | 0.59 | 0.65–0.75 | Medium-High |
| Image-level AUC (learned head) | 0.87 (heuristic) | 0.90+ | High |
| Boundary F1 | Not measured | New metric — baseline TBD | — |
| Robustness Δ (JPEG QF50) | −0.13 | < −0.06 | Medium |
| Optimal threshold | 0.1327 | 0.30–0.55 | Medium |

### Document Chain

Assignment.md → Docs7 → Run01 → Docs8 → Audit8 Pro → **Docs9** → Notebook v9

Docs9 closes the planning loop. The next action is implementation.
