## Plan: Pre-Implementation Documentation for Tampered Image Detection & Localization

**TL;DR**: Before writing any code, produce 7 structured documentation files that demonstrate deep understanding of the problem domain, survey the solution landscape, justify every design decision, and align the project with industry-grade forensic practices. This documentation becomes the foundation of your Colab notebook's narrative and shows evaluators your engineering maturity.

---

### Steps

#### Phase A: Problem & Research Foundation (Day 1)

**1. Problem Statement Explanation** — [Docs/Assignment.md](Docs/Assignment.md)
   - Define image tampering and its 3 core manipulation types: **splicing** (foreign region pasted in), **copy-move** (region duplicated within same image), **removal/inpainting** (object erased and filled)
   - Explain the dual-task nature: **image-level classification** (authentic vs. tampered) + **pixel-level localization** (binary segmentation mask)
   - Describe the forensic assumption: every stage of the image acquisition pipeline (sensor → ISP → demosaicing → compression) leaves a digital fingerprint; manipulations disrupt these fingerprints creating detectable statistical inconsistencies in noise patterns, CFA correlations, and compression artifacts
   - Explain **passive vs. active** forensics — this project is passive (no watermark/signature needed)
   - State **why it matters**: combating misinformation, digital evidence integrity in legal proceedings, journalism verification, social media platform trust
   - Decompose the assignment's 4 sections into concrete engineering tasks

**2. Possible Solutions Survey** 
   - **Traditional CV approaches**: Error Level Analysis (ELA), noise variance analysis, CFA pattern detection — fast but fragile against sophisticated edits
   - **Single-stream CNNs**: Standard U-Net/DeepLabV3+ on RGB — captures semantic inconsistencies but misses low-level artifact signals
   - **Dual-branch/Multi-view networks**: MVSS-Net++ (Edge-Supervised Branch + Noise-Sensitive Branch), BusterNet — fuses semantic + forensic features; strong localization
   - **Transformer-based**: SegFormer (hierarchical multi-scale + lightweight MLP decoder), ObjectFormer — captures long-range dependencies critical for copy-move detection
   - **Hybrid SOTA frameworks**: TruFor (SegFormer + Noiseprint++ camera fingerprint + reliability map), CAT-Net (RGB + DCT stream), REFORGE (segmentation + RL refinement)
   - **Explainable multimodal**: FakeShield (vision-language forensic reasoning)
   - Include a **comparison table**: Architecture | Forgery Types Targeted | Computational Cost | Colab T4 Feasibility | Localization Quality | Key Limitation

**3. Dataset Exploration** 
   - Survey all relevant datasets from research docs:
     - **CASIA v2.0**: ~4,975 images (1,701 Au + 3,274 Tp), splicing + copy-move, widely benchmarked, Kaggle-accessible. Known issue: 17 images with resolution misalignment
     - **CASIA v1.0**: Older, lower quality, fewer images
     - **COVERAGE**: 200 images, copy-move only, specifically designed for "similar genuine objects" ambiguity — very challenging
     - **CoMoFoD**: 260 sets, copy-move with post-processing variants (JPEG, blur, noise) — excellent for robustness testing
     - **IEEE Forensics**: ~1,451 images, splicing, high-quality masks
     - **NIST16**: Large-scale benchmark, multiple forgery types
     - **IMD2020**: In-the-wild dataset, real-world manipulations
   - Create a **comparison matrix**: Dataset | Total Images | Forgery Types | Mask Quality | Resolution | Known Issues | Accessibility | Benchmark SOTA F1

#### Phase B: Decision & Justification (Day 2)

**4. Best Dataset Selection** 
   - **Primary**: CASIA v2.0 — largest publicly available with both splicing + copy-move, Kaggle API accessible (slug: `divg07/casia-20-image-tampering-detection-dataset`), most widely benchmarked (SOTA Pixel-F1: 94.1% VAAS, IoU: 85.1% VASLNet)
   - **Supplementary** (if time permits): COVERAGE for copy-move bonus points
   - **Justification criteria**: size, diversity of manipulation types, mask quality, benchmarkability, accessibility in Colab, alignment with assignment requirements
   - **Data cleaning requirements**: Resolution misalignment fix (17 images), mask binarization (threshold > 128), naming convention mapping (Tp image → `_gt` mask)
   - **Split strategy**: 85% train / 7.5% val / 7.5% test, stratified by authentic/tampered ratio
   - **Class balancing**: Balanced subset (~1000 Au + 2000 Tp) to prevent bias

**5. Best Solution Architecture** 
   - **Recommended approach**: Dual-stream encoder with **segmentation_models_pytorch (SMP)** library
     - **Option A (Practical)**: U-Net or DeepLabV3+ with `efficientnet-b1` encoder, SRM noise residuals concatenated as additional input channels — via SMP for clean, tested implementation
     - **Option B (Advanced)**: SegFormer-B1 encoder with dual RGB + noise stream and spatial attention fusion — closer to SOTA but more complex
   - **Forensic preprocessing**: SRM filter layer (30 fixed high-pass kernels) to extract noise residuals, concatenated with RGB input → 6-channel input
   - **Loss function**: $L_{total} = \alpha \cdot L_{BCE} + \beta \cdot L_{Dice} + \gamma \cdot L_{Edge}$ — handles extreme class imbalance (tampered < 5% of pixels)
   - **Training**: AdamW optimizer, cosine annealing LR scheduler, AMP mixed precision, gradient accumulation (effective batch 16 via 4 micro-batches of 4)
   - **Decision rationale table**: Why this over alternatives (complexity, T4 feasibility, expected performance, implementation time within 1 week)

#### Phase C: Engineering Standards (Day 2-3)

**6. Best Practices** 
   - **Reproducibility**: Fixed random seeds (PyTorch, NumPy, Python), pinned library versions in notebook
   - **Data pipeline**: Use `albumentations` for synchronized image/mask augmentation (HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, RandomJPEGCompression)
   - **Training discipline**: AMP with `torch.cuda.amp.autocast()` + `GradScaler()`, `pin_memory=True` in DataLoader, `num_workers=2` for Colab
   - **Evaluation rigor**: Report Pixel-F1, Pixel-IoU, Image-Accuracy, AUC-ROC; use threshold sweep for Oracle-F1; separate robustness evaluation table
   - **Visualization standard**: 4-column grid (Original | GT Mask | Prediction Heatmap | Overlay with alpha=0.5)
   - **Notebook structure**: Clear markdown sections mirroring the 4 assignment sections, inline explanations of every design choice
   - **Checkpointing**: Save best model by validation F1, not loss

**7. Industry Relevance & Latest Technologies** 
   - **2024-2025 SOTA landscape**: TruFor, MVSS-Net++, DAE-Net, VAAS, FakeShield
   - **Key technologies to reference/use**:
     - `segmentation_models_pytorch` — industry-standard segmentation library
     - `albumentations` — fastest augmentation pipeline
     - PyTorch AMP — production training standard
     - SegFormer architecture — state-of-the-art efficiency/performance tradeoff
     - Noiseprint++ concept — camera-fingerprint-based detection
   - **Emerging frontiers**: Zero-shot detection of AI-generated forgeries (diffusion model artifacts), explainable forensics with VLMs
   - **Production considerations**: Model quantization for deployment, ONNX export, confidence/reliability maps for human-in-the-loop workflows
   - **What makes this "industry-grade"**: Handling class imbalance properly, robustness testing, clear evaluation methodology, reproducibility, visual evidence documentation

---

### Relevant Files
- [Docs/Assignment.md](Docs/Assignment.md) — original assignment requirements (reference for all docs)
- [Docs/Deep Research.md](Docs/Deep%20Research.md) — exhaustive technical analysis of SOTA architectures, loss functions, optimization, and evaluation (primary reference for Docs 2, 5, 6, 7)
- [Docs/Dataset Selection.md](Docs/Dataset%20Selection.md) — dataset comparison and recommendation (reference for Docs 3, 4)
- [Docs/Dataset.md](Docs/Dataset.md) — Kaggle vs HuggingFace decision and pipeline strategy (reference for Doc 4)
- [Docs/Overall Flow.md](Docs/Overall%20Flow.md) — step-by-step implementation workflow (reference for Docs 5, 6)

### Verification
1. Each document should be self-contained and reference specific papers/architectures from the research docs
2. Cross-check all dataset statistics against the research docs (e.g., CASIA v2.0 has ~4,975 images, 17 misaligned)
3. Verify T4 GPU feasibility claims — SegFormer-B1 with 512x512 input at batch 4 fits in 16GB VRAM
4. Ensure the recommended solution directly maps to every assignment requirement (detection + localization + metrics + visualization + bonus)
5. Review each doc through the lens: "Would a Principal Engineer at BigVision approve this reasoning?"

### Decisions
- **Documentation-first approach**: All 7 docs before any code — this is deliberate engineering discipline, not overhead
- **Scope**: These docs cover the _what_ and _why_; implementation details (exact code patterns) are deferred to the coding phase
- **Format**: Markdown files in the `Docs/` folder, structured for later integration into the Colab notebook narrative
- **Output**: 7 new `.md` files in `Docs/` folder

### Further Considerations
1. **Depth vs. Breadth in Docs**: Should each document be 2-3 pages with diagrams/tables, or shorter 1-page summaries? **Recommendation**: 2-3 pages each with tables/comparisons — this is your chance to demonstrate depth before the code does.
2. **Do you want architecture diagrams?** I can describe diagram structures in the docs that you can create using draw.io or similar. **Recommendation**: Yes, at least for the solution architecture in Doc 5.
3. **Should we include a timeline/Gantt chart** for the remaining ~5 days of implementation after docs? **Recommendation**: Yes, include it in a separate planning doc to keep you on track.
