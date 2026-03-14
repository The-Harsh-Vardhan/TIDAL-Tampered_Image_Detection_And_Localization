# Useful Engineering Ideas from vK.12.0 — Summary

| Field | Value |
|-------|-------|
| Source | vK.12.0 Image Detection and Localisation.ipynb |
| Architecture | Dual-head UNet + ResNet34 (PyTorch/SMP) — 151 cells |
| Outcome | FAILED — Tam-F1=0.1321, pixel-AUC=0.56, crashed at cell 77 |
| Extraction Date | 2026-03-15 |
| Purpose | Salvage reusable engineering patterns for vR.x and vR.P.x tracks |

---

## Why This Document Exists

vK.12.0 was the final iteration of the "Synthesis Era" (vK.11.x–12.0) that attempted to combine all prior learnings into a single dual-head architecture. It failed catastrophically — crashing with a KeyError and producing Tam-F1=0.1321 (worse than every prior run). However, the notebook contains 151 cells of engineering infrastructure built across 20+ experiment iterations. Many of these ideas — particularly in evaluation, visualization, and reproducibility — are framework-agnostic and directly applicable to the current ETASR ablation study (Track 1) and pretrained localization track (Track 2).

---

## Category Overview

| # | Category | Cell Range | Ideas | Adopt Now | Track 2 Only | Discard |
|---|----------|-----------|-------|-----------|-------------|---------|
| 1 | Dataset Handling | 20–38 | 5 | 2 | 0 | 3 |
| 2 | Preprocessing | 43–48 | 4 | 1 | 2 | 1 |
| 3 | Training Stability | 62–72 | 6 | 2 | 3 | 1 |
| 4 | Evaluation | 64–89 | 6 | 4 | 2 | 0 |
| 5 | Visualization | 12–108 | 6 | 4 | 2 | 0 |
| 6 | Debugging Tools | 58–112 | 4 | 2 | 1 | 1 |
| 7 | Reproducibility | 23–141 | 6 | 4 | 1 | 1 |
| 8 | Other | 60–120 | 5 | 2 | 1 | 2 |
| | **Total** | | **42** | **21** | **12** | **9** |

---

## Category 1: Dataset Handling (5 ideas)

| Idea | Cell | Description | Verdict |
|------|------|-------------|---------|
| Data leakage verification | 38 | Set intersection assertions on train/val/test paths | **ADOPT NOW** |
| Dataset summary table | 36 | Formatted class counts and ratios per split | **ADOPT NOW** |
| Multi-source dataset discovery | 20–21 | Kaggle → Drive → API download cascade | DISCARD (overengineered) |
| Metadata CSV caching | 32 | Cache dataset metadata with staleness detection | DISCARD (static dataset) |
| Cached stratified splitting | 34 | Cache split CSVs with row-count validation | DISCARD (sklearn handles this) |

---

## Category 2: Preprocessing (4 ideas)

| Idea | Cell | Description | Verdict |
|------|------|-------------|---------|
| ELA via cv2.imencode/imdecode | 43 | Alternative ELA using OpenCV JPEG codec | ABLATION CANDIDATE |
| Albumentations additional_targets | 44 | Synchronized multi-input augmentation | TRACK 2 ONLY |
| JPEG compression augmentation | 44 | Random QF 50–90 re-compression as augmentation | ABLATION CANDIDATE |
| Reproducible DataLoader (seed_worker) | 48 | PyTorch-specific reproducible data loading | TRACK 2 ONLY |

---

## Category 3: Training Stability (6 ideas)

| Idea | Cell | Description | Verdict |
|------|------|-------------|---------|
| Gradient clipping | 69 | clip_grad_norm_ (TF: clipnorm= in optimizer) | **ABLATION CANDIDATE** |
| ReduceLROnPlateau | 62 | factor=0.5, patience=3 | ALREADY PLANNED (vR.1.5) |
| Encoder freeze warmup | 72 | Freeze encoder for first N epochs | TRACK 2 (vR.P.1) |
| Differential learning rates | 62 | Encoder at lower LR than decoder | TRACK 2 ONLY |
| Three-file checkpoint strategy | 66 | Save last, best, and periodic checkpoints | **ADOPT NOW** |
| Gradient accumulation with AMP | 69 | PyTorch autocast + GradScaler | DISCARD (PyTorch-specific) |

---

## Category 4: Evaluation (6 ideas)

| Idea | Cell | Description | Verdict |
|------|------|-------------|---------|
| Tampered-only metric filtering | 64 | Report tampered-class metrics separately | **ADOPT NOW** |
| Threshold sweep optimization | 79 | Sweep 0.1–0.9 on val set, find optimal threshold | **ADOPT NOW** |
| Per-forgery-type evaluation | 87 | Parse Tp_D_ (splicing) vs Tp_S_ (copy-move) | **ADOPT NOW** |
| Worst-N failure case analysis | 110 | Show most confidently wrong predictions | **ADOPT NOW** |
| Pixel-level AUC-ROC | 83 | Flatten all pixel predictions for AUC | TRACK 2 ONLY |
| Mask-size stratified evaluation | 89 | Bucket by mask coverage, compute per-bucket metrics | TRACK 2 ONLY |

---

## Category 5: Visualization (6 ideas)

| Idea | Cell | Description | Verdict |
|------|------|-------------|---------|
| Training curves with best-epoch marker | 12, 77 | Vertical line + scatter at best epoch | **ADOPT NOW** |
| ELA heatmap with hot colormap | 103 | cv2.applyColorMap overlay on original | **ADOPT NOW** |
| TP/FP/FN color overlay | 14 | Green=TP, Red=FP, Blue=FN borders | **ADOPT NOW** |
| Mask coverage histogram + CDF | 51 | Distribution of tampered region sizes | **ADOPT NOW** |
| 6-panel enhanced visualization | 108 | Original, GT, pred, overlay, diff, contours | TRACK 2 ONLY |
| Contour overlays with OpenCV | 108 | cv2.findContours + drawContours on predictions | TRACK 2 ONLY |

---

## Category 6: Debugging Tools (4 ideas)

| Idea | Cell | Description | Verdict |
|------|------|-------------|---------|
| Shortcut learning detection | 91 | Mask randomization + boundary erosion tests | **ADOPT NOW** |
| FP/FN error analysis | 112 | Separate visualization of false positives and negatives | **ADOPT NOW** |
| Worst-N failure visualization | 110 | Grid of hardest failures with metadata | ALREADY COUNTED (Eval) |
| Model complexity + VRAM estimation | 58 | torchinfo summary + memory projection | DISCARD (architecture is fixed) |

---

## Category 7: Reproducibility (6 ideas)

| Idea | Cell | Description | Verdict |
|------|------|-------------|---------|
| Centralized CONFIG dict | 23 | All hyperparameters in one dict | ALREADY DONE |
| Seed verification cell | 133 | Assert seeds produce expected random values | **ADOPT NOW** |
| Split determinism verification | 135 | Hash split file paths, compare across runs | **ADOPT NOW** |
| Environment info logger | 141 | Print TF, GPU, NumPy, Python versions | **ADOPT NOW** |
| Full checkpoint + history save | 66 | Save history dict as JSON alongside weights | **ADOPT NOW** |
| VRAM-based batch auto-scaling | 26 | Auto-adjust batch size by GPU memory | DISCARD (ablation violation) |

---

## Category 8: Other (5 ideas)

| Idea | Cell | Description | Verdict |
|------|------|-------------|---------|
| Robustness testing suite | 116 | 8 degradation conditions (JPEG, noise, blur, resize) | **ADOPT NOW** |
| Inference speed benchmarking | 120 | Time predictions, report images/sec | **ADOPT NOW** |
| Grad-CAM explainability | 114 | Hook-based gradient visualization | FUTURE (needs tf-keras-vis) |
| Cross-version comparison table | 93 | Automated multi-version metrics table | DISCARD (manual table sufficient) |
| W&B experiment tracking | 60 | Weights & Biases with offline fallback | DISCARD (timeline risk) |

---

## Key Takeaway

The most valuable extraction from vK.12.0 is **NOT** its architecture or training code (which failed), but its **evaluation infrastructure**. The tampered-only metric filtering, threshold sweep, per-forgery-type evaluation, failure case analysis, and robustness testing suite represent evaluation maturity that the current vR.x track lacks. These can be integrated as non-variable additions to any notebook without violating the single-variable ablation rule.

---

## Cross-Reference

- Detailed adoption plan: `adoptable_improvements.md`
- Future ablation experiments: `ablation_candidates.md`
- What NOT to reuse: `discarded_elements.md`
- Integration into roadmap: `DocsR1/ablation_master_plan.md`, Section 8
