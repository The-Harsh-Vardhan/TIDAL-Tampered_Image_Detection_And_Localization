# Rigorous Silent-Failure Audit: v6.5 Segmentation Pipeline

Scope audited (static code/doc inspection only):
- `notebooks/tamper_detection_v6.5_kaggle.ipynb` (primary executable reference)
- `notebooks/tamper_detection_v6.5_colab.ipynb` (runtime variant delta check)
- `Docs6/` (documentation drift and consistency risk check)

No training rerun was performed. Any runtime-performance claim is marked as unverified unless directly inferable from code behavior.

## 1. Overall Pipeline Reliability Score (1–10)

**Score: 4.1 / 10**

Weighted rubric used:

| Dimension | Weight | Score (0-10) | Weighted Contribution |
|---|---:|---:|---:|
| Evaluation integrity (metrics, thresholding, aggregation, empty-mask handling) | 35% | 3.8 | 1.33 |
| Leakage/shortcut risk (split strategy, dedup/grouping, artifact learning risk) | 25% | 4.2 | 1.05 |
| Robustness realism (attack coverage + protocol realism) | 20% | 3.0 | 0.60 |
| Training stability/engineering reliability | 20% | 5.6 | 1.12 |
| **Total** | **100%** |  | **4.10** |

Interpretation: the pipeline is operationally coherent, but current evaluation design can overstate localization quality and under-detect shortcut dependence.

## 2. Potential Metric Inflation Risks

High-risk inflation mechanisms identified from implementation:

1. **Mixed-set pixel metrics are inflated by authentic images**: empty GT + empty prediction is scored as perfect (`F1=1`, `IoU=1`), then averaged per image.
2. **Recall is artificially protected on authentic false positives**: for `gt.sum()==0 and pred.sum()>0`, precision is `0.0` but recall is hard-set to `1.0`.
3. **Per-image macro averaging hides pixel-level failure concentration**: no global-pixel (micro) precision/recall/IoU is reported.
4. **Threshold selected on validation mixed-set F1** can bias toward background-heavy operating points.
5. **Image-level detection heuristic uses `max(prob_map)` and same segmentation threshold**, making decisions sensitive to isolated hot pixels.
6. **Documentation drift can cause misinterpretation of reported metrics** (docs claim top-k image score and other behaviors not in v6.5 code).

### 13-Check Audit Matrix (Code-First)

| Check | Observed Implementation Behavior (Evidence) | Silent Failure Mechanism | Severity | Confidence | Concrete Mitigation |
|---|---|---|---|---|---|
| 1. Dataset shortcut learning | CASIA used directly; no explicit anti-artifact controls beyond flips/rotations (`kaggle c16`, `c17`) | Model can learn compression seams/color discontinuities instead of semantic tamper logic | High | High | Add artifact-counterfactual tests: recompress both classes, blur boundaries, seam-randomization ablations |
| 2. Dataset bias | Dataset limited to classical splicing/copy-move; no distribution balancing by edit realism/scene families | Inflated in-domain metrics with poor real-world transfer | High | Medium | Add external forensic test set + bias-stratified reporting (scene type, edit size, compression level) |
| 3. Background dominance | `compute_pixel_f1` and `compute_iou` return 1.0 on true negatives (`kaggle c24`), mixed-set mean used in `evaluate` (`c31`) | Large authentic/background share can inflate mean localization metrics | Critical | High | Promote tampered-only and micro-pixel metrics to primary KPI; report authentic false-positive burden separately |
| 4. Empty mask problem | `compute_precision_recall`: empty GT + positive prediction returns `(0.0, 1.0)` (`kaggle c24`) | Recall becomes non-penalized for false positives on authentic images | Critical | High | Redefine recall as undefined/NA for empty GT or exclude empty-GT samples from recall mean |
| 5. Data leakage | Split uses stratified `train_test_split`; leakage check is path-overlap assertions only (`kaggle c13`) | Near-duplicate/source-scene leakage remains possible | High | High | Group split by perceptual hash / source-cluster ID; enforce cluster-level disjoint splits |
| 6. Mask alignment | Dimension equality checked pre-split (`c9`), masks binarized `>0` (`c17`), resize via Albumentations (`c16`) | Implicit interpolation defaults and coarse masks can silently distort boundaries | Medium | Medium | Explicitly set mask interpolation policy; add boundary IoU and pre/post resize mask consistency tests |
| 7. Augmentation leakage | Augmentation only in train transform; val/test use deterministic transform (`c16`, `c18`) | Leakage risk low; main residual risk is split-level content similarity, not transform leakage | Low | High | Keep current separation; add split-level dedup controls |
| 8. Metric implementation | Metrics computed per-image then averaged (`c25`, `c31`); no global-pixel confusion totals | Macro averaging can hide failures concentrated in small/rare tampered regions | High | High | Report both macro and micro metrics, plus size-stratified metrics by tamper area bins |
| 9. Threshold bias | 50-threshold sweep on validation to maximize mean F1 (`c30`) | Validation overfitting to chosen threshold, especially with mixed-set objective | High | High | Nested validation or fixed policy threshold; optimize threshold on tampered-only F1 + FP constraint |
| 10. Visualization bias | Grid picks best/median/worst tampered + first 2 authentic (`c38`) | Curated samples can underrepresent systematic authentic false positives | Medium | High | Add random sample grid and worst-authentic-FP panel with fixed seed |
| 11. Explainability reliability | Grad-CAM target is `output.mean()` over segmentation logits (`c41`); shown on top-F1 tampered samples (`c43`) | Heatmap can reflect generic activation intensity, not true tamper evidence | High | High | Use region-targeted objectives, include hard failures/authentic FPs, and add perturbation sanity checks |
| 12. Robustness testing | Only JPEG/noise/blur/resize degradations (`c46-c48`) | Misses modern edits (GAN, diffusion, semantic inpainting), overstating robustness | High | High | Add synthetic modern manipulations + cross-dataset robustness benchmark |
| 13. Training stability | BCE+Dice, grad clipping, early stopping present (`c22`, `c25`, `c27`); no class-weighting, no multi-seed variance | Stable training may still converge to shortcut features under imbalance | Medium | Medium | Add multi-seed runs, class-aware sampling/weights, calibration and drift diagnostics |

## 3. Dataset Shortcut Learning Risks

Primary shortcut channels likely present:

1. **Compression and resampling traces**: CASIA-style manipulations often introduce codec inconsistencies; the model can exploit these non-semantic cues.
2. **Boundary artifact dependence**: segmentation heads can overfit to hard splice seams rather than manipulated content semantics.
3. **Color/statistics mismatch shortcuts**: pasted regions can differ in illumination/noise statistics and become trivial cues.
4. **Scene repetition risk**: without group-aware splitting, model may memorize source-scene priors.

Why current pipeline is vulnerable:
- No counterfactual controls to neutralize seam/compression cues.
- No source-group dedup split.
- Primary optimization/evaluation still rewards background-safe predictions.

## 4. Evaluation Weaknesses

Core weaknesses:

1. **Metric semantics can mislead**:
   - true negatives are treated as perfect localization across multiple metrics;
   - recall handling for empty GT can be inflated by definition.
2. **Aggregation mismatch**:
   - macro per-image averaging only;
   - no micro/global pixel confusion reporting.
3. **Threshold protocol fragility**:
   - threshold tuned on mixed-set validation F1;
   - same threshold reused for both localization and image-level detection.
4. **Image-level detection heuristic**:
   - `tamper_score = max(prob_map)` in v6.5 (`kaggle c31`), sensitive to isolated spikes.

Notebook-specific deltas:

- **Kaggle notebook** is internally consistent and executable in structure.
- **Colab notebook has a reproducibility blocker**: malformed dataset-loading code in cell 8 (fused/garbled statements around dataset path printing), making execution unreliable without manual repair.

Docs-vs-code drift risks (informational, but high impact for reporting integrity):

1. `Docs6/05_Evaluation_Methodology.md` documents image-level top-k mean scoring; v6.5 code uses `max(prob_map)`.
2. `Docs6/12_Complete_Notebook_Structure.md` describes behaviors (e.g., threshold-aware training, global metric wording) not aligned with current v6.5 implementation.
3. `Docs6/00_Master_Report.md` is anchored to v5.1 context, not v6.5 notebooks.

## 5. Robustness Gaps

Current robustness suite is narrow and mostly post-processing based:
- Covered: JPEG compression, Gaussian noise, Gaussian blur, resize degradation.
- Missing high-relevance modern manipulations:
  - diffusion-based local edits
  - GAN-generated composites
  - semantic inpainting/object replacement
  - prompt-based generative edits
  - copy-paste with harmonization

Protocol gaps:
- No adversarially designed counterforensic edits.
- No threshold recalibration stress under shift (only fixed threshold reuse).
- No robustness reporting by tamper size/type under degradation.

## 6. Suggested Experiments to Validate the Model

Experiments specifically designed to **falsify shortcut learning**:

1. **Group-aware split audit**: cluster images by perceptual hash / source similarity and re-split by cluster. Compare drop vs path-level split.
2. **Compression equalization test**: force identical recompression pipeline on authentic and tampered images before inference; measure performance collapse.
3. **Boundary destruction ablation**: blur/feather GT seam neighborhoods only at evaluation time. If performance drops sharply, model is seam-driven.
4. **Patch-context swap test**: keep manipulated content but harmonize low-level statistics; check whether detections disappear.
5. **Tamper-area stratified evaluation**: report metrics across area bins (`<1%`, `1-5%`, `5-15%`, `>15%`) to expose small-region failure.
6. **Authentic hard-negative suite**: curated authentic images with strong texture/compression artifacts to quantify false alarms.
7. **Modern edit stress test**: evaluate on synthetic diffusion/GAN/inpainting edits with human-verified masks.
8. **Metric robustness re-computation**: compare macro vs micro metrics and NA-aware recall handling for empty GT.
9. **Threshold stability analysis**: bootstrap validation folds, estimate threshold variance and confidence interval.

## 7. Recommended Improvements

Priority order:

1. **Fix metric definitions and reporting first**:
   - replace empty-GT recall handling;
   - publish macro + micro metrics;
   - make tampered-only localization the primary headline.
2. **Harden split protocol**:
   - add perceptual-hash/source-cluster disjoint splitting;
   - report leakage stress-test results explicitly.
3. **Decouple image-level detection from segmentation threshold**:
   - use a dedicated image-level head or calibrated aggregator;
   - avoid `max(prob_map)` as sole decision statistic.
4. **Upgrade robustness evaluation**:
   - include modern generative-edit scenarios;
   - add robustness-by-forgery-type and tamper-area breakdown.
5. **Strengthen explainability validation**:
   - use failure-focused and authentic-FP-focused XAI sampling;
   - add deletion/insertion sanity checks for attribution validity.
6. **Repair Colab notebook reproducibility**:
   - correct malformed dataset-loading cell;
   - enforce a syntax validation pass for generated notebooks.
7. **Resolve docs drift**:
   - explicitly align `Docs6` with v6.5 behavior or freeze docs to v5.1 and state mismatch.

