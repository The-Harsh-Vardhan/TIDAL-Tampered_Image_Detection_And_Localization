# Best Notebooks — Curated Top 5 Experiments

These are the best-performing notebooks from across the entire project, selected based on **dataset results** and **assignment alignment** (image forgery detection AND localization on CASIA v2.0).

---

## Selection Criteria

- **Assignment requires both classification AND pixel-level localization** — notebooks that only classify (no segmentation masks) are less aligned
- Ranked by Pixel F1, Pixel IoU, Image Accuracy, and Image ROC-AUC
- The **vR.P.x (Pretrained) track** dominates all localization metrics; older vK.x notebooks (Notebooks/Runs/) scored significantly lower (best vK Tampered F1 = 0.41 vs vR.P.10's 0.73)

---

## Leaderboard

| # | Notebook | Pixel F1 | Pixel IoU | Pixel AUC | Img Acc | Img ROC-AUC | Why Selected |
|---|----------|----------|-----------|-----------|---------|-------------|--------------|
| 1 | **vR.P.10** — CBAM Attention | **0.7277** | **0.5719** | **0.9573** | 87.32% | **0.9633** | Best localization overall (5 of 11 metric crowns) |
| 2 | **vR.P.7** — Extended Training | 0.7154 | 0.5569 | 0.9504 | 87.37% | 0.9433 | 2nd best Pixel F1; best FN rate (25.9%) |
| 3 | **vR.P.8** — Progressive Unfreeze | 0.6985 | 0.5367 | 0.9541 | **87.59%** | 0.9578 | Best image accuracy among localization models |
| 4 | **vR.P.3** — ELA Breakthrough | 0.6920 | 0.5291 | 0.9528 | 86.79% | 0.9502 | Most impactful single innovation (+23.74pp Pixel F1 over RGB baseline) |
| 5 | **vR.1.6** — Deeper CNN | N/A | N/A | N/A | **90.23%** | **0.9657** | Best classification accuracy (ETASR track champion; no localization) |

---

## Folder Structure

Each subfolder contains:
- **Source notebook** (.ipynb) — the experiment code
- **Run output** (.ipynb) — Kaggle execution with all outputs/metrics
- **docs/** — experiment description, implementation plan, expected outcomes
- **Audit report** (.md) — detailed post-run analysis

---

## Key Research Findings

1. **ELA input is the #1 breakthrough** — vR.P.3 jumped +23.74pp Pixel F1 over the RGB baseline by feeding ELA images instead of RGB
2. **CBAM attention is #2** — vR.P.10 added +3.57pp Pixel F1 with only 11,402 extra parameters (0.36%)
3. **Extended training is #3** — vR.P.7 gained +2.34pp by training 50 epochs instead of 25
4. **Architecture changes > training tricks** — structural improvements (deeper CNN, attention modules) consistently outperformed hyperparameter tuning (loss functions, schedulers, class weights)
5. **Older vK.x track notebooks** (v6.5, vk-7-1, vK.10-12.x) all underperformed the vR.P.x series; vK.11.x and vK.12.x suffered model collapse (Tampered Dice ~0.13)

---

## Dataset

All experiments use **CASIA v2.0** (12,614 images: 7,491 Authentic + 5,123 Tampered) with 70/15/15 train/val/test split.
