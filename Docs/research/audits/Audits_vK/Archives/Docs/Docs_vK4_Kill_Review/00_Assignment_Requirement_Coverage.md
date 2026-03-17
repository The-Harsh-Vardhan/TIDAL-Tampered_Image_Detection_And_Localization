# 00 — Assignment Requirement Coverage

**Reviewer:** Arshad Siddiqui, Principal AI Engineer, BigVision LLC  
**Notebook:** vK.4 Image Detection and Localisation.ipynb

---

| # | Requirement | Status | Explanation |
|---|---|---|---|
| 1 | Dataset explanation | **Partial** | Section 3 auto-discovers the dataset and prints counts per class, but there is **no EDA** — no sample images shown with masks, no distribution of image sizes, no mask area histogram, no class imbalance discussion. The markdown cell (Cell 8) is three lines long. A senior reviewer expects analysis, not just autodiscovery. |
| 2 | Model architecture description | **✓** | Cell 17 (markdown) + Cell 18 (code) clearly define `UNetWithClassifier` with DoubleConv/Down/Up blocks and a classification head. Parameter count and shape check are printed. Adequate for an assignment. |
| 3 | Training strategy explanation | **✓** | Cell 23 (markdown) lists AMP, gradient accumulation, gradient clipping, early stopping, and LR scheduling. The training loop code (Cells 24-25) implements all of these. |
| 4 | Hyperparameter documentation | **✓** | The centralized `CONFIG` dict (Cell 4) documents every hyperparameter with inline comments. This is one of the strongest parts of the notebook. |
| 5 | Evaluation results | **Partial** | Cell 31 reports pixel F1, IoU, image accuracy, tampered-only F1, and mask-size stratified F1. However: **no confusion matrix**, **no AUC-ROC** (imported but never used), **no per-class precision/recall**, **no per-forgery-type breakdown** (forgery type is detected but never used in evaluation). The evaluation is narrower than it appears. |
| 6 | Visualization of predictions | **✓** | Cell 34 shows a prediction grid with Best / Median / Worst / Authentic samples. Cell 37 shows Grad-CAM. Cell 35 shows F1 vs threshold. Adequate range of visualizations. |
| 7 | Tamper detection | **✓** | The classifier head outputs 2-class logits (authentic/tampered). Image-level accuracy is reported. |
| 8 | Tamper localization | **✓** | The segmentation head outputs pixel-level masks. Threshold sweep and mask-size stratification are present. |
| 9 | Runnable Colab pipeline | **Partial** | The notebook is explicitly **Kaggle-only**. The requirement says "Runnable Colab pipeline" — this notebook will **not** run on Colab without modification: no Drive mount, no Colab dependency installation, hardcoded `/kaggle/` paths. Requirement says Colab, notebook targets Kaggle. |
| 10 | Architecture reasoning | **Partial** | Cell 17 states the architecture is "preserved from vK.2" but provides **no justification** for why this custom U-Net is appropriate for tamper detection. No comparison with alternatives, no ablation, no discussion of limitations vs. pretrained encoders. |

---

## Score: **6.5 / 10** (4 ✓, 4 Partial, 0 ✗ — but "Partial" on critical requirements)

> [!WARNING]
> The Colab requirement is explicitly violated. The notebook targets Kaggle only. This alone could be disqualifying depending on the evaluator's strictness.
