# 11 — Limitations and Future Work

## Purpose

Document known limitations of the current approach and potential improvements that are explicitly out of scope for the assignment.

## Known Limitations

### Data Split Integrity

CASIA v2.0 does not publish source-image groupings. Some tampered images may share the same source original, and the current stratified split does not control for this. This means related images could appear in both training and test sets, potentially inflating reported metrics.

**Mitigation:** Stratify by forgery type and randomize. Acknowledge the limitation. True group-aware splitting would require manual source annotation.

### Dataset Size

CASIA v2.0 contains approximately 5,000 usable images. This is small by modern deep learning standards. The model may overfit or fail to generalize to substantially different tampering styles.

### Image-Level Detection Fragility

Image-level tamper detection is derived from `max(probability_map)`. A single high-confidence false positive pixel is enough to misclassify an authentic image. More stable alternatives (top-k mean, mask area fraction) are worth exploring.

### Metric Inflation from Authentic Images

When averaging pixel metrics across all test images, authentic images with correct all-zero predictions contribute F1=1.0, inflating the mixed-set average. The tampered-only view gives the more honest localization assessment.

### Colab Session Limits

The free Colab tier has session time limits and may disconnect. Checkpointing to Google Drive mitigates this, but training may need manual resumption.

### SRM Placeholder Quality

The SRM filter implementation discussed in earlier documentation uses 3 base kernels repeated to simulate 30 channels. This is a placeholder, not a serious forensic SRM implementation. If SRM is attempted as a bonus, the filter bank quality should be acknowledged.

## Future Work (Out of Scope)

The following are potentially valuable but are not part of the assignment:

| Improvement | Description |
|---|---|
| Group-aware splitting | Annotate source originals and enforce no-leak splits |
| Cross-dataset evaluation | Test on COVERAGE, CoMoFoD, or Columbia datasets |
| Advanced architectures | SegFormer, HRNet, dual-stream forensic networks |
| Proper SRM implementation | Full 30-kernel SRM bank from published forensic literature |
| Adversarial robustness | Evaluate against anti-forensic attacks |
| W&B hyperparameter sweeps | Systematic search over LR, loss weights, encoder choice |
| HuggingFace deployment | Gradio/Spaces demo for interactive inference |
| Larger datasets | Train on merged multi-source data for better generalization |

These are documented for context. None should be attempted unless all core and bonus work is complete.

## Related Documents

- [01_Assignment_Overview.md](01_Assignment_Overview.md) — Scope boundaries
- [08_Robustness_Testing.md](08_Robustness_Testing.md) — Bonus evaluation
