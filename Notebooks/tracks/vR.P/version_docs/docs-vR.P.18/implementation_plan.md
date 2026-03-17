# vR.P.18 — Implementation Plan

## Core Implementation: Robustness Evaluation

### `recompress_image(image_path, quality_factor)`

1. Open image with PIL
2. Save to BytesIO buffer as JPEG at `quality_factor`
3. Re-open from buffer
4. Return recompressed PIL image

### `RobustnessTestDataset`

Custom dataset that optionally recompresses images before computing ELA:
- If `recompress_quality` is None: standard ELA pipeline (original image)
- If set: recompress first, then compute ELA on the recompressed image

### Evaluation Loop

```python
for qf in [None, 95, 90, 80, 70]:
    dataset = RobustnessTestDataset(..., recompress_quality=qf)
    loader = DataLoader(dataset, ...)
    probs, masks, labels = evaluate_test(model, loader, DEVICE)
    # compute all metrics → store in results dict
```

### Visualization

- Degradation curves: Pixel F1/IoU/AUC vs compression quality (line plot)
- Side-by-side predictions: same image under all 5 conditions
- Confusion matrices: one per condition (5 panels or 1x5 strip)
- ELA comparison: show how ELA changes across compression levels

### Cell Modification Map

| Cell | Action |
|------|--------|
| 0 | Title: "vR.P.18 — Compression Robustness Testing" |
| 1 | Changelog |
| 2 | VERSION='vR.P.18', add QUALITY_FACTORS=[70,80,90,95] |
| 7 | Explain robustness methodology |
| 8 | Add recompress function + RobustnessTestDataset |
| 9 | Create 5 test datasets (not 5 train/val/test splits — test only) |
| 11 | "No training — evaluating P.3 model" |
| 12 | Load P.3 checkpoint instead of building new model |
| 13-14 | Training section: SKIPPED |
| 15 | Replace training loop with robustness evaluation loop |
| 17-20 | Multi-condition metrics, confusion matrices, degradation curves |
| 22-23 | Side-by-side condition comparison |
| 25 | 5-row results table |
| 26 | Robustness analysis discussion |
| 27 | Save results dict (no model) |
