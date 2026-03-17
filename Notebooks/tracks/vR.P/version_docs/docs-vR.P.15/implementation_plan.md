# vR.P.15 — Implementation Plan

## Base Notebook
`vR.P.3 Image Detection and Localisation.ipynb` (from git HEAD)

## Cells Modified (10 of 28)

| Cell | Type | Change |
|------|------|--------|
| 0 | md | Title: "Multi-Quality ELA (Q=75, Q=85, Q=95)" |
| 1 | md | Changelog: add P.15 entry, diff table P.3 → P.15 |
| 2 | code | VERSION, CHANGE, ELA_QUALITIES=[75,85,95], remove ELA_QUALITY constant |
| 7 | md | Data prep: explain multi-quality ELA channel strategy |
| 8 | code | New `compute_multi_quality_ela()` function, updated Dataset class, updated `compute_ela_statistics()` |
| 9 | code | Call updated stats function with 3 qualities |
| 11 | md | Architecture: input is "Multi-Q ELA (75/85/95)" |
| 25 | code | Results table: "MQ-ELA 384²" input column |
| 26 | md | Discussion: multi-quality hypothesis, channel interpretation |
| 27 | code | Model save: config includes ela_qualities list |

## Key New Code: Multi-Quality ELA (Cell 8)

```python
ELA_QUALITIES = [75, 85, 95]

def compute_ela_grayscale(image_path, quality):
    """Compute single-quality ELA as grayscale numpy array [0,255]."""
    original = Image.open(image_path).convert('RGB')
    buffer = BytesIO()
    original.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    ela = ImageChops.difference(original, resaved)
    extrema = ela.getextrema()
    max_diff = max(val[1] for val in extrema)
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela = ImageEnhance.Brightness(ela).enhance(scale)
    return np.array(ela.convert('L'))  # grayscale

def compute_multi_quality_ela(image_path, qualities=[75, 85, 95], size=384):
    """Stack ELA maps at multiple quality levels as 3-channel image."""
    channels = []
    for q in qualities:
        gray = compute_ela_grayscale(image_path, q)
        gray = np.array(Image.fromarray(gray).resize((size, size), Image.BILINEAR))
        channels.append(gray)
    return np.stack(channels, axis=-1)  # (H, W, 3) — one channel per quality
```

## Dataset Class Changes
- `__getitem__` calls `compute_multi_quality_ela()` instead of `compute_ela_image()`
- Returns numpy array (H,W,3) → to_tensor → normalize with multi-quality stats
- `compute_ela_statistics()` updated to compute stats across multi-quality channels

## Unchanged Cells
3–6, 10, 12–16, 17–24 — dataset discovery, model build, loss, training, evaluation, visualization

## Verification Checklist
1. Cell count == 28
2. VERSION == 'vR.P.15'
3. ELA_QUALITIES = [75, 85, 95] in cell 2
4. No ELA_QUALITY single constant
5. compute_multi_quality_ela function in cell 8
6. compute_ela_grayscale function in cell 8
7. Dataset calls compute_multi_quality_ela (not compute_ela_image)
8. '{VERSION}' in cell 27 save
