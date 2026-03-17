# vR.P.14 — Implementation Plan

## Base Notebook
`vR.P.3 Image Detection and Localisation.ipynb` (from git HEAD)

## Cells Modified (8 of 28)

| Cell | Type | Change |
|------|------|--------|
| 0 | md | Title: "Test-Time Augmentation (TTA)" |
| 1 | md | Changelog: add P.14 entry, describe TTA |
| 2 | code | VERSION, CHANGE, add TTA_VIEWS constant |
| 13 | md | Training config: mention TTA in evaluation strategy |
| 17 | code | Replace evaluate_test() with TTA-enabled version |
| 25 | code | Results table: add "TTA" column, show with/without TTA comparison |
| 26 | md | Discussion: TTA hypothesis, inference cost, generalizability |
| 27 | code | Model save: config includes tta_views |

## Key New Code: TTA Prediction Function (Cell 17)

```python
TTA_TRANSFORMS = [
    lambda x: x,                              # original
    lambda x: torch.flip(x, dims=[-1]),       # horizontal flip
    lambda x: torch.flip(x, dims=[-2]),       # vertical flip
    lambda x: torch.flip(x, dims=[-2, -1]),   # both flips
]

TTA_INVERSE = [
    lambda x: x,
    lambda x: torch.flip(x, dims=[-1]),
    lambda x: torch.flip(x, dims=[-2]),
    lambda x: torch.flip(x, dims=[-2, -1]),
]

def predict_with_tta(model, images, device):
    preds = []
    for fwd, inv in zip(TTA_TRANSFORMS, TTA_INVERSE):
        augmented = fwd(images)
        with autocast('cuda'):
            out = model(augmented)
        probs = torch.sigmoid(out.float())
        preds.append(inv(probs))
    return torch.stack(preds).mean(dim=0)
```

## Unchanged Cells
3–12, 14–16, 18–24 — dataset, model build, loss, training loop, image-level eval, visualization

## Verification Checklist
1. Cell count == 28
2. VERSION == 'vR.P.14'
3. TTA_VIEWS = 4 in cell 2
4. TTA_TRANSFORMS list in cell 17
5. predict_with_tta function in cell 17
6. No changes to training cells (14, 15)
7. '{VERSION}' in cell 27 save
