# vR.P.13 — Implementation Plan

## Base Notebook
`vR.P.3 Image Detection and Localisation.ipynb` (from git HEAD)

## Cells Modified (14 of 28)

| Cell | Type | Change |
|------|------|--------|
| 0 | md | Title: "Combined Best-of (CBAM + Augmentation + 50 Epochs)" |
| 1 | md | Changelog: add P.13 entry, diff table P.3 → P.13 |
| 2 | code | VERSION, CHANGE, EPOCHS=50, PATIENCE=10, FOCAL_ALPHA/GAMMA, attention constants, `import albumentations as A`, NUM_WORKERS=4, PREFETCH_FACTOR=2, cudnn.benchmark=True |
| 7 | md | Data preparation: mention augmentation pipeline |
| 8 | code | Add `transform` parameter to Dataset class (from P.12 pattern) |
| 9 | code | Augmentation pipeline definition + train_transform, DataLoaders with prefetch |
| 11 | md | Architecture: add CBAM description in decoder diagram |
| 12 | code | CBAM classes + inject into decoder blocks (from P.10) |
| 13 | md | Training config: Focal+Dice, 50 epochs, patience 10, augmentation |
| 14 | code | Focal+Dice loss (from P.10), standard train_one_epoch/validate |
| 25 | code | Results table: add P.3, P.7, P.10 baselines, "Combined" input column |
| 26 | md | Discussion: combo hypothesis, stacking question |
| 27 | code | Model save: config includes augmentation + attention type |

## Unchanged Cells
3–6, 10, 15–24 — dataset discovery, ELA functions, visualization, training loop, evaluation

## Verification Checklist
1. Cell count == 28
2. VERSION == 'vR.P.13'
3. EPOCHS = 50
4. PATIENCE = 10
5. FOCAL_ALPHA = 0.25
6. FOCAL_GAMMA = 2.0
7. ATTENTION_TYPE = 'cbam'
8. 'albumentations' in cell 2 imports
9. 'train_transform' defined in cell 9 (augmentation pipeline)
10. CBAMBlock class in cell 12
11. FocalLoss in cell 14
12. 'transform' parameter in Dataset class (cell 8)
13. '{VERSION}' in cell 27 save
