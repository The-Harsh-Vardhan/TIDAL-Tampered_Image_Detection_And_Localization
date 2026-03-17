## Experiment Leaderboard (Top 15 by Pixel F1)

|   Rank | Version   | Change                                                                               |   Pixel F1 |    IoU |    AUC |
|-------:|:----------|:-------------------------------------------------------------------------------------|-----------:|-------:|-------:|
|      1 | vR.P.19   | Multi-Quality RGB ELA (9ch, Q=75/85/95 full-color)                                   |     0.7965 | 0.6618 | 0.9726 |
|      2 | vR.P.19   | Multi-Quality RGB ELA (9ch, Q=75/85/95 full-color)                                   |     0.7965 | 0.6618 | 0.9726 |
|      3 | vR.P.30.1 | Multi-quality ELA + CBAM attention (50ep, BCE+Dice)                                  |     0.7762 | 0.6343 | 0.9795 |
|      4 | vR.P.30.2 | Multi-quality ELA + CBAM + Progressive Unfreeze (40ep, BCE+Dice)                     |     0.7719 | 0.6286 | 0.9755 |
|      5 | vR.P.30.2 | Multi-quality ELA + CBAM + Progressive Unfreeze (40ep, BCE+Dice)                     |     0.7719 | 0.6286 | 0.9755 |
|      6 | vR.P.30.4 | Multi-quality ELA + CBAM + Geometric Augmentation (50ep, BCE+Dice)                   |     0.7662 | 0.6210 | 0.9726 |
|      7 | vR.P.40.1 | EfficientNet-B4 baseline (ELA Q=90, 3ch)                                             |     0.7550 | 0.6064 | 0.9678 |
|      8 | vR.P.40.1 | EfficientNet-B4 baseline (ELA Q=90, 3ch)                                             |     0.7550 | 0.6064 | 0.9678 |
|      9 | vR.P.30.3 | Multi-quality ELA + CBAM + Focal+Dice loss (25ep)                                    |     0.7509 | 0.6011 | 0.9694 |
|     10 | vR.P.30   | Multi-quality ELA + CBAM attention (25ep, BCE+Dice)                                  |     0.7438 | 0.5921 | 0.9733 |
|     11 | vR.P.15   | Multi-quality ELA input (Q=75/85/95 as 3 channels)                                   |     0.7329 | 0.5785 | 0.9608 |
|     12 | vR.P.13   | Combined: CBAM attention + Albumentations augmentation + 50 epochs + Focal+Dice loss |     0.7307 | 0.5756 | 0.9607 |
|     13 | vR.P.17   | ELA + DCT fusion (6-channel input)                                                   |     0.7302 | 0.5751 | 0.9431 |
|     14 | vR.P.10   | Focal+Dice loss + CBAM attention in UNet decoder                                     |     0.7238 | 0.5672 | 0.9525 |
|     15 | vR.P.28   | Cosine Annealing LR Scheduler (T_0=10, T_mult=2, 50 epochs)                          |     0.7215 | 0.5643 | 0.9572 |
