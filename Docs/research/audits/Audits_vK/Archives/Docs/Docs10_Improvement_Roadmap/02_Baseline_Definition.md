# 2. Baseline Definition

The baseline code configuration is `vK.7.1 Image Detection and Localisation.ipynb`.

## Dataset

- CASIA-style authentic (`Au`) and tampered (`Tp`) images with binary masks
- Kaggle-first dataset root: `casia-spicing-detection-localization`
- `70/15/15` split via stratified `train_test_split`

## Model

- Custom U-Net-style encoder-decoder
- Channel progression: `64 -> 128 -> 256 -> 512 -> 1024`
- One-channel segmentation head
- Classification head:
  - `AdaptiveAvgPool2d`
  - `Flatten`
  - `Linear(1024, 512)`
  - `ReLU`
  - `Dropout(0.5)`
  - `Linear(512, 2)`

## Training

- Image size: `256`
- Batch size: `8`
- Workers: `2`
- Focal-style classification loss with balanced class weights
- Segmentation loss: `0.5 * BCEWithLogits + 0.5 * Dice`
- `ALPHA = 1.5`
- `BETA = 1.0`
- `Adam(lr=1e-4)`
- `CosineAnnealingLR(T_max=10)`
- `50` epochs
- Gradient clipping: `max_norm=5.0`
- Best checkpoint chosen by validation accuracy

## Current Evaluation State

- Implemented metrics: accuracy, Dice, IoU, F1
- Missing from the baseline artifact: ROC-AUC and an executed final four-panel proof
- Current localization metrics are not tampered-only

## Baseline Prerequisite

Run `vK.7.1` unchanged once and save the output-complete notebook. That executed run becomes the baseline control for all 10.x comparisons.
