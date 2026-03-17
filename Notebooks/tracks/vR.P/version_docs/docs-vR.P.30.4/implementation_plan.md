# vR.P.30.4 -- Implementation Plan

## Cells Modified from Parent (vR.P.30.1)

[0,1,2] Header/Config (+albumentations), [8] Dataset (transform param), [9] Split (train_transform), [25,26] Results/Discussion

## Key Code Sources

Albumentations geometric pipeline (no brightness/blur to preserve ELA signal). Applied to multi-Q ELA array.

## Verification Checklist

- [ ] VERSION = 'vR.P.30.4' in Cell [2]
- [ ] ATTENTION_TYPE = 'cbam' in Cell [2]
- [ ] DECODER_CHANNELS = (256,128,64,32,16) in Cell [2]
- [ ] CBAMBlock class defined in Cell [12]
- [ ] Decoder injection loop in Cell [12]
- [ ] W&B config includes attention_type, attention_reduction, cbam_kernel_size
- [ ] All 28 cells present
- [ ] Notebook runs without errors on Kaggle
