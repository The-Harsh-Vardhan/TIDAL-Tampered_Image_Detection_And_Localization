# vR.P.30.3 -- Implementation Plan

## Cells Modified from Parent (vR.P.30)

[0,1,2] Header/Config (+FOCAL_ALPHA/GAMMA), [14] Loss (FocalLoss replaces SoftBCE), [25,26] Results/Discussion

## Key Code Sources

smp.losses.FocalLoss(mode='binary', alpha=0.25, gamma=2.0). P.9 showed Focal was neutral alone; tests CBAM interaction.

## Verification Checklist

- [ ] VERSION = 'vR.P.30.3' in Cell [2]
- [ ] ATTENTION_TYPE = 'cbam' in Cell [2]
- [ ] DECODER_CHANNELS = (256,128,64,32,16) in Cell [2]
- [ ] CBAMBlock class defined in Cell [12]
- [ ] Decoder injection loop in Cell [12]
- [ ] W&B config includes attention_type, attention_reduction, cbam_kernel_size
- [ ] All 28 cells present
- [ ] Notebook runs without errors on Kaggle
