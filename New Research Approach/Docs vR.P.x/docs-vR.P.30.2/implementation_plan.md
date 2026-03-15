# vR.P.30.2 -- Implementation Plan

## Cells Modified from Parent (vR.P.30)

[0,1,2] Header/Config, [12] Model (+freeze helpers from P.8), [14] Loss/Optimizer (progressive), [15] Training loop (stage-aware from P.8), [25,26] Results/Discussion

## Key Code Sources

Merges P.15 (dataset) + P.10 (CBAM) + P.8 (progressive unfreeze). rebuild_optimizer() includes CBAM params in decoder group.

## Verification Checklist

- [ ] VERSION = 'vR.P.30.2' in Cell [2]
- [ ] ATTENTION_TYPE = 'cbam' in Cell [2]
- [ ] DECODER_CHANNELS = (256,128,64,32,16) in Cell [2]
- [ ] CBAMBlock class defined in Cell [12]
- [ ] Decoder injection loop in Cell [12]
- [ ] W&B config includes attention_type, attention_reduction, cbam_kernel_size
- [ ] All 28 cells present
- [ ] Notebook runs without errors on Kaggle
