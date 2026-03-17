# vR.P.30 -- Implementation Plan

## Cells Modified from Parent (vR.P.15)

[0] Title, [1] Changelog, [2] Config, [12] Model (+CBAM), [25] Results, [26] Discussion

## Key Code Sources

CBAM classes (SEBlock, ChannelAttention, SpatialAttention, CBAMBlock) from P.10 Cell [12]. Decoder injection loop from P.10.

## Verification Checklist

- [ ] VERSION = 'vR.P.30' in Cell [2]
- [ ] ATTENTION_TYPE = 'cbam' in Cell [2]
- [ ] DECODER_CHANNELS = (256,128,64,32,16) in Cell [2]
- [ ] CBAMBlock class defined in Cell [12]
- [ ] Decoder injection loop in Cell [12]
- [ ] W&B config includes attention_type, attention_reduction, cbam_kernel_size
- [ ] All 28 cells present
- [ ] Notebook runs without errors on Kaggle
