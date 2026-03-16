# vR.P.40.4 — Implementation Plan

## Core Implementation: InceptionV2 Custom Encoder

### InceptionV2 Module Design

```python
class InceptionV2Module(nn.Module):
    # branch1: 1x1 conv + BN + ReLU
    # branch2: 1x1 reduce + BN + 3x3 conv + BN + ReLU
    # branch3: 1x1 reduce + BN + 3x3 + BN + 3x3 + BN (factorized 5x5)
    # branch4: AvgPool 3x3 + 1x1 conv + BN + ReLU
```

Key differences from V1:
- All convolutions use `bias=False` + `BatchNorm2d` + `ReLU`
- 5x5 branch replaced by two sequential 3x3 convolutions
- MaxPool replaced by AvgPool in pool branch

### Stage Architecture

Same dimensions as V1 but with BN and AvgPool:

| Stage | Inception out | Proj out | Pool |
|-------|---------------|----------|------|
| Stem | 32 (conv) | 32 | stride 2 |
| 1 | 72 | 128 | AvgPool 2x2 |
| 2 | 176 | 240 | AvgPool 2x2 |
| 3 | 264 | 336 | AvgPool 2x2 |
| 4 | 352 | 432 | AvgPool 2x2 |

### Cell Modification Map

| Cell | Action |
|------|--------|
| 0 | Title: InceptionV2 custom encoder |
| 2 | ENCODER='inception-v2-custom' |
| 12 | InceptionV2Module + InceptionV2Encoder + SMP registration |
| 14 | All params trainable, Adam LR=1e-3 |

### Risks

- Factorized 5x5 may miss some artifact patterns that require the full receptive field
- BN with small batches (8) may have noisy statistics
- AvgPool may blur fine-grained tamper boundaries

### Verification Checklist

- [ ] InceptionV2Encoder builds and forward passes without error
- [ ] out_channels = [9, 32, 128, 240, 336, 432]
- [ ] All BN layers initialized correctly
- [ ] Training is more stable than V1 (no NaN within first 5 epochs)
- [ ] Compare directly with P.40.3 to measure BN+factorization impact
