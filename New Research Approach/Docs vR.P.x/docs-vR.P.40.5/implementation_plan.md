# vR.P.40.5 — Implementation Plan

## Core Implementation: InceptionV3 Custom Encoder

### InceptionV3 Module Design

```python
class InceptionV3Module(nn.Module):
    # branch1: 1x1 conv + BN + ReLU
    # branch2: 1x1 reduce + BN + 3x3 conv + BN + ReLU
    # branch3: 1x1 reduce + BN + 1xn conv + BN + nx1 conv + BN (asymmetric)
    # branch4: AvgPool 3x3 + 1x1 conv + BN + ReLU
    # Parameter n is adaptive per stage
```

### Asymmetric Factorization Detail

A 7x7 convolution has 49 parameters per channel pair. The 1x7 + 7x1 factorization uses only 14 parameters (71% reduction). This provides:
- Better regularization (fewer parameters to overfit)
- Directional feature decomposition (horizontal then vertical)
- Same effective receptive field

### Adaptive n per Stage

| Stage | Feature map size | n (kernel) | Rationale |
|-------|-----------------|------------|-----------|
| 1 | 192x192 | 3 | Small features, small kernel |
| 2 | 96x96 | 5 | Medium features |
| 3 | 48x48 | 5 | Medium features |
| 4 | 24x24 | 3 | Small feature maps need small kernel |

### Cell Modification Map

| Cell | Action |
|------|--------|
| 0 | Title: InceptionV3 custom encoder |
| 2 | ENCODER='inception-v3-custom' |
| 12 | InceptionV3Module + InceptionV3Encoder + SMP registration |
| 14 | All params trainable, Adam LR=1e-3 |
| 26 | Discussion: asymmetric factorization analysis |

### Risks

- Asymmetric kernels cannot capture diagonal patterns directly
- Adaptive n adds hyperparameter complexity
- 1xn + nx1 sequential application may lose some cross-directional correlations

### Verification Checklist

- [ ] InceptionV3Module correctly applies 1xn then nx1 convolution
- [ ] Output shape matches V1/V2 for same input
- [ ] Total parameter count < V2 (factorization should reduce params)
- [ ] Training stability comparable to V2 (both have BN)
- [ ] Compare V1 < V2 < V3 progression validates progressive improvements
