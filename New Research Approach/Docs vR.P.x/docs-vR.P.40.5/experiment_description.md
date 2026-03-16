# vR.P.40.5 -- Experiment Description

## InceptionV3 Custom Encoder + Multi-Q RGB ELA (9ch)

### Hypothesis

InceptionV3's asymmetric 1xn + nx1 factorization provides the most parameter-efficient multi-scale feature extraction, offering better regularization for small-dataset training while capturing directional features (horizontal + vertical tamper edges) that symmetric kernels may miss.

### Motivation

InceptionV3 is the most advanced Inception variant, replacing square nxn convolutions with asymmetric 1xn + nx1 pairs. This reduces parameters by ~(2/n) while decomposing feature extraction into directional components. For image tampering detection, edge artifacts often have directional structure (splicing boundaries, copy-move edges), making asymmetric factorization potentially well-suited.

### Single Variable Changed from vR.P.40.4

**Inception module version** -- Replace InceptionV2 (factorized 5x5 → two 3x3) with InceptionV3 (asymmetric 1xn + nx1). Same BN, same AvgPool, same encoder dimensions.

### Key Configuration

| Parameter | P.40.4 (parent) | P.40.5 (this) |
|-----------|-----------------|---------------|
| Inception version | V2 (two 3x3) | V3 (1xn + nx1) |
| BatchNorm | Yes | Yes |
| Pool type | AvgPool | AvgPool |
| Branch 3 design | 1x1→3x3→3x3 | 1x1→1xn→nx1 |
| Asymmetric n | N/A | Adaptive (3-7) |

### InceptionV3 Module Architecture

```
Branch 1: 1x1 conv + BN + ReLU
Branch 2: 1x1 reduce + 3x3 conv + BN + ReLU
Branch 3: 1x1 reduce + 1xn conv + nx1 conv + BN + ReLU  (asymmetric)
Branch 4: AvgPool 3x3 + 1x1 conv + BN + ReLU
```

### Pipeline

```
Image -> Multi-Q RGB ELA (9ch, Q=75/85/95)
      -> InceptionV3 Encoder (5 stages, asymmetric conv, BN, AvgPool)
      -> UNet Decoder (skip connections)
      -> Sigmoid -> 384x384 binary mask
```

### Expected Impact

Most parameter-efficient Inception variant. Expected F1: 0.60–0.72. Best regularization for small dataset.

### Risk

Asymmetric kernels may not capture diagonal or radial tamper patterns. Adaptive n across stages adds complexity.
