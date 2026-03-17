# vR.P.27 -- Experiment Description

## JPEG Compression Augmentation (Training-Time)

### Hypothesis

Applying random JPEG compression at training time (Q=50-95) makes the model robust to compression artifacts that vary between images, improving generalization. Unlike P.18 (which tests robustness), this experiment trains with compression augmentation to build robustness directly into the model.

### Motivation

CASIA v2.0 images have variable compression quality. Real-world deployment encounters images compressed at unknown quality levels. Training with random compression augmentation teaches the model that compression artifacts are NOT tampering signal -- only genuine manipulation artifacts should be detected.

This is distinct from P.12 (geometric augmentation) and P.18 (robustness evaluation). P.27 adds compression-specific augmentation to training.

### Single Variable Changed from vR.P.3

**Data augmentation** -- Add random JPEG compression augmentation (Q=50-95) during training. Architecture unchanged.

### Key Configuration

| Parameter | P.3 (parent) | P.27 (this) |
|-----------|-------------|-------------|
| Augmentation | None | Random JPEG compression (Q=50-95, p=0.5) |
| Pipeline | image -> ELA -> model | image -> random_JPEG(Q) -> ELA -> model |
| Architecture | Unchanged | Unchanged |
| Everything else | Same | Same |

### Pipeline

```
Training:
    Image -> Random JPEG recompress (Q=50-95, p=0.5) -> ELA(Q=90) -> normalize -> UNet

Validation/Test:
    Image -> ELA(Q=90) -> normalize -> UNet (no compression augmentation)
```

### Implementation Notes

```python
def jpeg_compression_augmentation(image_path, min_quality=50, max_quality=95, p=0.5):
    """Apply random JPEG compression before ELA computation."""
    img = Image.open(image_path).convert('RGB')
    if random.random() < p:
        q = random.randint(min_quality, max_quality)
        buffer = BytesIO()
        img.save(buffer, 'JPEG', quality=q)
        buffer.seek(0)
        img = Image.open(buffer)
    return img  # then compute ELA on this (possibly recompressed) image
```

### Expected Impact

+1-3pp Pixel F1 on standard test set. Significant robustness improvement on compressed test sets (P.18 protocol).

### Risk

Aggressive compression (Q=50) may destroy the ELA signal entirely for that sample, providing a noisy training signal. May need to raise min_quality to 70.
