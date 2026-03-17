# Model Architecture

---

## Base Architecture

**Model:** `smp.Unet` from `segmentation-models-pytorch`
**Encoder:** ResNet34, pretrained on ImageNet
**Parameters:** ~24.4 million (all trainable)

```python
model = smp.Unet(
    encoder_name=config['encoder_name'],      # 'resnet34'
    encoder_weights=config['encoder_weights'], # 'imagenet'
    in_channels=config['in_channels'],         # 3 (RGB)
    classes=config['classes'],                 # 1 (binary mask)
    activation=None,                           # Raw logits — sigmoid applied externally
)
```

**Why U-Net?** U-Net is the standard architecture for dense prediction tasks where you need to combine high-level semantic features (from the encoder) with spatial detail (via skip connections). For tamper localization, the model must identify *what* is tampered (semantic) and *where* exactly the boundary is (spatial). U-Net's encoder-decoder structure with skip connections addresses both needs.

**Why ResNet34?** It provides a strong quality-to-cost ratio. ResNet34 has ~21.8M parameters in the encoder (vs ~25.6M for ResNet50) and trains faster on a T4 GPU. ImageNet pretraining gives the encoder general visual features that transfer well to forensic tasks — the encoder can already extract edges, textures, and compression artifacts before any fine-tuning. Deeper encoders (ResNet50, EfficientNet-B4) would increase VRAM usage and training time without guaranteed improvement on a dataset this size (~1000 tampered images).

**Why `activation=None`?** The loss function (`BCEWithLogitsLoss` inside BCEDiceLoss) expects raw logits for numerical stability. Sigmoid is applied separately when computing predictions.

---

## Model Instantiation via `setup_model()`

The v6.5 notebook wraps model creation in a dedicated function:

```python
def setup_model(config, device):
    """Create model, optionally wrap in DataParallel, and verify output shape."""
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights=config['encoder_weights'],
        in_channels=config['in_channels'],
        classes=config['classes'],
        activation=None,
    )
    model = model.to(device)

    # Optional multi-GPU
    is_parallel = False
    if torch.cuda.device_count() > 1 and config['use_multi_gpu']:
        model = torch.nn.DataParallel(model)
        is_parallel = True

    # Shape verification
    with torch.no_grad():
        dummy = torch.randn(1, 3, config['image_size'], config['image_size']).to(device)
        out = model(dummy)
        assert out.shape == (1, 1, config['image_size'], config['image_size'])

    return model, is_parallel
```

**Why shape verification?** A dummy forward pass catches configuration mismatches immediately (wrong `in_channels`, incompatible encoder, etc.) instead of failing silently during training.

**Why `is_parallel`?** The flag is threaded through the notebook to handle DataParallel-specific logic:
- **Optimizer setup:** `model.module.encoder.parameters()` vs `model.encoder.parameters()`
- **Checkpoint save:** Always saves `model.module.state_dict()` for portability
- **Checkpoint load:** Handles `module.` prefix mismatch in both directions
- **Grad-CAM:** Hooks into `model.module.encoder.layer4` when parallelized

---

## DataParallel Support

When `CONFIG['use_multi_gpu']` is True and multiple GPUs are detected:

```python
model = torch.nn.DataParallel(model)
```

DataParallel replicates the model across GPUs and splits each batch along the batch dimension. Gradients are reduced automatically.

**When does this apply?** Some Colab instances offer 2× T4 GPUs. Kaggle typically provides a single T4. The flag control ensures single-GPU environments are unaffected — no performance overhead, no code branching.

**Checkpoint portability:** Checkpoints always save the unwrapped model state (`model.module.state_dict()` if parallel, `model.state_dict()` otherwise). Loading checks for and strips the `module.` prefix, so checkpoints are portable across single-GPU and multi-GPU environments.

---

## Image-Level Detection

The model outputs a pixel-level probability map. For image-level tamper detection (binary: tampered or authentic), a **top-k mean** heuristic is used:

```python
def compute_image_score(prob_map, k_pct=0.01):
    """Image-level tamper score from top-k% pixel probabilities."""
    flat = prob_map.flatten()
    k = max(1, int(len(flat) * k_pct))
    topk = torch.topk(flat, k).values
    return topk.mean().item()
```

**Why top-k mean instead of max?** `max(prob_map)` is sensitive to single-pixel noise — one spurious high-probability pixel would flag the entire image. Top-k mean (top 1% of pixels) is more robust: it requires a spatially coherent cluster of high-probability pixels.

**Limitation:** This is a heuristic, not a learned classifier. Research papers (P2, reference notebook) implement dedicated classification heads for image-level detection. A dual-task head is listed as future work.

---

## Encoder Alternatives

| Encoder | Params (M) | ImageNet Top-1 | T4 VRAM (est.) | Notes |
|---|---|---|---|---|
| **ResNet34** | 21.8 | 73.3% | ~4 GB | **Selected** — fast, proven, sufficient for CASIA scale |
| ResNet50 | 25.6 | 76.1% | ~5 GB | Marginal accuracy gain; ~30% slower training |
| EfficientNet-B3 | 12.2 | 81.6% | ~6 GB | Better ImageNet accuracy; newer architecture |
| EfficientNet-B4 | 19.3 | 82.9% | ~8 GB | Strong features; higher VRAM |
| ResNeXt-50 | 25.0 | 77.6% | ~6 GB | Group convolutions; may help with texture features |

**Why not EfficientNet?** EfficientNet encoders have higher ImageNet accuracy but also higher VRAM usage due to wider layers. On a T4 with batch size 4 at 384×384, ResNet34 leaves comfortable headroom. EfficientNet-B4 would require reducing batch size, which conflicts with the gradient accumulation strategy.

---

## SMP Library Advantages

Using `segmentation-models-pytorch` instead of a custom U-Net implementation provides:

1. **Pretrained encoders** — 400+ encoder variants with ImageNet/ImageNet-21k weights
2. **Verified skip connections** — Skip connection wiring is well-tested
3. **Consistent API** — Swapping encoders requires only changing `encoder_name`
4. **Community maintained** — Bug fixes and new architectures without custom code

**Trade-off:** Less flexibility than a custom implementation. Custom attention mechanisms or edge-supervision modules would require modifying the decoder, which SMP supports but is less straightforward.

---

## Interview: "Why not a transformer-based model?"

Transformer-based models (e.g., Swin Transformer in EMT-Net, self-attention in TransU²-Net) show strong results in the research literature. However:

1. **VRAM:** Vision Transformers require significantly more memory than CNNs at the same resolution. A Swin-based model at 384×384 would not fit on a T4 with useful batch sizes.
2. **Data scale:** Transformers benefit from large datasets. CASIA has ~1000 tampered images — too small for a transformer to learn effectively without extensive pretraining.
3. **Assignment scope:** The assignment asks for a functional pipeline, not a frontier model. ResNet34 + U-Net is a well-understood baseline that can be trained, evaluated, and explained within the assignment timeline.
4. **Extensibility:** The SMP library supports transformer encoders. Upgrading is a one-line config change once hardware constraints allow it.
