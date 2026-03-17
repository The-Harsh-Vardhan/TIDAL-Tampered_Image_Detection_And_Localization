# Implementation Plan — vR.P.10: ELA + Attention Modules (CBAM)

## 1. Cell-by-Cell Changes from vR.P.3

| Cell | Type | Change |
|------|------|--------|
| 0 | Markdown | Title → "vR.P.10 — ELA + Attention Modules (CBAM)", update pipeline diagram, parent, change description |
| 1 | Markdown | Add P.10 changelog entry |
| 2 | Code | `VERSION='vR.P.10'`, `CHANGE` updated, add `ATTENTION_TYPE`, `ATTENTION_REDUCTION`, `CBAM_KERNEL_SIZE`, `DECODER_CHANNELS`, `FOCAL_ALPHA`, `FOCAL_GAMMA`, `NUM_WORKERS=4`, `PREFETCH_FACTOR=2` |
| 9 | Code | Add `prefetch_factor=PREFETCH_FACTOR` to DataLoaders |
| 11 | Markdown | Update architecture description to include attention |
| 12 | Code | Add SEBlock/CBAMBlock class definitions + injection after model creation |
| 14 | Code | Replace BCE loss with Focal loss from P.9 |
| 22-27 | Code | Fix P.3's NameError bug (`denormalize` → `denormalize_ela`) |
| 25 | Code | Tracking table → add P.10 live row, include P.3 hardcoded row |
| 26 | Markdown | Discussion → attention module rationale, compare with P.3 |
| 27 | Code | Model save → `vR.P.10` filename, add attention_type and focal params to config |

All other cells remain IDENTICAL to vR.P.3: ELA dataset class, ELA normalization, freeze strategy, training loop, evaluation, metrics.

---

## 2. Key Implementation Details

### 2.1 Configuration Changes (Cell 2)

```python
# --- CHANGED from P.3 ---
VERSION = 'vR.P.10'
CHANGE = 'Focal+Dice loss + CBAM attention in UNet decoder'
NUM_WORKERS = 4          # was 2 in P.3
PREFETCH_FACTOR = 2      # explicit (was default in P.3)

# --- NEW for P.10 ---
ATTENTION_TYPE = 'cbam'         # 'cbam', 'se', or None
ATTENTION_REDUCTION = 16        # Channel reduction ratio for SE/CBAM
CBAM_KERNEL_SIZE = 7            # Spatial attention conv kernel size
DECODER_CHANNELS = (256, 128, 64, 32, 16)  # SMP UNet default for resnet34
FOCAL_ALPHA = 0.25              # Focal loss alpha (from P.9)
FOCAL_GAMMA = 2.0               # Focal loss gamma (from P.9)

# --- UNCHANGED from P.3 ---
SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 3
NUM_CLASSES = 1
LEARNING_RATE = 1e-3
ELA_QUALITY = 90
EPOCHS = 25
PATIENCE = 7
```

### 2.2 SEBlock Implementation (Cell 12)

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block — channel attention only."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        scale = self.squeeze(x).view(b, c)
        scale = self.excitation(scale).view(b, c, 1, 1)
        return x * scale
```

### 2.3 CBAMBlock Implementation (Cell 12)

```python
class ChannelAttention(nn.Module):
    """Channel attention sub-module for CBAM."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention sub-module for CBAM."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        descriptor = torch.cat([avg, mx], dim=1)
        attention = self.sigmoid(self.conv(descriptor))
        return x * attention


class CBAMBlock(nn.Module):
    """CBAM: Convolutional Block Attention Module.
    Sequential channel attention → spatial attention.
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
```

### 2.4 Attention Injection (Cell 12, after model creation)

```python
# Inject attention into each decoder block's attention2 slot
# SMP's DecoderBlock.forward() calls self.attention2(x) after self.conv2(x)
if ATTENTION_TYPE is not None:
    attn_param_count = 0
    for i, block in enumerate(model.decoder.blocks):
        ch = DECODER_CHANNELS[i]
        if ATTENTION_TYPE == 'cbam':
            block.attention2 = CBAMBlock(ch, ATTENTION_REDUCTION, CBAM_KERNEL_SIZE)
        elif ATTENTION_TYPE == 'se':
            block.attention2 = SEBlock(ch, ATTENTION_REDUCTION)
        attn_param_count += sum(p.numel() for p in block.attention2.parameters())
    print(f'Attention type: {ATTENTION_TYPE.upper()}')
    print(f'Attention params: {attn_param_count:,}')
```

### 2.5 Loss Function Change (Cell 14)

```python
# BEFORE (P.3):
bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss()
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def criterion(pred, target):
    return bce_loss_fn(pred, target) + dice_loss_fn(pred, target)

# AFTER (P.10):
focal_loss_fn = smp.losses.FocalLoss(mode='binary', alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def criterion(pred, target):
    return focal_loss_fn(pred, target) + dice_loss_fn(pred, target)
```

### 2.6 P.3 Bug Fix (Cells 22-27)

```python
# BEFORE (P.3 — broken):
img_display = denormalize(img_tensor).permute(1, 2, 0).numpy()

# AFTER (P.10 — fixed):
img_display = denormalize_ela(img_tensor).permute(1, 2, 0).numpy()
```

---

## 3. Verification Checklist

- [ ] VERSION = 'vR.P.10'
- [ ] ATTENTION_TYPE = 'cbam'
- [ ] ATTENTION_REDUCTION = 16
- [ ] CBAM_KERNEL_SIZE = 7
- [ ] DECODER_CHANNELS = (256, 128, 64, 32, 16)
- [ ] FOCAL_ALPHA = 0.25
- [ ] FOCAL_GAMMA = 2.0
- [ ] SEBlock class defined (for switchability)
- [ ] CBAMBlock class defined (ChannelAttention + SpatialAttention)
- [ ] Attention injection loop replaces block.attention2 for all 5 decoder blocks
- [ ] Loss uses FocalLoss + DiceLoss (not BCE)
- [ ] EPOCHS = 25 (unchanged)
- [ ] PATIENCE = 7 (unchanged)
- [ ] LEARNING_RATE = 1e-3 (unchanged)
- [ ] ELA_QUALITY = 90 (unchanged)
- [ ] Encoder frozen except BN layers (unchanged)
- [ ] `denormalize_ela` bug fixed in visualization cells
- [ ] NUM_WORKERS = 4
- [ ] PREFETCH_FACTOR = 2
- [ ] AMP + TF32 enabled
- [ ] Model saved as `vR.P.10_unet_resnet34_model.pth`
- [ ] Saved config includes attention_type, focal_alpha, focal_gamma

---

## 4. Parameter Impact

| Component | P.3 (No Attention) | P.10 (With CBAM) |
|-----------|-------------------|-------------------|
| Encoder (frozen conv) | 21,267,648 | 21,267,648 |
| Encoder BN (trainable) | 17,024 | 17,024 |
| Decoder conv blocks | ~3,151,552 | ~3,151,552 |
| **Decoder attention** | **0** | **~11,200** |
| Segmentation head | 145 | 145 |
| **Total trainable** | **~3,168,721** | **~3,179,921** |

Attention adds ~11.2K parameters (0.35% increase in trainable params). This is negligible — the experiment isolates the attention mechanism's inductive bias, not its parameter capacity.

---

## 5. Runtime Estimate

| Component | Time |
|-----------|------|
| ELA stats computation (500 images) | ~30s |
| Training per epoch | ~3-4 min (with AMP, 4 workers) |
| Attention overhead per epoch | < 5s (CBAM is very lightweight) |
| Best case (early stop at ~18) | ~60-75 min |
| Worst case (full 25 epochs) | ~85-110 min |
| Evaluation | ~5 min |
| **Total session** | **~65-120 min** |

Well within Kaggle T4/P100 session limits.
