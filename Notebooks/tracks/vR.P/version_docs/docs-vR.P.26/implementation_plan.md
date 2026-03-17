# vR.P.26 — Implementation Plan

## Core Implementation: Dual-Task Segmentation + Classification Head

### DualTaskUNet Architecture

```python
class DualTaskUNet(nn.Module):
    def __init__(self, smp_model, num_classes=1):
        super().__init__()
        self.seg_model = smp_model  # existing SMP UNet
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.seg_model.encoder(x)
        seg_out = self.seg_model.decoder(*features)
        seg_mask = self.seg_model.segmentation_head(seg_out)
        cls_out = self.cls_head(features[-1])  # deepest encoder features
        return seg_mask, cls_out
```

### Combined Loss

```python
loss_seg = criterion(pred_mask, gt_mask)       # Dice + BCE
loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_label)
loss = loss_seg + CLS_WEIGHT * loss_cls
```

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.26 — Dual-Task Segmentation + Classification Head" |
| 1 | Changelog | Add P.26 entry: dual-task with CLS_WEIGHT=0.5 |
| 2 | Setup | VERSION='vR.P.26', CLS_WEIGHT=0.5 |
| 11 | Architecture header | Describe dual-task: shared encoder, segmentation decoder + classification head from bottleneck features |
| 12 | Model build | Define `DualTaskUNet` class wrapping SMP model; classification head branches from deepest encoder features |
| 14 | Loss function | Combined segmentation + classification loss with CLS_WEIGHT weighting |
| 15 | Training loop | `forward()` returns `(seg_mask, cls_logit)` tuple; compute both losses; backprop combined loss |
| 17 | Evaluation header | "Evaluating both segmentation and classification performance" |
| 18 | Evaluation | Compute seg metrics (Pixel F1, IoU, AUC) AND cls metrics (accuracy, precision, recall, F1) separately |
| 25 | Results table | Two-row table: segmentation metrics + classification metrics |
| 26 | Discussion | Dual-task synergy hypothesis, whether classification head regularizes segmentation |
| 27 | Save model | Config includes cls_weight=0.5, model type='DualTaskUNet' |

### Unchanged Cells

Cells 3, 4, 5, 6, 7, 8, 9, 10, 13, 16, 19, 20, 21, 22, 23, 24 remain unchanged from P.3. The data pipeline, dataset class, and visualization cells are unmodified.

### Key New Code

- `DualTaskUNet` class (~30 lines): wraps SMP model with classification head
- Modified training loop (~10 lines changed): unpacks tuple output, computes dual loss
- Modified evaluation (~15 lines): computes both seg and cls metrics
- Classification head: AdaptiveAvgPool2d -> Flatten -> Linear(encoder_dim, 256) -> ReLU -> Dropout -> Linear(256, 1)

### Verification Checklist

- [ ] `DualTaskUNet` forward pass returns tuple of (seg_mask, cls_logit) with correct shapes
- [ ] Classification head input dimension matches encoder bottleneck output channels
- [ ] Combined loss balances properly (CLS_WEIGHT=0.5 means cls loss is ~33% of total)
- [ ] Training loop correctly unpacks the tuple and computes both losses
- [ ] Evaluation computes both segmentation and classification metrics independently
- [ ] Image-level classification accuracy is reported separately from pixel-level metrics
- [ ] Model checkpoint saves the full DualTaskUNet (including cls_head weights)
- [ ] No gradient conflicts between segmentation and classification objectives

### Risks

- Classification head gradient may interfere with segmentation decoder through shared encoder
- CLS_WEIGHT=0.5 may be too strong — classification is easier and may dominate encoder updates
- AdaptiveAvgPool2d on encoder features may lose spatial information needed for segmentation
