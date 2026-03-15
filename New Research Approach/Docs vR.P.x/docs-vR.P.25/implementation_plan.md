# vR.P.25 — Implementation Plan

## Core Implementation: Edge Supervision Loss (Sobel Edge Loss)

### Edge Map Computation

```python
def compute_edge_map(mask):
    """Compute Sobel edge map from ground truth mask."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    # Apply Sobel filters to mask
    edges_x = F.conv2d(mask, sobel_x.view(1,1,3,3), padding=1)
    edges_y = F.conv2d(mask, sobel_y.view(1,1,3,3), padding=1)
    edge_map = torch.sqrt(edges_x**2 + edges_y**2)
    return (edge_map > 0.1).float()  # binary edge map
```

### Combined Loss

```python
loss_seg = criterion(pred_mask, gt_mask)  # existing Dice+BCE
loss_edge = F.binary_cross_entropy_with_logits(pred_mask, edge_map, reduction='mean')
loss = loss_seg + LAMBDA_EDGE * loss_edge
```

The edge loss forces the model to pay extra attention to forgery boundaries, where prediction errors are most common.

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.25 — Edge Supervision Loss (Sobel Edge Loss)" |
| 1 | Changelog | Add P.25 entry: edge-aware loss with lambda=0.3 |
| 2 | Setup | VERSION='vR.P.25', LAMBDA_EDGE=0.3 |
| 13 | Training config | Describe edge supervision strategy: Sobel edge extraction from GT masks, weighted BCE on boundary pixels |
| 14 | Loss function | Add `compute_edge_map()` function + modify combined loss to include edge BCE term |
| 25 | Results table | Note "Edge Loss lambda=0.3" in config column |
| 26 | Discussion | Edge supervision hypothesis, boundary quality analysis |
| 27 | Save model | Config includes lambda_edge=0.3 |

### Unchanged Cells

Cells 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain unchanged from P.3. This is a loss-only modification — no changes to data pipeline, model architecture, or evaluation.

### Key New Code

- `compute_edge_map(mask)` (~15 lines): Sobel filter application to ground truth masks
- Modified loss computation in Cell 14 (~5 lines): adds edge BCE term with LAMBDA_EDGE weighting
- Total new code is minimal (~20 lines), making this a clean ablation of edge supervision

### Verification Checklist

- [ ] `compute_edge_map()` produces binary edge maps with reasonable edge width (1-2 pixels)
- [ ] Edge maps are non-zero for tampered images and zero for authentic images
- [ ] LAMBDA_EDGE=0.3 does not dominate the total loss (edge loss should be ~20-30% of total)
- [ ] Training converges normally (no divergence from edge term)
- [ ] Boundary quality visually improves in prediction overlay visualizations
- [ ] All metric cells execute and produce valid Pixel F1 / IoU / AUC values
- [ ] Sobel kernels are on correct device (CUDA) during training

### Risks

- Edge loss may conflict with interior fill — model might learn sharp boundaries but hollow interiors
- LAMBDA_EDGE=0.3 may be too strong or too weak — may need tuning
- Sobel edge maps from low-resolution GT masks may be too coarse
