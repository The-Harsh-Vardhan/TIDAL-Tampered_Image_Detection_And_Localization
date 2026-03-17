# vR.P.25 -- Experiment Description

## Edge Supervision Loss

### Hypothesis

Adding an auxiliary edge-aware loss that penalizes prediction errors at tampered region boundaries improves boundary precision and recall, yielding sharper localization masks. The standard Dice+BCE loss treats all pixels equally; edge supervision forces the model to focus on the hardest pixels (boundaries).

### Motivation

Current models achieve decent area overlap (IoU ~0.57) but boundary accuracy is poor -- predictions are "blobby" with imprecise edges. An edge supervision branch computes the Sobel/Canny edges of both the predicted mask and the ground truth mask, then applies a binary cross-entropy loss between them. This directly penalizes boundary misalignment.

Edge supervision has been shown to improve segmentation boundary quality in medical imaging (BASNet, U2-Net) and is directly applicable to forensic segmentation where sharp boundaries are forensically meaningful.

### Single Variable Changed from vR.P.3

**Loss function** -- Add edge supervision auxiliary loss to existing BCE+Dice. Architecture unchanged.

### Key Configuration

| Parameter | P.3 (parent) | P.25 (this) |
|-----------|-------------|-------------|
| Loss | BCE + Dice | BCE + Dice + EdgeBCE (weighted sum) |
| Edge weight | N/A | 0.3 (lambda_edge) |
| Edge extraction | N/A | Sobel filter on predicted & GT masks |
| Architecture | Unchanged | Unchanged |
| Everything else | Same | Same |

### Pipeline

```
GT mask -> Sobel edges -> GT_edges
Predicted mask -> Sobel edges -> Pred_edges
    |
    v
Loss = BCE(pred, gt) + Dice(pred, gt) + 0.3 * BCE(Pred_edges, GT_edges)
```

### Implementation Notes

```python
def compute_edge_map(mask, threshold=0.5):
    """Differentiable Sobel edge extraction."""
    sobel_x = F.conv2d(mask, SOBEL_KERNEL_X, padding=1)
    sobel_y = F.conv2d(mask, SOBEL_KERNEL_Y, padding=1)
    edges = torch.sqrt(sobel_x**2 + sobel_y**2 + 1e-6)
    return edges

edge_loss = F.binary_cross_entropy(compute_edge_map(pred_probs), compute_edge_map(gt_mask))
total_loss = bce_loss + dice_loss + LAMBDA_EDGE * edge_loss
```

### Expected Impact

+1-3pp Pixel F1, with disproportionate improvement in boundary regions. IoU may improve more than F1 due to better overlap at edges.

### Risk

Edge loss can overweight boundary pixels and destabilize training if lambda_edge is too high. Start with 0.3, tune down if val loss spikes.
