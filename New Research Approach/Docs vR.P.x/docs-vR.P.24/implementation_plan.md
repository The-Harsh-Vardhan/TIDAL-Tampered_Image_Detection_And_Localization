# vR.P.24 — Implementation Plan

## Core Implementation: Noiseprint Forensic Features (DnCNN Residual)

### Noiseprint Extraction Pipeline

1. Define a DnCNN residual network (17-layer blind denoiser architecture)
2. Load pretrained DnCNN weights (or train from scratch if unavailable)
3. For each image: `noiseprint = image - DnCNN(image)` (the residual is the camera/manipulation fingerprint)
4. The noiseprint is a single-channel map; replicate to 3 channels for encoder compatibility
5. Normalize noiseprint maps using training set statistics

### `DnCNN` Model Definition

```python
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17):
        # Conv + ReLU, 15x (Conv + BN + ReLU), Conv
        ...
    def forward(self, x):
        return x - self.net(x)  # residual = noiseprint
```

### `NoiseprintDataset`

Custom dataset that computes noiseprint instead of ELA:
- Load image, resize to target_size
- Pass through DnCNN to extract noiseprint residual
- Normalize with precomputed noiseprint statistics
- Return (noiseprint_tensor, mask, label)

### Cell Modification Map

| Cell | Section | Action |
|------|---------|--------|
| 0 | Title | "vR.P.24 — Noiseprint Forensic Features (DnCNN Residual)" |
| 1 | Changelog | Add P.24 entry describing noiseprint approach |
| 2 | Setup | VERSION='vR.P.24', INPUT_TYPE='Noiseprint' |
| 7 | Data prep header | Explain noiseprint theory: DnCNN residual captures camera model fingerprint and manipulation traces |
| 8 | Dataset class | DnCNN model definition + `extract_noiseprint()` function + NoiseprintDataset class |
| 9 | Splitting/stats | `compute_noiseprint_statistics()` — sample 500 images, compute per-channel mean/std of noiseprints |
| 10 | Visualization | Noiseprint vs ELA comparison: side-by-side showing both representations for authentic and tampered images |
| 11 | Architecture header | Note "Noiseprint spatial map input — DnCNN residual features" |
| 12 | Model build | Load DnCNN weights (from pretrained checkpoint or random init), build segmentation model with standard 3-channel input |
| 25 | Results table | "Noiseprint 384sq" in input column |
| 26 | Discussion | Noiseprint hypothesis, comparison to ELA, camera model fingerprint theory |
| 27 | Save model | Config includes input_type='Noiseprint', DnCNN weight source |

### Unchanged Cells

Cells 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 remain unchanged from P.3 (standard training loop, evaluation pipeline, loss function, scheduler, prediction visualization).

### Key New Code

- `DnCNN` class (~50 lines): 17-layer blind denoiser with batch normalization
- `extract_noiseprint(image, dncnn_model)`: forward pass + residual computation
- `NoiseprintDataset`: replaces ELA computation with noiseprint extraction in `__getitem__`
- `compute_noiseprint_statistics()`: analogous to `compute_ela_statistics()`

### Verification Checklist

- [ ] DnCNN model loads without errors (check weight shapes match architecture)
- [ ] Noiseprint extraction produces non-zero residual maps
- [ ] Noiseprint statistics (mean/std) are computed and applied correctly
- [ ] Visualization cell shows meaningful difference between authentic and tampered noiseprints
- [ ] Training loop runs without shape mismatches (3-channel noiseprint input)
- [ ] All metric cells execute and produce valid Pixel F1 / IoU / AUC values
- [ ] Model checkpoint saves correctly with noiseprint config metadata

### Risks

- DnCNN pretrained weights may not be available for Kaggle environment — fallback to random init (much weaker)
- Noiseprint computation adds significant overhead (~2x slower than ELA per image)
- Single-channel replicated to 3 channels wastes 2/3 of input capacity — consider 1-channel input modification
