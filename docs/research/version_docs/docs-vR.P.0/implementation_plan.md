# Implementation Plan: vR.P.0 — ResNet-34 UNet Baseline

---

## Notebook Structure

**File:** `vR.P.0 Image Detection and Localisation.ipynb`
**Cells:** 28 (18 code, 10 markdown)

| Cell | Type | Section | Content |
|------|------|---------|---------|
| 0 | Markdown | Title | Version info, pipeline diagram, rationale |
| 1 | Markdown | Change log | Version history, ETASR vs pretrained comparison |
| 2 | Code | Setup | pip install SMP, imports, config, GPU check, seeds |
| 3 | Markdown | Dataset | CASIA v2.0 description, mask strategy |
| 4 | Code | Dataset | Auto-detect paths (Au/, Tp/, GT masks) |
| 5 | Code | Dataset | Collect image paths, build GT mask mapping |
| 6 | Code | Dataset | ELA pseudo-mask generation (fallback), get_gt_mask() |
| 7 | Markdown | Data prep | Input pipeline spec, why RGB not ELA |
| 8 | Code | Data prep | PyTorch Dataset class, ImageNet transforms |
| 9 | Code | Data prep | 70/15/15 split, DataLoader creation |
| 10 | Code | Data prep | Sample visualization (RGB, ELA, mask, overlay) |
| 11 | Markdown | Architecture | UNet + ResNet-34 diagram, parameter comparison |
| 12 | Code | Architecture | Build model, freeze encoder, print summary |
| 13 | Markdown | Training | Training config table, why BCEDiceLoss |
| 14 | Code | Training | Loss function, optimizer, scheduler, train/val functions |
| 15 | Code | Training | Training loop with early stopping |
| 16 | Markdown | Evaluation | Metrics overview (pixel + classification) |
| 17 | Code | Evaluation | Test set pixel metrics (F1, IoU, Dice, AUC) |
| 18 | Code | Evaluation | Image-level classification from masks |
| 19 | Code | Evaluation | Confusion matrix + ROC curve plots |
| 20 | Code | Evaluation | Training curves (loss, F1, IoU, LR) |
| 21 | Markdown | Visualization | Section header |
| 22 | Code | Visualization | Original / GT / Predicted / Overlay grid |
| 23 | Code | Visualization | Per-image metric distribution histograms |
| 24 | Markdown | Results | Section header |
| 25 | Code | Results | Results tracking table, cross-track comparison |
| 26 | Markdown | Discussion | Findings, ETASR comparison, next steps, limitations |
| 27 | Code | Save | Save model weights + config as .pth |

---

## Key Implementation Details

### 1. Dataset Discovery

```python
def find_dataset():
    search_roots = ['/kaggle/input', '/content/drive/MyDrive']
    for base in search_roots:
        for dirpath, dirnames, _ in os.walk(base):
            if 'Au' in dirnames and 'Tp' in dirnames:
                return dirpath, os.path.join(dirpath, 'Au'), os.path.join(dirpath, 'Tp')
    return None, None, None
```

Same pattern as ETASR track — walks /kaggle/input/ looking for Au/ and Tp/ directories.

Additionally searches for ground truth mask directories using keywords: 'groundtruth', 'gt', 'mask'.

### 2. Ground Truth Mask Handling

Three-tier fallback:
1. **GT masks found:** Load from GT directory, resize to 384×384, binarize
2. **GT masks not found:** Generate ELA pseudo-mask with adaptive thresholding
3. **Authentic images:** Always return all-zero mask

### 3. ImageNet Normalization

```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Critical for pretrained encoder — without this, features would be numerically distorted.

### 4. Model Freezing

```python
for param in model.encoder.parameters():
    param.requires_grad = False
```

Verified by counting: ~21.3M frozen + ~500K trainable = ~21.8M total.

### 5. Training Loop

Custom PyTorch training loop (not model.fit()):
- Epoch loop with tqdm progress bars
- train_one_epoch() → validate() → scheduler.step() → early stopping check
- Best model state saved in memory, restored at end
- History dict tracks loss, F1, IoU, LR per epoch

### 6. Classification from Masks

Image classified as "tampered" if:
```python
tampered_pixel_count = (predicted_mask > 0.5).sum()
is_tampered = tampered_pixel_count >= 100  # minimum pixel threshold
```

Score for ROC-AUC: max probability in the predicted mask.

---

## Validation Checklist

Before submitting to Kaggle, verify:

- [ ] `!pip install -q segmentation-models-pytorch` in first code cell
- [ ] `isInternetEnabled: True` in notebook metadata (needed for pip install)
- [ ] GPU enabled in Kaggle settings
- [ ] CASIA v2.0 dataset added as Kaggle input
- [ ] (Optional) GT mask dataset added as second Kaggle input
- [ ] All 28 cells present and in order
- [ ] No hardcoded paths (auto-detect pattern used)
- [ ] Seed = 42 set for all random sources
- [ ] Model save filename: `vR.P.0_unet_resnet34_model.pth`

---

## Dependencies

```
torch >= 1.10          (pre-installed on Kaggle)
torchvision            (pre-installed on Kaggle)
segmentation-models-pytorch >= 0.3.0  (pip install in notebook)
numpy                  (pre-installed)
matplotlib             (pre-installed)
seaborn                (pre-installed)
Pillow                 (pre-installed)
scikit-learn           (pre-installed)
tqdm                   (pre-installed)
```

---

## Runtime Estimate

| Stage | Time (T4 GPU) |
|-------|---------------|
| SMP pip install | ~30 sec |
| Dataset discovery + path collection | ~5 sec |
| Image loading (first batch, lazy) | immediate |
| Model build + freeze | ~2 sec |
| Training (25 epochs × ~45 sec/epoch) | ~15-20 min |
| Test evaluation | ~2 min |
| Visualizations | ~1 min |
| **Total** | **~20-25 min** |
