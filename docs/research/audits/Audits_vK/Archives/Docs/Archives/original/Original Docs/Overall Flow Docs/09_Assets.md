# 9. Assets — Deliverables & Submission Guide

## 9.1 Assignment Requirement

> *"Submit a link to a Google Colab Notebook."*
> *"The notebook should have outputs."*

---

## 9.2 Required Deliverables

| Asset | Format | Location | Status |
|-------|--------|----------|--------|
| **Google Colab Notebook** | `.ipynb` (with outputs) | Google Drive, shared via link | **Primary deliverable** |
| **Trained Model Weights** | `.pt` file | Google Drive (linked from notebook) | Required for reproduction |
| **Saved Figures** | `.png` files | Embedded in notebook output cells | Auto-generated during execution |

---

## 9.3 Colab Notebook Setup

### Step 1: Create the Notebook
1. Go to [Google Colab](https://colab.research.google.com)
2. New Notebook
3. Runtime → Change runtime type → GPU → T4
4. Copy all code and markdown cells from the implementation

### Step 2: Run the Full Notebook
1. Runtime → Restart runtime
2. Runtime → Run all (Ctrl+F9)
3. Wait for all cells to complete (~3-4 hours for training)
4. **Do NOT clear outputs** — the evaluators need to see results

### Step 3: Verify Outputs
Before sharing, scroll through and confirm:
- [ ] All code cells show output (no empty output blocks)
- [ ] Training progress shows loss decreasing over epochs
- [ ] Metric results are printed
- [ ] All figures are rendered inline
- [ ] No error tracebacks in any cell

---

## 9.4 Model Checkpoint Management

### Save to Google Drive During Training

```python
from google.colab import drive
drive.mount('/content/drive')

SAVE_DIR = '/content/drive/MyDrive/BigVision'
os.makedirs(SAVE_DIR, exist_ok=True)

# After training completes — save best model
torch.save({
    'model_state_dict': best_model_state,
    'config': {
        'encoder_name': 'efficientnet-b1',
        'in_channels': 6,
        'classes': 1,
        'image_size': 512,
        'threshold': oracle_threshold,
    },
    'metrics': {
        'pixel_f1': results['pixel_f1_mean'],
        'pixel_iou': results['pixel_iou_mean'],
        'image_auc': results['image_auc_roc'],
    },
    'training_info': {
        'epochs_trained': best_epoch,
        'total_epochs': NUM_EPOCHS,
        'dataset': 'CASIA_v2.0',
        'seed': SEED,
    }
}, f'{SAVE_DIR}/best_model.pt')

print(f"Model saved to {SAVE_DIR}/best_model.pt")
```

### Load Model for Inference

```python
# In a fresh session or for evaluator reproduction:
checkpoint = torch.load(f'{SAVE_DIR}/best_model.pt', weights_only=False)

model = TamperingDetector(
    encoder_name=checkpoint['config']['encoder_name']
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Loaded model with Pixel-F1: {checkpoint['metrics']['pixel_f1']:.4f}")
```

---

## 9.5 Notebook Sharing

### Option A: Share via Google Drive Link (Recommended)
1. In the notebook: File → Save a copy in Drive
2. Right-click the file in Google Drive → Share
3. Change access to "Anyone with the link" → Viewer
4. Copy the link

### Option B: Share as Colab Link
1. In the notebook: Share button (top-right)
2. Change access to "Anyone with the link"
3. Copy the direct Colab link

### Link Format
```
https://colab.research.google.com/drive/NOTEBOOK_ID
```

---

## 9.6 Submission Checklist

### Before Submitting

**Notebook Content:**
- [ ] Title cell with name, date, and problem statement
- [ ] Table of contents
- [ ] All imports in first code cell
- [ ] Kaggle credentials use placeholders (not real keys)
- [ ] Every code section preceded by explanatory markdown
- [ ] Architecture rationale explained (not just code pasted)
- [ ] Loss function choice justified
- [ ] Metric definitions and interpretations included

**Notebook Outputs:**
- [ ] All cells executed with visible outputs
- [ ] Training loop shows per-epoch metrics
- [ ] Final evaluation metrics printed in formatted table
- [ ] Per-category (splicing vs. copy-move) results shown
- [ ] Training curves rendered (loss + F1 + IoU)
- [ ] ROC curve rendered
- [ ] 4-column prediction grids rendered (best/average/worst)
- [ ] Authentic image check rendered

**Technical Correctness:**
- [ ] GPU verified as T4 at start
- [ ] Seeds set for reproducibility
- [ ] AMP used throughout training
- [ ] Gradient accumulation implemented correctly (loss divided)
- [ ] Validation uses `@torch.no_grad()` 
- [ ] Best model saved based on val F1 (not train loss)
- [ ] Test set used ONLY for final evaluation (no threshold tuning on test)

**Sharing:**
- [ ] Notebook link is accessible (test in incognito browser)
- [ ] Link opens in Colab (not Drive file viewer)
- [ ] All output cells preserved (not cleared)
- [ ] Runtime type set to T4 GPU

---

## 9.7 File Size Considerations

| Component | Expected Size |
|-----------|--------------|
| Notebook (with outputs) | 5–15 MB |
| Model checkpoint | ~35 MB |
| CASIA dataset (downloaded at runtime) | ~1.5 GB |
| Generated figures | ~2 MB total |

The model checkpoint stays on Google Drive. The dataset is downloaded fresh each runtime by the Kaggle API call. Nothing needs to be bundled with the notebook file itself.

---

## 9.8 Recovery Plan

If the Colab session disconnects during training:

1. Checkpoints are saved to Google Drive every epoch
2. Re-run cells 1–18 (setup through optimizer)
3. Load last checkpoint:
   ```python
   checkpoint = torch.load(f'{SAVE_DIR}/checkpoint_epoch_N.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   # ... (load optimizer, scheduler, scaler states)
   ```
4. Continue training from `start_epoch = checkpoint['epoch'] + 1`
5. After training, re-run evaluation + visualization cells
