#!/usr/bin/env python3
"""Build vK.10.6 notebook from vK.10.5 by adding evaluation-only cells."""
import json
import sys

NB = 'Notebooks/vK.10.6 Image Detection and Localisation.ipynb'

with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Remove stray cell at index 0 if it's the shortcut learning artifact
if cells[0].get('id') == '3lvpzkyl9sw':
    cells.pop(0)
    print("Removed stray cell 0")

print(f"Base: {len(cells)} cells")

# ---- HELPERS ----
def mk(ctype, source, cid):
    lines = source.split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    c = {"cell_type": ctype, "metadata": {}, "source": src, "id": cid}
    if ctype == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

# ================================================================
# VERSION UPDATES
# ================================================================

# Cell 0: TOC
cells[0] = mk("markdown", """# Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Configuration](#2-configuration)
3. [Reproducibility and Device Setup](#3-reproducibility-and-device-setup)
4. [Dataset Discovery and Metadata Cache](#4-dataset-discovery-and-metadata-cache)
5. [Dependencies and Imports](#5-dependencies-and-imports)
6. [Data Loading and Preprocessing](#6-data-loading-and-preprocessing)
7. [Data Visualization](#7-data-visualization)
8. [Model Architecture](#8-model-architecture)
9. [Experiment Tracking](#9-experiment-tracking)
10. [Training Utilities](#10-training-utilities)
11. [Training Loop](#11-training-loop)
12. [Evaluation](#12-evaluation)
13. [Visualization of Predictions](#13-visualization-of-predictions)
14. [Inference Examples](#14-inference-examples)
15. [Robustness Testing](#15-robustness-testing)
16. [Conclusion](#16-conclusion)""", 'toc')

# Cell 1: Title
cells[1] = mk("markdown", """# Tampered Image Detection and Localization (vK.10.6)

This Kaggle-first notebook presents a complete assignment submission for tampered image
detection and tampered region localization.

**Engineering upgrades (carried from vK.10.5)**
- Multi-GPU training with `nn.DataParallel` (utilizes both Kaggle T4 GPUs)
- Centralized CONFIG dictionary for all hyperparameters
- Full reproducibility seeding (Python, NumPy, PyTorch CPU/CUDA)
- Automatic Mixed Precision (AMP) for faster training
- Three-file checkpoint system with automatic resume
- Early stopping based on tampered-only Dice coefficient
- Tampered-only metric reporting to address metric inflation
- GPU diagnostics and VRAM-based batch size auto-adjustment
- Metadata caching to skip redundant dataset scanning
- Optimized DataLoaders (persistent workers, seeded workers, drop_last)

**Evaluation upgrades in vK.10.6**
- Data leakage verification (pre-training path overlap check)
- Segmentation threshold optimization via validation sweep
- Pixel-level AUC-ROC metric
- Confusion matrix + ROC curve + Precision-Recall curve
- Per-forgery-type evaluation (splicing vs copy-move)
- Mask-size stratified evaluation (tiny/small/medium/large)
- Shortcut learning validation (mask randomization + boundary sensitivity)
- Failure case analysis (worst predictions with metadata)
- Grad-CAM explainability visualization
- Robustness testing suite (8 degradation conditions)

**Notebook deliverables**
- Image-level tamper detection through the classifier head
- Pixel-level tampered region localization through the segmentation branch
- Reproducible Kaggle-first execution with Colab/Drive fallback
- Qualitative visual evidence showing predicted masks and overlays""", 'b31f02b9')

# Cell 2: Objectives
cells[2] = mk("markdown", """## Project Objectives: Fulfilled vs Remaining

| Requirement | Status | Evidence |
|---|---|---|
| Dataset: authentic + tampered + masks | Fulfilled | CASIA dataset with IMAGE/MASK dirs |
| Model performs detection + localization | Fulfilled | UNetWithClassifier dual-head |
| Evaluation with Dice / IoU / F1 | Fulfilled | Tampered-only and all-sample metrics |
| Visual results (Original, GT, Pred, Overlay) | Fulfilled | Submission prediction grid |
| Single notebook | Fulfilled | All code in one notebook |
| Reproducibility | Fulfilled | Full seeding + checkpoint resume |
| AMP training | Fulfilled | autocast + GradScaler |
| Early stopping | Fulfilled | Patience-based on tampered Dice |
| Multi-GPU utilization | Fulfilled | nn.DataParallel across T4 x2 |
| **Data leakage verification** | **Fulfilled (vK.10.6)** | **Path overlap assertion** |
| **Threshold optimization** | **Fulfilled (vK.10.6)** | **Validation sweep 0.05-0.80** |
| **Confusion matrix + ROC/PR curves** | **Fulfilled (vK.10.6)** | **sklearn + seaborn** |
| **Per-forgery-type evaluation** | **Fulfilled (vK.10.6)** | **Splicing vs copy-move** |
| **Mask-size stratified evaluation** | **Fulfilled (vK.10.6)** | **4 size buckets** |
| **Shortcut learning checks** | **Fulfilled (vK.10.6)** | **Mask randomization + boundary** |
| **Failure case analysis** | **Fulfilled (vK.10.6)** | **10 worst predictions** |
| **Grad-CAM explainability** | **Fulfilled (vK.10.6)** | **Encoder bottleneck heatmaps** |
| **Robustness testing** | **Fulfilled (vK.10.6)** | **8 degradation conditions** |
| **Pixel-level AUC-ROC** | **Fulfilled (vK.10.6)** | **Threshold-independent metric** |""", 'b39fbe60')

# Cell 38 (W&B): Update version
src38 = ''.join(cells[38]['source'])
src38 = src38.replace('vK.10.5-tampered', 'vK.10.6-tampered')
src38 = src38.replace('vK.10.5-unet', 'vK.10.6-unet')
src38 = src38.replace('"vk10.5"', '"vk10.6"')
src38 = src38.replace('vk10.5-offline', 'vk10.6-offline')
lines38 = src38.split('\n')
cells[38]['source'] = [l + '\n' for l in lines38[:-1]] + [lines38[-1]]

# Cell 68 (Conclusion)
cells[68] = mk("markdown", """## 16. Conclusion

This notebook (vK.10.6) presents a complete, evaluation-enhanced pipeline for tampered
image detection and localization.

**Evaluation upgrades in vK.10.6:**
- Data leakage verification, threshold optimization, pixel-level AUC-ROC
- Confusion matrix + ROC/PR curves, forgery-type breakdown, mask-size stratification
- Shortcut learning validation, failure case analysis, Grad-CAM explainability
- Robustness testing suite (8 degradation conditions)

**Engineering carried from vK.10.5:**
- Multi-GPU DataParallel, AMP, checkpoint system, early stopping, CONFIG dict
- Full reproducibility, VRAM-based batch sizing, metadata caching

**Assignment coverage:**
- Detection + localization, Dice/IoU/F1/AUC-ROC, visual results, Grad-CAM, robustness
- Scientific rigor: shortcut checks, failure analysis, threshold optimization""", 'd4cd2f5d')

print("Version updates done")

# ================================================================
# INSERT: Data Leakage after cell 22
# ================================================================
leakage_cells = [
    mk("markdown", "### 4.5 Data Leakage Verification\n\nExplicit check that no image path appears in more than one split.", 'v106-leak-md'),
    mk("code", "# ================== Data Leakage Verification ==================\ntrain_paths = set(train_df['image_path'].values)\nval_paths = set(val_df['image_path'].values)\ntest_paths = set(test_df['image_path'].values)\n\ntrain_val = train_paths & val_paths\ntrain_test = train_paths & test_paths\nval_test = val_paths & test_paths\ntotal = len(train_val) + len(train_test) + len(val_test)\n\nprint(f\"Train-Val overlap:  {len(train_val)}\")\nprint(f\"Train-Test overlap: {len(train_test)}\")\nprint(f\"Val-Test overlap:   {len(val_test)}\")\nprint(f\"{'=' * 40}\")\nprint(f\"{'PASS' if total == 0 else 'FAIL'}: {total} overlapping paths.\")\nassert total == 0, f\"Data leakage: {total} overlapping paths.\"\nprint(f\"\\nTrain={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}\")", 'v106-leak-code'),
]
for i, c in enumerate(leakage_cells):
    cells.insert(23 + i, c)
# +2 shift

print(f"After leakage: {len(cells)}")

# ================================================================
# INSERT: Eval cells after training curves (old 55, now 57)
# ================================================================

# Read eval cell sources from separate files would be cleaner,
# but let's inline them here.

eval_cells = []

# --- Threshold Sweep ---
eval_cells.append(mk("markdown", "### 12.3 Segmentation Threshold Optimization\n\nSweep thresholds 0.05-0.80 on validation, select optimal.", 'v106-thresh-md'))
eval_cells.append(mk("code",
"# ================== Threshold Sweep ==================\n"
"import matplotlib.pyplot as plt\n"
"\n"
"@torch.no_grad()\n"
"def sweep_thresholds(loader, thresholds):\n"
"    model.eval()\n"
"    all_sl, all_m, all_l = [], [], []\n"
"    for imgs, masks, labels in tqdm(loader, desc='Sweep', leave=False):\n"
"        imgs = imgs.to(device)\n"
"        with autocast('cuda', enabled=CONFIG['use_amp']):\n"
"            _, sl = model(imgs)\n"
"        all_sl.append(sl.cpu()); all_m.append(masks.cpu()); all_l.append(labels.cpu())\n"
"    all_sl=torch.cat(all_sl); all_m=torch.cat(all_m); all_l=torch.cat(all_l)\n"
"    all_p = torch.sigmoid(all_sl)\n"
"    results = []\n"
"    eps = 1e-7\n"
"    for t in thresholds:\n"
"        f1s = []\n"
"        for i in range(all_sl.size(0)):\n"
"            if all_l[i].item() == 1:\n"
"                pb = (all_p[i:i+1] > t).float(); m = all_m[i:i+1]\n"
"                tp=(pb*m).sum(); fp=(pb*(1-m)).sum(); fn=((1-pb)*m).sum()\n"
"                p=(tp+eps)/(tp+fp+eps); r=(tp+eps)/(tp+fn+eps)\n"
"                f1s.append((2*p*r/(p+r+eps)).item())\n"
"        results.append(np.mean(f1s) if f1s else 0.0)\n"
"    return results\n"
"\n"
"thresholds = np.arange(0.05, 0.85, 0.05)\n"
"val_f1s = sweep_thresholds(val_loader, thresholds)\n"
"oi = np.argmax(val_f1s)\n"
"optimal_threshold = thresholds[oi]\n"
"di = int(np.argmin(np.abs(thresholds - 0.50)))\n"
"print(f'Optimal: {optimal_threshold:.2f} (F1={val_f1s[oi]:.4f})')\n"
"print(f'Default: 0.50 (F1={val_f1s[di]:.4f})')\n"
"\n"
"fig, ax = plt.subplots(1, 1, figsize=(8, 4))\n"
"ax.plot(thresholds, val_f1s, 'b-o', ms=4)\n"
"ax.axvline(x=optimal_threshold, color='r', ls='--', label=f'Optimal: {optimal_threshold:.2f}')\n"
"ax.axvline(x=0.5, color='gray', ls=':', label='Default: 0.50')\n"
"ax.set_xlabel('Threshold'); ax.set_ylabel('Tampered-Only F1')\n"
"ax.set_title('Threshold vs F1 (Validation)'); ax.legend(); ax.grid(True, alpha=0.3)\n"
"plt.tight_layout(); plt.show()\n"
"\n"
"OPTIMAL_THRESHOLD = float(optimal_threshold)\n"
"print(f'\\nUsing {OPTIMAL_THRESHOLD:.2f} for subsequent evaluations.')",
'v106-thresh-code'))

# --- Re-evaluate test ---
eval_cells.append(mk("markdown", "### 12.4 Test Re-Evaluation with Optimal Threshold", 'v106-reeval-md'))
eval_cells.append(mk("code",
"# ================== Re-evaluate test ==================\n"
"@torch.no_grad()\n"
"def evaluate_with_threshold(loader, threshold, name='Test'):\n"
"    model.eval()\n"
"    correct, total = 0, 0\n"
"    all_c, all_sp, all_m, all_l = [], [], [], []\n"
"    for imgs, masks, labels in tqdm(loader, desc=f'{name} t={threshold:.2f}', leave=False):\n"
"        imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)\n"
"        with autocast('cuda', enabled=CONFIG['use_amp']):\n"
"            cl, sl = model(imgs)\n"
"        correct += (torch.argmax(cl,1)==labels).sum().item()\n"
"        total += labels.size(0)\n"
"        all_c.append(cl.cpu()); all_sp.append(torch.sigmoid(sl).cpu())\n"
"        all_m.append(masks.cpu()); all_l.append(labels.cpu())\n"
"    all_c=torch.cat(all_c); all_sp=torch.cat(all_sp); all_m=torch.cat(all_m); all_l=torch.cat(all_l)\n"
"    eps=1e-7; ad,ai,af,td,ti,tf=[],[],[],[],[],[]\n"
"    for i in range(all_sp.size(0)):\n"
"        pb=(all_sp[i:i+1]>threshold).float(); m=all_m[i:i+1]\n"
"        inter=(pb*m).sum(); ud=pb.sum()+m.sum(); ui=ud-inter\n"
"        tp=inter; fp=(pb*(1-m)).sum(); fn=((1-pb)*m).sum()\n"
"        d=(2*inter+eps)/(ud+eps); io=(inter+eps)/(ui+eps)\n"
"        p=(tp+eps)/(tp+fp+eps); r=(tp+eps)/(tp+fn+eps); f=2*p*r/(p+r+eps)\n"
"        ad.append(d.item()); ai.append(io.item()); af.append(f.item())\n"
"        if all_l[i].item()==1: td.append(d.item()); ti.append(io.item()); tf.append(f.item())\n"
"    cp=torch.softmax(all_c,1)[:,1].numpy(); ln=all_l.numpy()\n"
"    try: auc=roc_auc_score(ln,cp)\n"
"    except: auc=0.0\n"
"    return {'acc':correct/total,'roc_auc':auc,'dice':np.mean(ad),'iou':np.mean(ai),'f1':np.mean(af),\n"
"            'tampered_dice':np.mean(td) if td else 0,'tampered_iou':np.mean(ti) if ti else 0,\n"
"            'tampered_f1':np.mean(tf) if tf else 0,\n"
"            'cls_probs':cp,'labels_np':ln,'all_seg_probs':all_sp,'all_masks':all_m,'all_labels':all_l}\n"
"\n"
"test_opt = evaluate_with_threshold(test_loader, OPTIMAL_THRESHOLD)\n"
"test_def = evaluate_with_threshold(test_loader, 0.5)\n"
"print(f\"\\n{'Metric':<20} {'Default':>10} {'Optimal':>10} {'Delta':>10}\")\n"
"print('='*52)\n"
"for k in ['acc','roc_auc','dice','iou','f1','tampered_dice','tampered_iou','tampered_f1']:\n"
"    print(f\"{k:<20} {test_def[k]:>10.4f} {test_opt[k]:>10.4f} {test_opt[k]-test_def[k]:>+10.4f}\")",
'v106-reeval-code'))

# --- Pixel AUC ---
eval_cells.append(mk("markdown", "### 12.5 Pixel-Level AUC-ROC\n\nThreshold-independent localization quality.", 'v106-pixauc-md'))
eval_cells.append(mk("code",
"# ================== Pixel-Level AUC-ROC ==================\n"
"from sklearn.metrics import roc_auc_score as sklearn_roc_auc\n"
"tam_idx=[i for i in range(test_opt['all_labels'].size(0)) if test_opt['all_labels'][i].item()==1]\n"
"if tam_idx:\n"
"    tp=test_opt['all_seg_probs'][tam_idx].numpy().flatten()\n"
"    tm=test_opt['all_masks'][tam_idx].numpy().flatten()\n"
"    try: pixel_auc_tam=sklearn_roc_auc(tm,tp)\n"
"    except: pixel_auc_tam=0.0\n"
"else: pixel_auc_tam=0.0\n"
"sp=test_opt['all_seg_probs'].numpy().flatten()\n"
"mf=test_opt['all_masks'].numpy().flatten()\n"
"try: pixel_auc_all=sklearn_roc_auc(mf,sp)\n"
"except: pixel_auc_all=0.0\n"
"print(f'Pixel AUC (all):      {pixel_auc_all:.4f}')\n"
"print(f'Pixel AUC (tampered): {pixel_auc_tam:.4f}')\n"
"print(f'Image AUC (cls):      {test_opt[\"roc_auc\"]:.4f}')",
'v106-pixauc-code'))

# --- Confusion Matrix + ROC/PR ---
eval_cells.append(mk("markdown", "### 12.6 Classification Analysis\n\nConfusion matrix, ROC, and PR curves.", 'v106-cm-md'))
eval_cells.append(mk("code",
"# ================== Confusion Matrix + ROC + PR ==================\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score\n"
"\n"
"cp=test_opt['cls_probs']; ln=test_opt['labels_np']\n"
"cm=confusion_matrix(ln,(cp>=0.5).astype(int))\n"
"fpr,tpr,_=roc_curve(ln,cp); ra=auc(fpr,tpr)\n"
"pa,ra2,_=precision_recall_curve(ln,cp); ap=average_precision_score(ln,cp)\n"
"\n"
"fig,axes=plt.subplots(1,3,figsize=(16,4.5))\n"
"sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=axes[0],\n"
"            xticklabels=['Auth','Tamp'],yticklabels=['Auth','Tamp'])\n"
"axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual'); axes[0].set_title('Confusion Matrix')\n"
"axes[1].plot(fpr,tpr,'b-',lw=2,label=f'AUC={ra:.3f}')\n"
"axes[1].plot([0,1],[0,1],'k--',alpha=0.5)\n"
"axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR'); axes[1].set_title('ROC')\n"
"axes[1].legend(); axes[1].grid(True,alpha=0.3)\n"
"axes[2].plot(ra2,pa,'g-',lw=2,label=f'AP={ap:.3f}')\n"
"axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision'); axes[2].set_title('PR')\n"
"axes[2].legend(); axes[2].grid(True,alpha=0.3)\n"
"plt.tight_layout(); plt.show()\n"
"print(f'Acc={test_opt[\"acc\"]:.4f} AUC={ra:.4f} AP={ap:.4f} TP={cm[1,1]} FP={cm[0,1]} TN={cm[0,0]} FN={cm[1,0]}')",
'v106-cm-code'))

# --- Forgery-Type ---
eval_cells.append(mk("markdown", "### 12.7 Per-Forgery-Type Evaluation\n\nSplicing vs copy-move breakdown.", 'v106-ftype-md'))
eval_cells.append(mk("code",
"# ================== Forgery-Type Breakdown ==================\n"
"import os\n"
"\n"
"@torch.no_grad()\n"
"def collect_per_sample(loader, threshold):\n"
"    model.eval(); results=[]; idx=0\n"
"    for imgs,masks,labels in tqdm(loader,desc='Per-sample',leave=False):\n"
"        imgs_d=imgs.to(device)\n"
"        with autocast('cuda',enabled=CONFIG['use_amp']):\n"
"            cl,sl=model(imgs_d)\n"
"        sp=torch.sigmoid(sl).cpu(); cp=torch.argmax(cl,1).cpu()\n"
"        for i in range(imgs.size(0)):\n"
"            pb=(sp[i:i+1]>threshold).float(); m=masks[i:i+1]; eps=1e-7\n"
"            inter=(pb*m).sum(); tp=inter; fp=(pb*(1-m)).sum(); fn=((1-pb)*m).sum()\n"
"            d=(2*inter+eps)/(pb.sum()+m.sum()+eps)\n"
"            io=(inter+eps)/(pb.sum()+m.sum()-inter+eps)\n"
"            p=(tp+eps)/(tp+fp+eps); r=(tp+eps)/(tp+fn+eps); f1=2*p*r/(p+r+eps)\n"
"            mp=m.sum().item()/m.numel()*100\n"
"            fp_=test_dataset.df.iloc[idx]['image_path'] if idx<len(test_dataset) else '?'\n"
"            fn_=os.path.basename(fp_)\n"
"            ft='splicing' if fn_.startswith('Tp_S') else 'copy-move' if fn_.startswith('Tp_D') else 'authentic'\n"
"            results.append({'idx':idx,'filename':fn_,'label':labels[i].item(),'cls_pred':cp[i].item(),\n"
"                           'dice':d.item(),'iou':io.item(),'f1':f1.item(),'mask_pct':mp,\n"
"                           'forgery_type':ft,'image':imgs[i],'mask':masks[i],'pred_prob':sp[i]})\n"
"            idx+=1\n"
"    return results\n"
"\n"
"per_sample = collect_per_sample(test_loader, OPTIMAL_THRESHOLD)\n"
"\n"
"print(f\"{'Type':<15} {'N':>5} {'Dice':>8} {'IoU':>8} {'F1':>8}\")\n"
"print('='*48)\n"
"for ft in ['splicing','copy-move','authentic']:\n"
"    s=[r for r in per_sample if r['forgery_type']==ft]\n"
"    if s: print(f\"{ft:<15} {len(s):>5} {np.mean([r['dice'] for r in s]):>8.4f} \"\n"
"              f\"{np.mean([r['iou'] for r in s]):>8.4f} {np.mean([r['f1'] for r in s]):>8.4f}\")\n"
"    else: print(f\"{ft:<15} {0:>5} {'N/A':>8} {'N/A':>8} {'N/A':>8}\")\n"
"print(f'\\nClassification accuracy by type:')\n"
"for ft in ['splicing','copy-move','authentic']:\n"
"    s=[r for r in per_sample if r['forgery_type']==ft]\n"
"    if s:\n"
"        c=sum(1 for r in s if r['cls_pred']==r['label'])\n"
"        print(f'  {ft}: {c}/{len(s)} = {c/len(s):.4f}')",
'v106-ftype-code'))

# --- Mask-Size ---
eval_cells.append(mk("markdown", "### 12.8 Mask-Size Stratified Evaluation\n\nMetrics by tampered region size.", 'v106-msize-md'))
eval_cells.append(mk("code",
"# ================== Mask-Size Stratification ==================\n"
"import matplotlib.pyplot as plt\n"
"ts=[r for r in per_sample if r['label']==1]\n"
"bk={'tiny (<2%)':[],  'small (2-5%)':[],  'medium (5-15%)':[],  'large (>15%)':[]}\n"
"for r in ts:\n"
"    p=r['mask_pct']\n"
"    if p<2: bk['tiny (<2%)'].append(r)\n"
"    elif p<5: bk['small (2-5%)'].append(r)\n"
"    elif p<15: bk['medium (5-15%)'].append(r)\n"
"    else: bk['large (>15%)'].append(r)\n"
"print(f\"{'Bucket':<18} {'N':>5} {'Dice':>8} {'IoU':>8} {'F1':>8}\")\n"
"print('='*52)\n"
"bn,bf,bc=[],[],[]\n"
"for k,v in bk.items():\n"
"    f1=np.mean([r['f1'] for r in v]) if v else 0\n"
"    if v: print(f\"{k:<18} {len(v):>5} {np.mean([r['dice'] for r in v]):>8.4f} \"\n"
"                f\"{np.mean([r['iou'] for r in v]):>8.4f} {f1:>8.4f}\")\n"
"    else: print(f\"{k:<18} {0:>5} {'N/A':>8} {'N/A':>8} {'N/A':>8}\")\n"
"    bn.append(k); bf.append(f1); bc.append(len(v))\n"
"fig,ax=plt.subplots(figsize=(8,4))\n"
"bars=ax.bar(bn,bf,color=['#ff6b6b','#ffa07a','#90ee90','#4ecdc4'])\n"
"for b,c in zip(bars,bc): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f'n={c}',ha='center',fontsize=9)\n"
"ax.set_ylabel('F1'); ax.set_title('F1 by Region Size')\n"
"ax.set_ylim(0,max(bf+[0.1])*1.15); ax.grid(True,alpha=0.3,axis='y')\n"
"plt.tight_layout(); plt.show()",
'v106-msize-code'))

# --- Shortcut Learning ---
eval_cells.append(mk("markdown", "### 12.9 Shortcut Learning Checks\n\n1. Mask randomization 2. Boundary sensitivity.", 'v106-short-md'))
eval_cells.append(mk("code",
"# ================== Shortcut Learning ==================\n"
"import torch.nn.functional as F\n"
"tonly=[r for r in per_sample if r['label']==1]; eps=1e-7\n"
"# Test 1: Mask Randomization\n"
"if len(tonly)>1:\n"
"    rf1=np.mean([r['f1'] for r in tonly])\n"
"    np.random.seed(42); si=np.random.permutation(len(tonly))\n"
"    sf1=[]\n"
"    for i,r in enumerate(tonly):\n"
"        j=si[i]; pb=(r['pred_prob'].unsqueeze(0)>OPTIMAL_THRESHOLD).float()\n"
"        wm=tonly[j]['mask'].unsqueeze(0)\n"
"        tp=(pb*wm).sum(); fp=(pb*(1-wm)).sum(); fn=((1-pb)*wm).sum()\n"
"        p=(tp+eps)/(tp+fp+eps); rc=(tp+eps)/(tp+fn+eps)\n"
"        sf1.append((2*p*rc/(p+rc+eps)).item())\n"
"    sm=np.mean(sf1); drop=rf1-sm\n"
"    print('Test 1: Mask Randomization')\n"
"    print(f'  Real F1:     {rf1:.4f}')\n"
"    print(f'  Shuffled F1: {sm:.4f}')\n"
"    print(f'  Drop:        {drop:+.4f}')\n"
"    print(f'  {\"PASS\" if drop>0.05 else \"WARNING\"}')\n"
"# Test 2: Boundary Sensitivity\n"
"print(f'\\nTest 2: Boundary Sensitivity')\n"
"if tonly:\n"
"    ed,od=[],[]\n"
"    for r in tonly:\n"
"        pb=(r['pred_prob'].unsqueeze(0)>OPTIMAL_THRESHOLD).float(); m=r['mask'].unsqueeze(0)\n"
"        eroded=1.0-F.max_pool2d(1.0-pb,kernel_size=3,stride=1,padding=1)\n"
"        ie=(eroded*m).sum(); ue=eroded.sum()+m.sum()\n"
"        ed.append((2*ie+eps)/(ue+eps)); od.append(r['dice'])\n"
"    mo=np.mean(od); me=np.mean([d.item() if hasattr(d,'item') else d for d in ed])\n"
"    delta=mo-me\n"
"    print(f'  Original Dice: {mo:.4f}')\n"
"    print(f'  Eroded Dice:   {me:.4f}')\n"
"    print(f'  Drop:          {delta:+.4f}')\n"
"    print(f'  {\"PASS\" if delta<0.05 else \"NOTE\"}')",
'v106-short-code'))

# Insert all eval cells after training curves code (old index 55, now 57)
for i, c in enumerate(eval_cells):
    cells.insert(58 + i, c)

print(f"After eval: {len(cells)} cells, {len(eval_cells)} eval cells added")

# ================================================================
# INSERT: Failure + Grad-CAM before inference section
# ================================================================
# Find inference markdown (shifted by +2 leakage + eval_cells)
inf_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown':
        src = ''.join(c['source'])
        if '## 13. Inference' in src or 'Inference Examples' in src:
            inf_idx = i
            break

viz_cells = []

# Failure Case Analysis
viz_cells.append(mk("markdown", "### 13.3 Failure Case Analysis\n\n10 worst predictions with metadata.", 'v106-fail-md'))
viz_cells.append(mk("code",
"# ================== Failure Case Analysis ==================\n"
"import matplotlib.pyplot as plt\n"
"tsorted=sorted([r for r in per_sample if r['label']==1],key=lambda r:r['f1'])\n"
"worst=tsorted[:10]\n"
"print(f\"{'#':<3} {'Filename':<30} {'F1':>7} {'Dice':>7} {'Mask%':>7} {'Type':<10}\")\n"
"print('='*70)\n"
"for i,r in enumerate(worst):\n"
"    print(f\"{i+1:<3} {r['filename']:<30} {r['f1']:>7.4f} {r['dice']:>7.4f} {r['mask_pct']:>6.2f}% {r['forgery_type']:<10}\")\n"
"n=min(10,len(worst))\n"
"if n>0:\n"
"    mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])\n"
"    fig,axes=plt.subplots(n,4,figsize=(16,3.5*n))\n"
"    if n==1: axes=axes.reshape(1,-1)\n"
"    for i,r in enumerate(worst[:n]):\n"
"        img=(r['image'].permute(1,2,0).numpy()*std+mean).clip(0,1)\n"
"        gt=r['mask'].squeeze().numpy()\n"
"        pred=(r['pred_prob'].squeeze().numpy()>OPTIMAL_THRESHOLD).astype(float)\n"
"        ov=img.copy(); ov[pred>0.5]=ov[pred>0.5]*0.5+np.array([1,0,0])*0.5\n"
"        axes[i,0].imshow(img); axes[i,0].set_title('Original',fontsize=9); axes[i,0].axis('off')\n"
"        axes[i,1].imshow(gt,cmap='gray'); axes[i,1].set_title(f'GT ({r[\"mask_pct\"]:.1f}%)',fontsize=9); axes[i,1].axis('off')\n"
"        axes[i,2].imshow(pred,cmap='gray'); axes[i,2].set_title(f'Pred (F1={r[\"f1\"]:.3f})',fontsize=9); axes[i,2].axis('off')\n"
"        axes[i,3].imshow(ov); axes[i,3].set_title(f'{r[\"forgery_type\"]}',fontsize=8); axes[i,3].axis('off')\n"
"    plt.suptitle('Failure Analysis: 10 Worst', fontsize=14, y=1.01)\n"
"    plt.tight_layout(); plt.show()",
'v106-fail-code'))

# Grad-CAM
viz_cells.append(mk("markdown", "### 13.4 Grad-CAM Explainability\n\nEncoder bottleneck attention maps.", 'v106-gcam-md'))
viz_cells.append(mk("code",
"# ================== Grad-CAM ==================\n"
"import matplotlib.pyplot as plt\n"
"import torch.nn.functional as F\n"
"\n"
"class GradCAM:\n"
"    def __init__(self, model, layer):\n"
"        self.model=model; self.grads=None; self.acts=None\n"
"        layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))\n"
"        layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))\n"
"    def generate(self, x, cls=1):\n"
"        self.model.eval()\n"
"        cl,_=self.model(x)\n"
"        self.model.zero_grad()\n"
"        cl[:,cls].backward(retain_graph=True)\n"
"        if self.grads is None or self.acts is None:\n"
"            return np.zeros((x.shape[2],x.shape[3]))\n"
"        w=self.grads.mean(dim=(2,3),keepdim=True)\n"
"        cam=F.relu((w*self.acts).sum(1,keepdim=True))\n"
"        cam=F.interpolate(cam,size=(x.shape[2],x.shape[3]),mode='bilinear',align_corners=False).squeeze()\n"
"        if cam.max()>0: cam=(cam-cam.min())/(cam.max()-cam.min())\n"
"        return cam.cpu().numpy()\n"
"\n"
"bm=get_base_model(model)\n"
"gcam=GradCAM(bm,bm.down4.conv)\n"
"auth_s=[r for r in per_sample if r['label']==0][:3]\n"
"tam_s=[r for r in per_sample if r['label']==1][:3]\n"
"gc_s=auth_s+tam_s\n"
"mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])\n"
"n=len(gc_s)\n"
"if n>0:\n"
"    fig,axes=plt.subplots(n,3,figsize=(12,3.5*n))\n"
"    if n==1: axes=axes.reshape(1,-1)\n"
"    for i,r in enumerate(gc_s):\n"
"        inp=r['image'].unsqueeze(0).to(device).requires_grad_(True)\n"
"        cam=gcam.generate(inp,cls=1)\n"
"        img=(r['image'].permute(1,2,0).numpy()*std+mean).clip(0,1)\n"
"        pred=(r['pred_prob'].squeeze().numpy()>OPTIMAL_THRESHOLD).astype(float)\n"
"        lbl='Tampered' if r['label']==1 else 'Authentic'\n"
"        axes[i,0].imshow(img); axes[i,0].set_title(f'{lbl}',fontsize=9); axes[i,0].axis('off')\n"
"        axes[i,1].imshow(img); axes[i,1].imshow(cam,cmap='jet',alpha=0.4)\n"
"        axes[i,1].set_title('Grad-CAM',fontsize=9); axes[i,1].axis('off')\n"
"        axes[i,2].imshow(pred,cmap='gray'); axes[i,2].set_title(f'Mask (F1={r[\"f1\"]:.3f})',fontsize=9); axes[i,2].axis('off')\n"
"    plt.suptitle('Grad-CAM: Encoder Attention',fontsize=14,y=1.01)\n"
"    plt.tight_layout(); plt.show()\n"
"model.eval()",
'v106-gcam-code'))

if inf_idx:
    print(f"Inference at {inf_idx}, inserting {len(viz_cells)} viz cells before it")
    for i, c in enumerate(viz_cells):
        cells.insert(inf_idx + i, c)

print(f"After viz: {len(cells)} cells")

# ================================================================
# INSERT: Robustness before conclusion
# ================================================================
conc_idx = None
for i, c in enumerate(cells):
    if c.get('id') == 'd4cd2f5d':
        conc_idx = i
        break

robust_cells = [
    mk("markdown", "## 15. Robustness Testing\n\n8 degradation conditions at inference time (**Bonus B1**).", 'v106-rob-md'),
    mk("code",
"# ================== Robustness Testing ==================\n"
"import cv2\n"
"import matplotlib.pyplot as plt\n"
"\n"
"def apply_deg(img,cond):\n"
"    if cond=='jpeg_70': _,b=cv2.imencode('.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),70]); return cv2.imdecode(b,1)\n"
"    elif cond=='jpeg_50': _,b=cv2.imencode('.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),50]); return cv2.imdecode(b,1)\n"
"    elif cond=='noise_10': return np.clip(img.astype(np.float32)+np.random.normal(0,10,img.shape),0,255).astype(np.uint8)\n"
"    elif cond=='noise_25': return np.clip(img.astype(np.float32)+np.random.normal(0,25,img.shape),0,255).astype(np.uint8)\n"
"    elif cond=='blur_3': return cv2.GaussianBlur(img,(3,3),0)\n"
"    elif cond=='blur_5': return cv2.GaussianBlur(img,(5,5),0)\n"
"    elif cond=='resize_75': h,w=img.shape[:2]; return cv2.resize(cv2.resize(img,(int(w*.75),int(h*.75))),(w,h))\n"
"    elif cond=='resize_50': h,w=img.shape[:2]; return cv2.resize(cv2.resize(img,(int(w*.5),int(h*.5))),(w,h))\n"
"    return img\n"
"\n"
"conds=[('clean','Clean'),('jpeg_70','JPEG QF=70'),('jpeg_50','JPEG QF=50'),\n"
"       ('noise_10','Noise s=10'),('noise_25','Noise s=25'),\n"
"       ('blur_3','Blur k=3'),('blur_5','Blur k=5'),\n"
"       ('resize_75','Resize 0.75x'),('resize_50','Resize 0.5x')]\n"
"\n"
"mt=torch.tensor([0.485,0.456,0.406]).view(3,1,1)\n"
"st=torch.tensor([0.229,0.224,0.225]).view(3,1,1)\n"
"\n"
"@torch.no_grad()\n"
"def eval_robust(loader,ck,thr):\n"
"    model.eval(); f1s=[]; eps=1e-7\n"
"    for imgs,masks,labels in loader:\n"
"        for i in range(imgs.size(0)):\n"
"            if labels[i].item()!=1: continue\n"
"            if ck=='clean': t=imgs[i:i+1].to(device)\n"
"            else:\n"
"                im=(imgs[i]*st+mt).permute(1,2,0).numpy()\n"
"                im8=(im*255).clip(0,255).astype(np.uint8)\n"
"                bgr=cv2.cvtColor(im8,cv2.COLOR_RGB2BGR)\n"
"                deg=apply_deg(bgr,ck)\n"
"                rgb=cv2.cvtColor(deg,cv2.COLOR_BGR2RGB)\n"
"                t=(torch.from_numpy(rgb).float().permute(2,0,1)/255.0-mt)/st\n"
"                t=t.unsqueeze(0).to(device)\n"
"            with autocast('cuda',enabled=CONFIG['use_amp']):\n"
"                _,sl=model(t)\n"
"            pb=(torch.sigmoid(sl).cpu()>thr).float(); m=masks[i:i+1]\n"
"            tp=(pb*m).sum(); fp=(pb*(1-m)).sum(); fn=((1-pb)*m).sum()\n"
"            p=(tp+eps)/(tp+fp+eps); r=(tp+eps)/(tp+fn+eps)\n"
"            f1s.append((2*p*r/(p+r+eps)).item())\n"
"    return np.mean(f1s) if f1s else 0.0\n"
"\n"
"print('Running robustness tests...')\n"
"rr={}\n"
"for ck,cn in conds:\n"
"    f1=eval_robust(test_loader,ck,OPTIMAL_THRESHOLD)\n"
"    rr[ck]={'name':cn,'f1':f1}\n"
"    print(f'  {cn:<20} F1={f1:.4f}')\n"
"\n"
"cf1=rr['clean']['f1']\n"
"print(f\"\\n{'Condition':<20} {'F1':>8} {'Delta':>10}\")\n"
"print('='*40)\n"
"for k,v in rr.items(): print(f\"{v['name']:<20} {v['f1']:>8.4f} {v['f1']-cf1:>+10.4f}\")\n"
"\n"
"fig,ax=plt.subplots(figsize=(12,5))\n"
"ns=[v['name'] for v in rr.values()]; fs=[v['f1'] for v in rr.values()]\n"
"cols=['#2ecc71' if k=='clean' else '#3498db' for k in rr]\n"
"bars=ax.bar(range(len(ns)),fs,color=cols)\n"
"ax.axhline(y=cf1,color='red',ls='--',alpha=0.7,label=f'Clean: {cf1:.3f}')\n"
"ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns,rotation=35,ha='right',fontsize=9)\n"
"ax.set_ylabel('Tampered-Only F1'); ax.set_title('Robustness Testing')\n"
"ax.legend(); ax.grid(True,alpha=0.3,axis='y')\n"
"for b,v in zip(bars,fs): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f'{v:.3f}',ha='center',fontsize=8)\n"
"plt.tight_layout(); plt.show()",
'v106-rob-code'),
]

if conc_idx:
    print(f"Conclusion at {conc_idx}")
    for i, c in enumerate(robust_cells):
        cells.insert(conc_idx + i, c)

print(f"After robust: {len(cells)} cells")

# ================================================================
# FIX SECTION NUMBERING
# ================================================================
renames = {
    '## 10. Training Loop': '## 11. Training Loop',
    '## 11. Evaluation': '## 12. Evaluation',
    '### 11.1 Metric Inflation': '### 12.1 Metric Inflation',
    '### 11.2 Training Curves': '### 12.2 Training Curves',
    '## 10. Visualization of Predictions': '## 13. Visualization of Predictions',
    '### 10.1 Sample Collection': '### 13.1 Sample Collection',
    '### 10.2 Submission-Ready': '### 13.2 Submission-Ready',
    '## 13. Inference Examples': '## 14. Inference Examples',
}
for cell in cells:
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        changed = False
        for old, new in renames.items():
            if old in src:
                src = src.replace(old, new)
                changed = True
        if changed:
            lines = src.split('\n')
            cell['source'] = [l+'\n' for l in lines[:-1]] + [lines[-1]]

# ================================================================
# SAVE
# ================================================================
nb['cells'] = cells
with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nFINAL: {len(cells)} cells written to {NB}")
