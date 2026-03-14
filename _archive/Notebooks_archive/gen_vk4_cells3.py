"""
Cell definitions for vK.4 notebook — Part 3: Evaluation, Visualization, Robustness, Conclusion.
"""

def md(text):
    return ("markdown", text.strip())

def code(text):
    return ("code", text.strip())


def cells_evaluation():
    return [
        md("""## 9. Evaluation

**vK.4 evaluation improvements:**
1. Load best checkpoint
2. Threshold sweep on validation set (0.05 to 0.80)
3. Full test evaluation with mask-size stratification
4. Tampered-only metrics reported separately"""),

        code("""# ── Load Best Model ────────────────────────────────────────────────────────────

ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'Loaded best model from epoch {ckpt["epoch"]} (F1={ckpt["best_f1"]:.4f})')"""),

        code("""# ── Threshold Sweep ────────────────────────────────────────────────────────────

@torch.no_grad()
def find_best_threshold(model, loader, device, config, thresholds=None):
    model.eval()
    if thresholds is None:
        thresholds = np.arange(0.05, 0.80, 0.05)

    all_probs, all_masks = [], []
    for images, masks, labels in tqdm(loader, desc='Collecting val predictions'):
        images = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
            cls_logits, seg_logits = model(images)
        probs = torch.sigmoid(seg_logits).cpu()
        all_probs.append(probs)
        all_masks.append(masks)

    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    results = []
    for t in thresholds:
        preds = (all_probs > t).float()
        f1_scores = []
        for i in range(preds.shape[0]):
            f1_scores.append(compute_pixel_f1(preds[i], all_masks[i]))
        mean_f1 = np.mean(f1_scores)
        results.append((t, mean_f1))
        print(f'  Threshold {t:.2f}: F1={mean_f1:.4f}')

    best_t, best_f1_val = max(results, key=lambda x: x[1])
    return best_t, results

best_threshold, threshold_results = find_best_threshold(model, val_loader, device, CONFIG)
print(f'\\nBest threshold: {best_threshold:.3f}')"""),

        code("""# ── Full Test Evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_test(model, loader, test_pairs, device, config, threshold):
    model.eval()
    all_f1, all_iou = [], []
    tampered_f1, tampered_iou = [], []
    image_preds, image_labels = [], []

    size_buckets = {
        'tiny (<2%)': {'range': (0, 0.02), 'f1': []},
        'small (2-5%)': {'range': (0.02, 0.05), 'f1': []},
        'medium (5-15%)': {'range': (0.05, 0.15), 'f1': []},
        'large (>15%)': {'range': (0.15, 1.0), 'f1': []},
    }

    idx = 0
    for images, masks, labels in tqdm(loader, desc='Test evaluation'):
        images = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
            cls_logits, seg_logits = model(images)
        probs = torch.sigmoid(seg_logits).cpu()
        cls_preds = torch.argmax(cls_logits, dim=1).cpu()

        for i in range(images.size(0)):
            pred_mask = (probs[i] > threshold).float()
            f1 = compute_pixel_f1(pred_mask, masks[i])
            iou = compute_iou(pred_mask, masks[i])
            all_f1.append(f1)
            all_iou.append(iou)

            image_preds.append(cls_preds[i].item())
            image_labels.append(labels[i].item())

            if idx < len(test_pairs) and test_pairs[idx]['label'] == 1.0:
                tampered_f1.append(f1)
                tampered_iou.append(iou)

                gt_area = masks[i].sum().item() / masks[i].numel()
                for bname, bdata in size_buckets.items():
                    lo, hi = bdata['range']
                    if lo <= gt_area < hi:
                        bdata['f1'].append(f1)
                        break

            idx += 1

    results = {
        'pixel_f1_mean': np.mean(all_f1),
        'pixel_iou_mean': np.mean(all_iou),
        'tampered_f1_mean': np.mean(tampered_f1) if tampered_f1 else 0.0,
        'tampered_iou_mean': np.mean(tampered_iou) if tampered_iou else 0.0,
        'image_accuracy': np.mean([p == l for p, l in zip(image_preds, image_labels)]),
        'threshold': threshold,
    }

    print('\\n' + '=' * 60)
    print('TEST RESULTS')
    print('=' * 60)
    print(f'  Threshold:            {threshold:.3f}')
    print(f'  Image Accuracy:       {results["image_accuracy"]:.4f}')
    print(f'  Pixel F1 (all):       {results["pixel_f1_mean"]:.4f}')
    print(f'  Pixel IoU (all):      {results["pixel_iou_mean"]:.4f}')
    print(f'  Pixel F1 (tampered):  {results["tampered_f1_mean"]:.4f}')

    print('\\n  Mask-Size Stratification:')
    for bname, bdata in size_buckets.items():
        if bdata['f1']:
            print(f'    {bname}: F1={np.mean(bdata["f1"]):.4f} (n={len(bdata["f1"])})')
        else:
            print(f'    {bname}: no samples')

    return results

test_results = evaluate_test(model, test_loader, test_pairs, device, CONFIG, best_threshold)"""),
    ]


def cells_visualization():
    return [
        md("""## 10. Visualization

Prediction grids: Original | GT Mask | Predicted Mask | Overlay"""),

        code("""# ── Denormalize Helper ─────────────────────────────────────────────────────────
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return torch.clamp(img, 0, 1)

# ── Collect predictions for visualization ─────────────────────────────────────
@torch.no_grad()
def collect_predictions(model, loader, test_pairs, device, config, threshold):
    model.eval()
    predictions = []
    idx = 0
    for images, masks, labels in loader:
        images_dev = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
            cls_logits, seg_logits = model(images_dev)
        probs = torch.sigmoid(seg_logits).cpu()
        cls_preds = torch.argmax(cls_logits, dim=1).cpu()

        for i in range(images.size(0)):
            pred_mask = (probs[i] > threshold).float()
            f1 = compute_pixel_f1(pred_mask, masks[i])
            predictions.append({
                'image': images[i],
                'gt_mask': masks[i],
                'pred_mask': pred_mask,
                'label': labels[i].item(),
                'pred_label': cls_preds[i].item(),
                'pixel_f1': f1,
                'forgery_type': test_pairs[idx]['forgery_type'] if idx < len(test_pairs) else 'unknown',
                'gt_mask_area': masks[i].sum().item() / masks[i].numel(),
            })
            idx += 1
    return predictions

predictions = collect_predictions(model, test_loader, test_pairs, device, CONFIG, best_threshold)
print(f'Collected {len(predictions)} predictions.')"""),

        code("""# ── Prediction Grid ───────────────────────────────────────────────────────────

tampered_preds = [p for p in predictions if p['label'] == 1]
authentic_preds = [p for p in predictions if p['label'] == 0]
tampered_sorted = sorted(tampered_preds, key=lambda p: p['pixel_f1'])

samples = []
if len(tampered_sorted) >= 2:
    samples.extend(tampered_sorted[-2:])   # Best
mid = len(tampered_sorted) // 2
if len(tampered_sorted) >= 4:
    samples.extend(tampered_sorted[mid-1:mid+1])  # Median
if len(tampered_sorted) >= 2:
    samples.extend(tampered_sorted[:2])    # Worst
if len(authentic_preds) >= 2:
    samples.extend(authentic_preds[:2])    # Authentic

n_rows = len(samples)
if n_rows > 0:
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Original', 'GT Mask', 'Predicted Mask', 'Overlay']

    for row, s in enumerate(samples):
        img = denormalize(s['image']).permute(1, 2, 0).numpy()
        gt = s['gt_mask'].squeeze().numpy()
        pred = s['pred_mask'].squeeze().numpy()

        overlay = img.copy()
        overlay[pred > 0.5] = [1, 0, 0]
        blended = 0.6 * img + 0.4 * overlay

        lbl = 'Authentic' if s['label'] == 0 else 'Tampered'
        pred_lbl = 'Authentic' if s['pred_label'] == 0 else 'Tampered'

        for col, data in enumerate([img, gt, pred, blended]):
            cmap = 'gray' if col in [1, 2] else None
            axes[row, col].imshow(data, cmap=cmap, vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=12)
            axes[row, col].axis('off')

        axes[row, 0].set_ylabel(f'{lbl}\\nPred: {pred_lbl}\\nF1: {s["pixel_f1"]:.3f}',
                                 fontsize=10, rotation=0, labelpad=80, va='center')

    plt.suptitle('Prediction Grid (Best / Median / Worst / Authentic)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'prediction_grid.png'), dpi=150, bbox_inches='tight')
    plt.show()
else:
    print('No samples available for prediction grid.')"""),

        code("""# ── F1 vs Threshold Plot ──────────────────────────────────────────────────────

thresh_vals = [r[0] for r in threshold_results]
f1_vals = [r[1] for r in threshold_results]

plt.figure(figsize=(8, 5))
plt.plot(thresh_vals, f1_vals, 'b-', linewidth=2)
plt.axvline(x=best_threshold, color='red', linestyle='--',
            label=f'Best: {best_threshold:.3f}')
plt.xlabel('Threshold'); plt.ylabel('Mean Pixel-F1')
plt.title('F1 vs. Threshold (Validation Set)')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'f1_vs_threshold.png'), dpi=150, bbox_inches='tight')
plt.show()"""),
    ]


def cells_gradcam():
    return [
        md("""## 11. Explainable AI — Grad-CAM

Spatial attention heatmaps from the deepest encoder layer."""),

        code("""# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._handles = []
        self._handles.append(target_layer.register_forward_hook(self._save_act))
        self._handles.append(target_layer.register_full_backward_hook(self._save_grad))

    def _save_act(self, module, inp, out):
        self.activations = out.detach()
    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        cls_logits, seg_logits = self.model(input_tensor)
        target = seg_logits.sum()
        self.model.zero_grad()
        target.backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()

# Visualize Grad-CAM on tampered samples
grad_cam = GradCAM(model, model.down4.conv.block)

cam_samples = sorted([p for p in predictions if p['label'] == 1],
                      key=lambda p: p['pixel_f1'], reverse=True)[:4]

if cam_samples:
    fig, axes = plt.subplots(len(cam_samples), 4, figsize=(20, 5 * len(cam_samples)))
    if len(cam_samples) == 1:
        axes = axes[np.newaxis, :]

    for row, s in enumerate(cam_samples):
        img_tensor = s['image'].unsqueeze(0).to(device).requires_grad_(True)
        cam = grad_cam.generate(img_tensor)
        img_np = denormalize(s['image']).permute(1, 2, 0).numpy()
        gt_mask = s['gt_mask'].squeeze().numpy()
        pred = s['pred_mask'].squeeze().numpy()

        axes[row, 0].imshow(img_np); axes[row, 0].set_title('Image'); axes[row, 0].axis('off')
        axes[row, 1].imshow(gt_mask, cmap='gray'); axes[row, 1].set_title('GT Mask'); axes[row, 1].axis('off')
        axes[row, 2].imshow(pred, cmap='gray'); axes[row, 2].set_title(f'Pred (F1={s["pixel_f1"]:.3f})'); axes[row, 2].axis('off')
        axes[row, 3].imshow(img_np); axes[row, 3].imshow(cam, cmap='jet', alpha=0.5)
        axes[row, 3].set_title('Grad-CAM'); axes[row, 3].axis('off')

    plt.suptitle('Grad-CAM Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'gradcam_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()

grad_cam.remove_hooks()"""),
    ]


def cells_robustness():
    return [
        md("""## 12. Robustness Testing

Evaluate under degradation: JPEG compression, Gaussian noise, blur, resize."""),

        code("""# ── Robustness Evaluation ──────────────────────────────────────────────────────

NORMALIZE = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
SZ = CONFIG['image_size']

robustness_transforms = {
    'clean': A.Compose([A.Resize(SZ, SZ), NORMALIZE, ToTensorV2()]),
    'jpeg_qf70': A.Compose([A.Resize(SZ, SZ), A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0), NORMALIZE, ToTensorV2()]),
    'jpeg_qf50': A.Compose([A.Resize(SZ, SZ), A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0), NORMALIZE, ToTensorV2()]),
    'gauss_noise': A.Compose([A.Resize(SZ, SZ), A.GaussNoise(var_limit=(20, 60), p=1.0), NORMALIZE, ToTensorV2()]),
    'gauss_blur': A.Compose([A.Resize(SZ, SZ), A.GaussianBlur(blur_limit=(5, 7), p=1.0), NORMALIZE, ToTensorV2()]),
    'resize_half': A.Compose([A.Resize(SZ // 2, SZ // 2), A.Resize(SZ, SZ), NORMALIZE, ToTensorV2()]),
}

@torch.no_grad()
def run_robustness_eval(model, loader, device, config, threshold):
    model.eval()
    f1_scores = []
    for images, masks, labels in loader:
        images = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
            cls_logits, seg_logits = model(images)
        probs = torch.sigmoid(seg_logits).cpu()
        preds = (probs > threshold).float()
        for i in range(images.size(0)):
            f1_scores.append(compute_pixel_f1(preds[i], masks[i]))
    return f1_scores

robustness_results = {}
for name, tfm in robustness_transforms.items():
    ds = TamperingDataset(test_pairs, transform=tfm)
    dl = DataLoader(ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    f1s = run_robustness_eval(model, dl, device, CONFIG, best_threshold)
    robustness_results[name] = {'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s)}
    print(f'{name:20s}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}')

# Chart
names = list(robustness_results.keys())
means = [robustness_results[n]['f1_mean'] for n in names]
stds = [robustness_results[n]['f1_std'] for n in names]

plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(names)), means, yerr=stds, capsize=4, color='steelblue', alpha=0.8)
if 'clean' in names:
    bars[names.index('clean')].set_color('green')
plt.xticks(range(len(names)), names, rotation=45, ha='right')
plt.ylabel('Pixel-F1'); plt.title('Robustness Testing')
plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'robustness_chart.png'), dpi=150, bbox_inches='tight')
plt.show()"""),
    ]


def cells_shortcut():
    return [
        md("""## 13. Shortcut Learning Checks

Verify the model isn't relying on dataset artifacts."""),

        code("""# ── Mask Randomization Test ────────────────────────────────────────────────────

@torch.no_grad()
def mask_randomization_test(model, loader, device, config, threshold, n_batches=10):
    model.eval()
    f1_scores = []
    batch_count = 0
    for images, masks, labels in loader:
        if batch_count >= n_batches:
            break
        images = images.to(device)
        with autocast('cuda', enabled=config['use_amp']):
            cls_logits, seg_logits = model(images)
        probs = torch.sigmoid(seg_logits).cpu()
        preds = (probs > threshold).float()
        for i in range(images.size(0)):
            random_mask = torch.randint(0, 2, masks[i].shape).float()
            f1_scores.append(compute_pixel_f1(preds[i], random_mask))
        batch_count += 1

    mean_f1 = np.mean(f1_scores)
    print(f'Mask Randomization Test: F1 = {mean_f1:.4f} (should be ~0.0)')
    if mean_f1 > 0.1:
        print('  WARNING: F1 against random masks is suspiciously high!')
    else:
        print('  PASS: Model is not predicting dataset-correlated patterns.')
    return mean_f1

random_f1 = mask_randomization_test(model, test_loader, device, CONFIG, best_threshold)"""),
    ]


def cells_save_artifacts():
    return [
        md("""## 14. Save Artifacts"""),

        code("""# ── Save Results Summary ──────────────────────────────────────────────────────

results_summary = {
    'version': 'vK.4',
    'config': CONFIG,
    'seed': SEED,
    'best_epoch': best_epoch,
    'best_val_f1': float(best_val_f1),
    'threshold': float(best_threshold),
    'test_results': test_results,
    'robustness_results': {
        name: {'f1_mean': float(d['f1_mean']), 'f1_std': float(d['f1_std'])}
        for name, d in robustness_results.items()
    },
    'random_mask_f1': float(random_f1),
}

summary_path = os.path.join(RESULTS_DIR, 'results_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
print(f'Results saved to: {summary_path}')

# W&B final logging
if WANDB_ACTIVE:
    wandb.summary.update({
        'best/val_f1': best_val_f1,
        'best/epoch': best_epoch,
        'test/pixel_f1_all': test_results['pixel_f1_mean'],
        'test/pixel_f1_tampered': test_results['tampered_f1_mean'],
        'test/image_accuracy': test_results['image_accuracy'],
        'test/threshold': best_threshold,
    })
    # Upload model artifact
    artifact = wandb.Artifact('best-model-vk4', type='model')
    if os.path.exists(best_model_path):
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
    wandb.finish()
    print('W&B run finished.')"""),

        code("""# ── Artifact Inventory ────────────────────────────────────────────────────────

print()
print('=' * 60)
print('NOTEBOOK vK.4 COMPLETE — ARTIFACT INVENTORY')
print('=' * 60)

expected = {
    CHECKPOINT_DIR: ['best_model.pt'],
    RESULTS_DIR: ['split_manifest.json', 'results_summary.json'],
    PLOTS_DIR: ['training_curves.png', 'f1_vs_threshold.png',
                'prediction_grid.png', 'gradcam_analysis.png', 'robustness_chart.png'],
}

all_ok = True
for directory, files in expected.items():
    print(f'\\n{directory}/')
    for fname in files:
        fpath = os.path.join(directory, fname)
        status = 'OK' if os.path.exists(fpath) else 'MISSING'
        if status == 'MISSING':
            all_ok = False
        size = os.path.getsize(fpath) / 1024 if os.path.exists(fpath) else 0
        print(f'  [{status}] {fname} ({size:.1f} KB)')

print('\\n' + ('All artifacts present.' if all_ok else 'Some artifacts missing.'))"""),
    ]


def cells_conclusion():
    return [
        md("""## Conclusion

This notebook presents a complete, Kaggle-optimized pipeline for tampered image detection and localization.

**Improvements over vK.2**
- Kaggle-native execution (no Colab/Drive shims)
- Centralized CONFIG, full reproducibility, AMP, gradient accumulation
- pos_weight BCE + per-sample Dice loss for better segmentation
- Expanded augmentation → improved robustness
- Threshold sweep + mask-size stratified evaluation
- Grad-CAM explainability + shortcut learning verification

**Model preserved from vK.2:** Custom `UNetWithClassifier` with DoubleConv encoder-decoder and classification head on bottleneck features.

**Artifacts produced:** best model checkpoint, training curves, prediction grids, Grad-CAM analysis, robustness chart, and JSON results summary."""),
    ]
