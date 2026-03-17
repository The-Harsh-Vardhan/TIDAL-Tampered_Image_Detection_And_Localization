import json, copy

src_path = 'c:/D Drive/Projects/BigVision Assignment/Notebooks/vK.10.2 Image Detection and Localisation.ipynb'
dst_path = 'c:/D Drive/Projects/BigVision Assignment/Notebooks/vK.10.3 Image Detection and Localisation.ipynb'

with open(src_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb = copy.deepcopy(nb)


def set_source(cell_idx, lines):
    nb['cells'][cell_idx]['source'] = lines


def get_source(cell_idx):
    return ''.join(nb['cells'][cell_idx]['source'])


# ============================================================
# Fix 4+5: CONFIG — weight_decay=1e-4, scheduler_T_max=50
# ============================================================
cell9 = get_source(9)
cell9 = cell9.replace("'weight_decay': 0.0,", "'weight_decay': 1e-4,")
cell9 = cell9.replace("'scheduler_T_max': 10,", "'scheduler_T_max': 50,            # single cosine decay over all epochs")
nb['cells'][9]['source'] = cell9.splitlines(True)
print("[OK] Fix 4+5: CONFIG updated (weight_decay=1e-4, T_max=50)")

# ============================================================
# Fix 3: dataset_dir path mismatch in cell 16
# ============================================================
cell16 = get_source(16)
cell16 = cell16.replace(
    "DATASET_DIR = ATTACHED_DATASET_DIR",
    "DATASET_DIR = dataset_dir  # resolved in Environment Setup (Kaggle -> Drive -> API)"
)
nb['cells'][16]['source'] = cell16.splitlines(True)
print("[OK] Fix 3: DATASET_DIR now uses resolved dataset_dir")

# ============================================================
# Fix 7: Safe mask loading in cell 29 (ImageMaskDataset)
# ============================================================
cell29 = get_source(29)
cell29 = cell29.replace(
    '        if mask is None:\n            raise RuntimeError(f"Failed to read mask: {mask_path}")',
    '        if mask is None:\n            # Authentic images may lack a physical mask file \xe2\x80\x94 use zeros\n            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)'
)
nb['cells'][29]['source'] = cell29.splitlines(True)
print("[OK] Fix 7: Safe mask loading (zeros for missing masks)")

# ============================================================
# Fix 2: W&B run name and tags in cell 36
# ============================================================
cell36 = get_source(36)
cell36 = cell36.replace(
    'name="vK.10.3-unet-resnet34-seed{SEED}-kaggle\'",',
    'name=f"vK.10.3-unet-seed{SEED}-kaggle",'
)
# Also handle the case where version was already bumped or not
cell36 = cell36.replace(
    'name="vK.10.2-unet-resnet34-seed{SEED}-kaggle\'",',
    'name=f"vK.10.3-unet-seed{SEED}-kaggle",'
)
cell36 = cell36.replace(
    'tags=["vk10.1", "assignment", "amp", "checkpointing", "early-stopping"],',
    'tags=["vk10.3", "assignment", "amp", "checkpointing", "early-stopping"],'
)
cell36 = cell36.replace(
    'project="vK.10.2-tampered-image-detection-assignment",',
    'project="vK.10.3-tampered-image-detection-assignment",'
)
cell36 = cell36.replace(
    'name="vk10.1-offline",',
    'name="vk10.3-offline",'
)
nb['cells'][36]['source'] = cell36.splitlines(True)
print("[OK] Fix 2: W&B run name and tags fixed")

# ============================================================
# Fix 4 (part 2): Add weight_decay to Adam in cell 38
# ============================================================
cell38 = get_source(38)
cell38 = cell38.replace(
    "optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])",
    "optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])"
)
cell38 = cell38.replace(
    """print(f'Optimizer: Adam(lr={CONFIG["learning_rate"]})')""",
    """print(f'Optimizer: Adam(lr={CONFIG["learning_rate"]}, wd={CONFIG["weight_decay"]})')"""
)
nb['cells'][38]['source'] = cell38.splitlines(True)
print("[OK] Fix 4b: Adam now uses weight_decay from CONFIG")

# ============================================================
# Fix 1: load_checkpoint returns history (cell 42)
# ============================================================
set_source(42, [
    "def save_checkpoint(state, filepath):\n",
    "    \"\"\"Save training state to a checkpoint file.\"\"\"\n",
    "    torch.save(state, filepath)\n",
    "\n",
    "\n",
    "def load_checkpoint(filepath, model, optimizer, scaler, scheduler=None):\n",
    "    \"\"\"Restore training state from a checkpoint file.\"\"\"\n",
    "    ckpt = torch.load(filepath, map_location=device, weights_only=False)\n",
    "    model.load_state_dict(ckpt['model_state_dict'])\n",
    "    optimizer.load_state_dict(ckpt['optimizer_state_dict'])\n",
    "    scaler.load_state_dict(ckpt['scaler_state_dict'])\n",
    "    if scheduler is not None and 'scheduler_state_dict' in ckpt:\n",
    "        scheduler.load_state_dict(ckpt['scheduler_state_dict'])\n",
    "    restored_history = ckpt.get('history', None)\n",
    "    return ckpt['epoch'] + 1, ckpt.get('best_metric', 0.0), ckpt.get('best_epoch', 0), restored_history\n",
    "\n",
    "\n",
    "print('Checkpoint helpers defined.')\n",
])
print("[OK] Fix 1a: load_checkpoint now returns history")

# ============================================================
# Fix 6: train_one_epoch returns train_dice (cell 44)
# ============================================================
set_source(44, [
    "def train_one_epoch(epoch):\n",
    "    \"\"\"Train for one epoch with AMP and gradient clipping. Returns loss, acc, and train dice.\"\"\"\n",
    "    model.train()\n",
    "    running_loss = 0.0\n",
    "    correct = 0\n",
    "    all_seg_logits, all_masks, all_labels = [], [], []\n",
    "\n",
    "    for images, masks, labels in train_loader:\n",
    "        images = images.to(device)\n",
    "        masks  = masks.to(device)\n",
    "        labels = labels.to(device)\n",
    "\n",
    "        optimizer.zero_grad(set_to_none=True)\n",
    "\n",
    "        with autocast('cuda', enabled=CONFIG['use_amp']):\n",
    "            cls_logits, seg_logits = model(images)\n",
    "            loss_cls = criterion_cls(cls_logits, labels)\n",
    "            loss_seg = CONFIG['seg_bce_weight'] * bce_loss(seg_logits, masks) + \\\n",
    "                       CONFIG['seg_dice_weight'] * dice_loss(seg_logits, masks)\n",
    "            loss = ALPHA * loss_cls + BETA * loss_seg\n",
    "\n",
    "        scaler.scale(loss).backward()\n",
    "        scaler.unscale_(optimizer)\n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])\n",
    "        scaler.step(optimizer)\n",
    "        scaler.update()\n",
    "\n",
    "        running_loss += loss.item() * images.size(0)\n",
    "        preds = torch.argmax(cls_logits, dim=1)\n",
    "        correct += (preds == labels).sum().item()\n",
    "\n",
    "        all_seg_logits.append(seg_logits.detach().cpu())\n",
    "        all_masks.append(masks.cpu())\n",
    "        all_labels.append(labels.cpu())\n",
    "\n",
    "    scheduler.step()\n",
    "\n",
    "    epoch_loss = running_loss / len(train_dataset)\n",
    "    epoch_acc = correct / len(train_dataset)\n",
    "\n",
    "    # Compute train segmentation dice (tampered-only)\n",
    "    all_seg_logits = torch.cat(all_seg_logits)\n",
    "    all_masks = torch.cat(all_masks)\n",
    "    all_labels = torch.cat(all_labels)\n",
    "    tampered_mask = all_labels == 1\n",
    "    if tampered_mask.any():\n",
    "        train_dice = dice_coef(all_seg_logits[tampered_mask], all_masks[tampered_mask]).mean().item()\n",
    "    else:\n",
    "        train_dice = 0.0\n",
    "\n",
    "    return epoch_loss, epoch_acc, train_dice\n",
])
print("[OK] Fix 6a: train_one_epoch now returns train_dice")

# ============================================================
# Fix 9: Remove epoch param from evaluate (cell 45)
# ============================================================
set_source(45, [
    "@torch.no_grad()\n",
    "def evaluate(loader, dataset_len, name='Val'):\n",
    "    \"\"\"Evaluate model with AMP, returning all-sample and tampered-only metrics.\"\"\"\n",
    "    model.eval()\n",
    "    running_loss = 0.0\n",
    "    correct = 0\n",
    "    all_seg_logits, all_masks, all_labels = [], [], []\n",
    "\n",
    "    for images, masks, labels in loader:\n",
    "        images = images.to(device)\n",
    "        masks  = masks.to(device)\n",
    "        labels = labels.to(device)\n",
    "\n",
    "        with autocast('cuda', enabled=CONFIG['use_amp']):\n",
    "            cls_logits, seg_logits = model(images)\n",
    "            loss_cls = criterion_cls(cls_logits, labels)\n",
    "            loss_seg = CONFIG['seg_bce_weight'] * bce_loss(seg_logits, masks) + \\\n",
    "                       CONFIG['seg_dice_weight'] * dice_loss(seg_logits, masks)\n",
    "            loss = ALPHA * loss_cls + BETA * loss_seg\n",
    "\n",
    "        running_loss += loss.item() * images.size(0)\n",
    "        preds = torch.argmax(cls_logits, dim=1)\n",
    "        correct += (preds == labels).sum().item()\n",
    "\n",
    "        all_seg_logits.append(seg_logits.cpu())\n",
    "        all_masks.append(masks.cpu())\n",
    "        all_labels.append(labels.cpu())\n",
    "\n",
    "    all_seg_logits = torch.cat(all_seg_logits)\n",
    "    all_masks = torch.cat(all_masks)\n",
    "    all_labels = torch.cat(all_labels)\n",
    "\n",
    "    seg_metrics = compute_metrics_split(all_seg_logits, all_masks, all_labels)\n",
    "\n",
    "    epoch_loss = running_loss / dataset_len\n",
    "    epoch_acc = correct / dataset_len\n",
    "\n",
    "    seg_metrics['loss'] = epoch_loss\n",
    "    seg_metrics['acc'] = epoch_acc\n",
    "\n",
    "    print(\n",
    "        f'  {name} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | '\n",
    "        f'Dice(all): {seg_metrics[\"dice\"]:.4f} | '\n",
    "        f'Dice(tam): {seg_metrics[\"tampered_dice\"]:.4f} | '\n",
    "        f'IoU(tam): {seg_metrics[\"tampered_iou\"]:.4f}'\n",
    "    )\n",
    "    return seg_metrics\n",
])
print("[OK] Fix 9a: evaluate() no longer takes epoch param")

# ============================================================
# Fix 1b + 6b: Training state init (cell 46)
# ============================================================
set_source(46, [
    "# ================== Training state initialization ==================\n",
    "history = {\n",
    "    \"train_loss\": [], \"train_acc\": [], \"train_dice\": [],\n",
    "    \"val_loss\": [], \"val_acc\": [],\n",
    "    \"val_dice\": [], \"val_iou\": [], \"val_f1\": [],\n",
    "    \"val_tampered_dice\": [], \"val_tampered_iou\": [], \"val_tampered_f1\": [],\n",
    "    \"lr\": [],\n",
    "}\n",
    "\n",
    "best_metric = 0.0  # tampered-only Dice\n",
    "best_epoch = 0\n",
    "start_epoch = 1\n",
    "\n",
    "# Resume from checkpoint if available\n",
    "resume_path = os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt')\n",
    "if os.path.exists(resume_path):\n",
    "    start_epoch, best_metric, best_epoch, restored_history = load_checkpoint(\n",
    "        resume_path, model, optimizer, scaler, scheduler\n",
    "    )\n",
    "    if restored_history:\n",
    "        history = restored_history\n",
    "    print(f'Resumed from epoch {start_epoch}, best tampered Dice={best_metric:.4f} at epoch {best_epoch}')\n",
    "else:\n",
    "    print('Starting fresh training.')\n",
])
print("[OK] Fix 1b+6b: History init with train_dice + checkpoint restore")

# ============================================================
# Fixes 1c, 6c, 8, 9b, 10: Main training loop (cell 47)
# ============================================================
set_source(47, [
    "# ================== Main training loop ==================\n",
    "best_model_path = os.path.join(str(KAGGLE_WORKING_DIR), 'best_model.pth')\n",
    "\n",
    "for epoch in range(start_epoch, CONFIG['max_epochs'] + 1):\n",
    "    print(f'\\nEpoch {epoch}/{CONFIG[\"max_epochs\"]}')\n",
    "\n",
    "    train_loss, train_acc, train_dice = train_one_epoch(epoch)\n",
    "    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Dice(tam): {train_dice:.4f}')\n",
    "\n",
    "    val_metrics = evaluate(val_loader, len(val_dataset), name='Val')\n",
    "\n",
    "    # Record history\n",
    "    history['train_loss'].append(train_loss)\n",
    "    history['train_acc'].append(train_acc)\n",
    "    history['train_dice'].append(train_dice)\n",
    "    history['val_loss'].append(val_metrics['loss'])\n",
    "    history['val_acc'].append(val_metrics['acc'])\n",
    "    history['val_dice'].append(val_metrics['dice'])\n",
    "    history['val_iou'].append(val_metrics['iou'])\n",
    "    history['val_f1'].append(val_metrics['f1'])\n",
    "    history['val_tampered_dice'].append(val_metrics['tampered_dice'])\n",
    "    history['val_tampered_iou'].append(val_metrics['tampered_iou'])\n",
    "    history['val_tampered_f1'].append(val_metrics['tampered_f1'])\n",
    "    history['lr'].append(optimizer.param_groups[0]['lr'])\n",
    "\n",
    "    # W&B logging\n",
    "    if WANDB_ACTIVE:\n",
    "        wandb.log({\n",
    "            'epoch': epoch,\n",
    "            'train/loss': train_loss, 'train/accuracy': train_acc, 'train/dice': train_dice,\n",
    "            'val/loss': val_metrics['loss'], 'val/accuracy': val_metrics['acc'],\n",
    "            'val/dice': val_metrics['dice'], 'val/iou': val_metrics['iou'],\n",
    "            'val/f1': val_metrics['f1'],\n",
    "            'val/tampered_dice': val_metrics['tampered_dice'],\n",
    "            'val/tampered_iou': val_metrics['tampered_iou'],\n",
    "            'val/tampered_f1': val_metrics['tampered_f1'],\n",
    "            'lr': optimizer.param_groups[0]['lr'],\n",
    "        })\n",
    "\n",
    "    # Build checkpoint state (includes history for resume)\n",
    "    state = {\n",
    "        'epoch': epoch,\n",
    "        'model_state_dict': model.state_dict(),\n",
    "        'optimizer_state_dict': optimizer.state_dict(),\n",
    "        'scaler_state_dict': scaler.state_dict(),\n",
    "        'scheduler_state_dict': scheduler.state_dict(),\n",
    "        'best_metric': best_metric,\n",
    "        'best_epoch': best_epoch,\n",
    "        'config': CONFIG,\n",
    "        'history': history,\n",
    "    }\n",
    "\n",
    "    # Save last checkpoint every epoch\n",
    "    save_checkpoint(state, os.path.join(CHECKPOINT_DIR, 'last_checkpoint.pt'))\n",
    "\n",
    "    # Save history CSV every epoch for crash resilience\n",
    "    pd.DataFrame(history).to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)\n",
    "\n",
    "    # Best model selection: tampered-only Dice\n",
    "    current_metric = val_metrics['tampered_dice']\n",
    "    if current_metric > best_metric:\n",
    "        best_metric = current_metric\n",
    "        best_epoch = epoch\n",
    "        state['best_metric'] = best_metric\n",
    "        state['best_epoch'] = best_epoch\n",
    "        save_checkpoint(state, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))\n",
    "        torch.save(model.state_dict(), best_model_path)\n",
    "        print(f'  => New best model (tampered Dice={best_metric:.4f})')\n",
    "\n",
    "    # Periodic checkpoint\n",
    "    if epoch % CONFIG['checkpoint_every'] == 0:\n",
    "        save_checkpoint(state, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt'))\n",
    "\n",
    "    # Early stopping\n",
    "    if epoch - best_epoch >= CONFIG['patience']:\n",
    "        print(f'Early stopping at epoch {epoch}. Best tampered Dice={best_metric:.4f} at epoch {best_epoch}')\n",
    "        break\n",
    "\n",
    "print(f'\\nTraining complete. Best tampered Dice={best_metric:.4f} at epoch {best_epoch}')\n",
    "print(f'Training history saved to {RESULTS_DIR}/training_history.csv')\n",
])
print("[OK] Fixes 1c+6c+8+9b+10: Training loop (history in ckpt, train_dice, no empty_cache, CSV every epoch)")

# ============================================================
# Fix 9c: evaluate call in test evaluation (cell 49)
# ============================================================
cell49 = get_source(49)
cell49 = cell49.replace(
    "test_metrics = evaluate(0, test_loader, len(test_dataset), name='Test')",
    "test_metrics = evaluate(test_loader, len(test_dataset), name='Test')"
)
nb['cells'][49]['source'] = cell49.splitlines(True)
print("[OK] Fix 9c: evaluate() call in test eval updated")

# ============================================================
# Training curves — add train_dice (cell 52)
# ============================================================
cell52 = get_source(52)
cell52 = cell52.replace(
    "axes[0,1].plot(epochs_range, history_df['val_dice'], label='Dice (all)', ls='--', alpha=0.5)",
    "if 'train_dice' in history_df.columns:\n    axes[0,1].plot(epochs_range, history_df['train_dice'], label='Train Dice (tam)', ls='--', alpha=0.5)\naxes[0,1].plot(epochs_range, history_df['val_dice'], label='Val Dice (all)', ls='--', alpha=0.5)"
)
nb['cells'][52]['source'] = cell52.splitlines(True)
print("[OK] Train dice added to training curves plot")

# ============================================================
# Version bump: vK.10.2 -> vK.10.3 everywhere
# ============================================================
version_bumps = 0
for i, cell in enumerate(nb['cells']):
    old_src = cell['source'][:]
    new_src = [line.replace('vK.10.2', 'vK.10.3') for line in old_src]
    if new_src != old_src:
        cell['source'] = new_src
        version_bumps += 1
print(f"[OK] Version bump: {version_bumps} cells updated")

# ============================================================
# Save
# ============================================================
with open(dst_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nSaved: {dst_path}")
print(f"Total cells: {len(nb['cells'])}")
