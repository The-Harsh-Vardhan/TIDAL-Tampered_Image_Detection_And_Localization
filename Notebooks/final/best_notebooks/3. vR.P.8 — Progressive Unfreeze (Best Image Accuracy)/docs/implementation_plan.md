# Implementation Plan — vR.P.8: ELA + Gradual Encoder Unfreeze

## 1. Cell-by-Cell Changes from vR.P.3

| Cell | Type | Change |
|------|------|--------|
| 0 | Markdown | Title → "vR.P.8 — ELA + Gradual Encoder Unfreeze", metadata, pipeline with 3 stages |
| 1 | Markdown | Changelog + diff table (P.3 → P.8), unfreeze schedule summary |
| 2 | Code | VERSION='vR.P.8', CHANGE, add ENCODER_LR=1e-5, EPOCHS=40, NUM_WORKERS=4, UNFREEZE_SCHEDULE |
| 9 | Code | DataLoaders: NUM_WORKERS=4, add prefetch_factor=2 |
| 11 | Markdown | Architecture: progressive unfreeze stage diagram, design decisions |
| 12 | Code | Keep model + initial freeze. Add 5 helper functions |
| 13 | Markdown | Training config: differential LR, 40 epochs, 3 stages, patience reset |
| 14 | Code | Initial optimizer = single group (Stage 0). History uses lr_encoder/lr_decoder |
| 15 | Code | Stage transition checks, optimizer rebuild, patience reset, save current_stage |
| 20 | Code | Training curves: dual LR plot, stage transition vertical lines |
| 25 | Code | Results table version info |
| 26 | Markdown | Discussion: progressive unfreeze analysis, next steps |
| 27 | Code | Model filename → vR.P.8_unet_resnet34_model.pth, config includes unfreeze info |

All other cells (3–8, 10, 16–19, 21–24) remain unchanged from vR.P.3.

---

## 2. Key Implementation Details

### 2.1 Configuration Constants (Cell 2)

```python
VERSION = 'vR.P.8'
CHANGE = 'ELA + gradual encoder unfreeze (progressive: frozen → layer4 → layer3+layer4)'
SEED = 42
IMAGE_SIZE = 384
BATCH_SIZE = 16
ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 3
NUM_CLASSES = 1
LEARNING_RATE = 1e-3          # Decoder LR
ENCODER_LR = 1e-5             # Encoder LR (100x lower)
ELA_QUALITY = 90
EPOCHS = 40                   # Up from 25 (need 26+ for Stage 2)
PATIENCE = 7
NUM_WORKERS = 4
PREFETCH_FACTOR = 2
CHECKPOINT_PATH = f'{VERSION}_checkpoint.pth'

UNFREEZE_SCHEDULE = {
    0: {'epochs': (1, 10),  'desc': 'Encoder frozen, BN unfrozen'},
    1: {'epochs': (11, 25), 'desc': 'layer4 unfrozen'},
    2: {'epochs': (26, 40), 'desc': 'layer3 + layer4 unfrozen'},
}
```

### 2.2 Helper Functions (Cell 12)

```python
def freeze_encoder(model):
    """Freeze all encoder params, then unfreeze BN for domain adaptation."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    for module in model.encoder.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            for param in module.parameters():
                param.requires_grad = True
            module.track_running_stats = True

def unfreeze_layer4(model):
    """Unfreeze layer4 of ResNet-34 encoder."""
    for param in model.encoder.layer4.parameters():
        param.requires_grad = True
    n = sum(p.numel() for p in model.encoder.layer4.parameters())
    print(f'  Unfroze encoder.layer4: {n:,} params')

def unfreeze_layer3_and_layer4(model):
    """Unfreeze layer3 + layer4 of ResNet-34 encoder."""
    for param in model.encoder.layer3.parameters():
        param.requires_grad = True
    for param in model.encoder.layer4.parameters():
        param.requires_grad = True
    n3 = sum(p.numel() for p in model.encoder.layer3.parameters())
    n4 = sum(p.numel() for p in model.encoder.layer4.parameters())
    print(f'  Unfroze encoder.layer3: {n3:,} params')
    print(f'  Unfroze encoder.layer4: {n4:,} params')

def get_current_stage(epoch):
    """Return stage number (0, 1, or 2) based on epoch."""
    if epoch <= 10:
        return 0
    elif epoch <= 25:
        return 1
    else:
        return 2

def rebuild_optimizer(model, encoder_lr, decoder_lr):
    """Rebuild optimizer with correct param groups for current freeze state."""
    encoder_trainable = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    if any(p.requires_grad and p not in set(m.weight for m in model.encoder.modules()
           if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)) and hasattr(m, 'weight'))
           for p in model.encoder.parameters()):
        # Stage 1 or 2: conv layers unfrozen → dual param groups
        optimizer = optim.Adam([
            {'params': encoder_trainable, 'lr': encoder_lr, 'weight_decay': 1e-5},
            {'params': decoder_params, 'lr': decoder_lr, 'weight_decay': 1e-5},
        ])
    else:
        # Stage 0: only BN + decoder → single group
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(all_trainable, lr=decoder_lr, weight_decay=1e-5)
    return optimizer
```

Note: The stage detection in `rebuild_optimizer` is simplified in the actual notebook — since we always call it after the appropriate unfreeze function, we can use a simpler check: `get_current_stage(epoch) > 0`.

### 2.3 Optimizer Initialization (Cell 14)

```python
# Stage 0: single param group (decoder + encoder BN at LEARNING_RATE)
trainable_params_list = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params_list, lr=LEARNING_RATE, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)
```

History dict:
```python
history = {
    'train_loss': [], 'val_loss': [],
    'val_pixel_f1': [], 'val_pixel_iou': [],
    'lr_encoder': [], 'lr_decoder': []
}
```

### 2.4 Training Loop with Stage Transitions (Cell 15)

```python
current_stage = 0

# --- Checkpoint resume with stage awareness ---
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)

    # Re-apply freeze state from saved stage
    saved_stage = ckpt.get('current_stage', 0)
    current_stage = saved_stage
    freeze_encoder(model)  # Start frozen + BN
    if saved_stage >= 1:
        unfreeze_layer4(model)
    if saved_stage >= 2:
        unfreeze_layer3_and_layer4(model)

    # Rebuild optimizer, then load state
    optimizer = rebuild_optimizer(model, ENCODER_LR, LEARNING_RATE)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    best_epoch = ckpt['best_epoch']
    patience_counter = ckpt['patience_counter']
    history = ckpt['history']
    best_model_state = ckpt.get('best_model_state')

for epoch in range(start_epoch, EPOCHS + 1):
    # --- Stage transition check ---
    new_stage = get_current_stage(epoch)
    if new_stage != current_stage:
        old_stage = current_stage
        current_stage = new_stage
        print(f'\n{"="*60}')
        print(f'  STAGE TRANSITION: {old_stage} → {current_stage} at epoch {epoch}')
        print(f'  {UNFREEZE_SCHEDULE[current_stage]["desc"]}')
        print(f'{"="*60}')

        if current_stage == 1:
            unfreeze_layer4(model)
        elif current_stage == 2:
            unfreeze_layer3_and_layer4(model)

        optimizer = rebuild_optimizer(model, ENCODER_LR, LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        scaler = GradScaler('cuda')
        patience_counter = 0  # Reset — don't penalize transition dips
        trainable_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Optimizer rebuilt | Trainable: {trainable_now:,} | Patience reset')

    # --- Record LR ---
    if len(optimizer.param_groups) == 2:
        enc_lr = optimizer.param_groups[0]['lr']
        dec_lr = optimizer.param_groups[1]['lr']
    else:
        enc_lr = 0.0
        dec_lr = optimizer.param_groups[0]['lr']

    # ... train_one_epoch, validate, scheduler.step ...

    history['lr_encoder'].append(enc_lr)
    history['lr_decoder'].append(dec_lr)

    # --- Checkpoint with stage ---
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'patience_counter': patience_counter,
        'best_model_state': best_model_state,
        'history': history,
        'current_stage': current_stage,  # NEW
    }, CHECKPOINT_PATH)
```

### 2.5 Training Curves with Stage Lines (Cell 20)

```python
# Stage transition vertical lines on all subplots
for ax in axes.flat:
    ax.axvline(x=10.5, color='orange', linestyle=':', alpha=0.7, label='Unfreeze L4')
    ax.axvline(x=25.5, color='purple', linestyle=':', alpha=0.7, label='Unfreeze L3+L4')

# LR plot: dual lines
axes[1, 1].plot(epochs_range, history['lr_encoder'], 'b-', label='Encoder LR', linewidth=2)
axes[1, 1].plot(epochs_range, history['lr_decoder'], 'r-', label='Decoder LR', linewidth=2)
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
```

### 2.6 Model Save (Cell 27)

```python
model_filename = f'{VERSION}_unet_resnet34_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_epoch': best_epoch,
    'best_val_loss': best_val_loss,
    'history': history,
    'config': {
        'version': VERSION,
        'encoder': ENCODER,
        'encoder_weights': ENCODER_WEIGHTS,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'encoder_lr': ENCODER_LR,
        'ela_quality': ELA_QUALITY,
        'input_type': 'ELA',
        'unfreeze_schedule': 'progressive: frozen → layer4 → layer3+layer4',
        'epochs_trained': len(history['train_loss']),
        'seed': SEED,
    }
}, model_filename)
```

---

## 3. Verification Checklist

- [ ] VERSION = 'vR.P.8'
- [ ] ENCODER_LR = 1e-5
- [ ] EPOCHS = 40
- [ ] NUM_WORKERS = 4
- [ ] PREFETCH_FACTOR = 2
- [ ] UNFREEZE_SCHEDULE dict defined with 3 stages
- [ ] `freeze_encoder()` function defined
- [ ] `unfreeze_layer4()` function defined
- [ ] `unfreeze_layer3_and_layer4()` function defined
- [ ] `rebuild_optimizer()` function defined
- [ ] `get_current_stage()` function defined
- [ ] `current_stage` saved in checkpoint dict
- [ ] History tracks `lr_encoder` and `lr_decoder`
- [ ] Model filename contains 'vR.P.8'
- [ ] Cell count = 28 (same as P.3)

---

## 4. Runtime Estimate

| Component | Time |
|-----------|------|
| ELA stats computation (500 images) | ~30s |
| Stage 0: 10 epochs (~500K trainable) | ~30-40 min |
| Stage 1: 15 epochs (~2M trainable) | ~55-70 min |
| Stage 2: 15 epochs (~5M trainable) | ~60-75 min |
| Evaluation + visualization | ~5 min |
| **Total** | **~150-190 min** |

Note: 40 epochs at ~4 min/epoch = ~160 min. This is within Kaggle's T4 session limit but leaves minimal margin. The checkpoint system handles session interruptions. Stages 1–2 are slightly slower due to more gradient computation on unfrozen encoder layers.
