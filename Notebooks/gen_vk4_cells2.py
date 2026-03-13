"""
Cell definitions for vK.4 notebook generator — Part 2: Model, Loss, Training, Evaluation.
"""

def md(text):
    return ("markdown", text.strip())

def code(text):
    return ("code", text.strip())


def cells_model():
    return [
        md("""## 5. Model Architecture

**Custom U-Net with Classifier Head** (preserved from vK.2):
- `DoubleConv`, `Down`, `Up` blocks form the encoder-decoder backbone
- Segmentation head outputs a 1-channel tampering mask
- Classification head on the bottleneck predicts authentic vs tampered"""),

        code("""# ── Model Architecture (preserved from vK.2) ─────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetWithClassifier(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_labels=2):
        super().__init__()
        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        # Decoder
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, n_labels),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # bottleneck
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_logits = self.outc(x)
        cls_logits = self.classifier(x5)
        return cls_logits, seg_logits

model = UNetWithClassifier(
    n_channels=CONFIG['n_channels'],
    n_classes=CONFIG['n_classes'],
    n_labels=CONFIG['n_labels'],
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total parameters:     {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

# Shape check
with torch.no_grad():
    dummy = torch.randn(1, 3, CONFIG['image_size'], CONFIG['image_size']).to(device)
    cls_out, seg_out = model(dummy)
    print(f'Shape check: cls={cls_out.shape}, seg={seg_out.shape}')"""),
    ]


def cells_loss_optimizer():
    return [
        md("""## 6. Loss, Optimizer & Scheduler

**vK.4 improvements:**
1. **BCE `pos_weight`** — computed from training masks to fix class imbalance
2. **Per-sample Dice** — each image contributes equally to Dice gradient
3. **FocalLoss** for classification — handles class imbalance
4. **ReduceLROnPlateau** — reduces LR when val metric stalls"""),

        code("""# ── Compute pos_weight from Training Masks ────────────────────────────────────

pos_weight = None

if CONFIG['use_pos_weight']:
    print('Computing pos_weight from training masks...')
    total_fg, total_bg = 0, 0
    for pair in tqdm(train_pairs, desc='Scanning masks'):
        if pair['mask_path'] is not None:
            mask = cv2.imread(pair['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                fg = (mask > 0).sum()
                total_fg += fg
                total_bg += mask.size - fg
        else:
            total_bg += CONFIG['image_size'] ** 2

    pw_value = total_bg / max(total_fg, 1)
    pos_weight = torch.tensor([pw_value], dtype=torch.float32).to(device)
    print(f'pos_weight: {pw_value:.2f} (bg={total_bg:,}, fg={total_fg:,})')
    print(f'Foreground fraction: {total_fg / (total_fg + total_bg):.4%}')
else:
    print('pos_weight disabled.')"""),

        code("""# ── Loss Functions ────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, smooth=1.0, per_sample_dice=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.smooth = smooth
        self.per_sample_dice = per_sample_dice

    def _dice_loss_per_sample(self, logits, targets):
        probs = torch.sigmoid(logits)
        B = probs.shape[0]
        p_flat = probs.view(B, -1)
        t_flat = targets.view(B, -1)
        inter = (p_flat * t_flat).sum(dim=1)
        union = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def _dice_loss_batch(self, logits, targets):
        probs = torch.sigmoid(logits)
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)
        inter = (p_flat * t_flat).sum()
        union = p_flat.sum() + t_flat.sum()
        return 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        if self.per_sample_dice:
            dice = self._dice_loss_per_sample(logits, targets)
        else:
            dice = self._dice_loss_batch(logits, targets)
        return 0.5 * bce_loss + 0.5 * dice


# Classification loss
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]),
                                      y=[int(p['label']) for p in train_pairs])
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f'Class weights: {class_weights}')

criterion_cls = FocalLoss(alpha=class_weights)
criterion_seg = BCEDiceLoss(pos_weight=pos_weight,
                             per_sample_dice=CONFIG['dice_per_sample'])

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'],
                              weight_decay=CONFIG['weight_decay'])

# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=CONFIG['scheduler_factor'],
    patience=CONFIG['scheduler_patience'], min_lr=CONFIG['scheduler_min_lr'],
)

# AMP scaler
scaler = GradScaler('cuda', enabled=CONFIG['use_amp'])

print(f'Optimizer: Adam(lr={CONFIG[\"lr\"]}, wd={CONFIG[\"weight_decay\"]})')
print(f'Scheduler: ReduceLROnPlateau(patience={CONFIG[\"scheduler_patience\"]})')
print(f'AMP enabled: {CONFIG[\"use_amp\"]}')"""),
    ]


def cells_metrics():
    return [
        code("""# ── Evaluation Metrics ────────────────────────────────────────────────────────

def compute_pixel_f1(pred, gt, eps=1e-8):
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 and pred.sum() > 0:
        return 0.0
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return (2 * precision * recall / (precision + recall + eps)).item()

def compute_iou(pred, gt, eps=1e-8):
    pred, gt = pred.flatten(), gt.flatten()
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() - inter
    return (inter / (union + eps)).item()

def dice_coef_batch(pred_logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs > threshold).float()
    inter = (pred_bin * targets).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return ((2.0 * inter + eps) / (union + eps)).mean().item()

def iou_coef_batch(pred_logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs > threshold).float()
    inter = (pred_bin * targets).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def f1_coef_batch(pred_logits, targets, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(pred_logits)
    pred_bin = (probs > threshold).float()
    tp = (pred_bin * targets).sum(dim=(1,2,3))
    fp = (pred_bin * (1.0 - targets)).sum(dim=(1,2,3))
    fn = ((1.0 - pred_bin) * targets).sum(dim=(1,2,3))
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return (2.0 * precision * recall / (precision + recall + eps)).mean().item()

print('Metric functions defined.')"""),
    ]


def cells_training():
    return [
        md("""## 7. Training Pipeline

**vK.4 training improvements:**
- Mixed-precision (AMP) with GradScaler
- Gradient accumulation (effective batch size = batch_size × accumulation_steps)
- Gradient clipping at `max_grad_norm=1.0`
- Early stopping on validation F1 (patience=10)
- Per-epoch LR logging
- `tqdm` progress bars"""),

        code("""# ── Training & Validation Functions ───────────────────────────────────────────

ALPHA = CONFIG['cls_loss_weight']
BETA = CONFIG['seg_loss_weight']

def train_one_epoch(model, loader, criterion_cls, criterion_seg, optimizer, scaler,
                    device, config):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    accum_steps = config['accumulation_steps']
    use_amp = config['use_amp']

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(loader), total=len(loader), leave=False, desc='Training')

    for batch_idx, (images, masks, labels) in pbar:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        with autocast('cuda', enabled=use_amp):
            cls_logits, seg_logits = model(images)
            loss_cls = criterion_cls(cls_logits, labels)
            loss_seg = criterion_seg(seg_logits, masks)
            loss = (ALPHA * loss_cls + BETA * loss_seg) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps * images.size(0)
        preds = torch.argmax(cls_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix({'loss': running_loss / total, 'acc': correct / total})

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion_cls, criterion_seg, device, config):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    dice_sum, iou_sum, f1_sum = 0.0, 0.0, 0.0
    num_batches = 0
    use_amp = config['use_amp']

    for images, masks, labels in loader:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        with autocast('cuda', enabled=use_amp):
            cls_logits, seg_logits = model(images)
            loss_cls = criterion_cls(cls_logits, labels)
            loss_seg = criterion_seg(seg_logits, masks)
            loss = ALPHA * loss_cls + BETA * loss_seg

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(cls_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        dice_sum += dice_coef_batch(seg_logits, masks)
        iou_sum += iou_coef_batch(seg_logits, masks)
        f1_sum += f1_coef_batch(seg_logits, masks)
        num_batches += 1

    metrics = {
        'loss': running_loss / total,
        'acc': correct / total,
        'dice': dice_sum / max(num_batches, 1),
        'iou': iou_sum / max(num_batches, 1),
        'f1': f1_sum / max(num_batches, 1),
    }
    return metrics"""),

        code("""# ── Training Loop ─────────────────────────────────────────────────────────────

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'val_dice': [], 'val_iou': [], 'val_f1': [],
    'lr': [],
}

best_val_f1 = 0.0
best_epoch = 0
patience_counter = 0

NUM_EPOCHS = CONFIG['max_epochs']
best_model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')

print(f'Starting training for {NUM_EPOCHS} epochs...')
print(f'Early stopping patience: {CONFIG["patience"]}')
print('=' * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    current_lr = optimizer.param_groups[0]['lr']
    print(f'\\nEpoch {epoch:02d}/{NUM_EPOCHS} | LR: {current_lr:.2e}')

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion_cls, criterion_seg,
        optimizer, scaler, device, CONFIG
    )

    val_metrics = validate(model, val_loader, criterion_cls, criterion_seg, device, CONFIG)

    # Update scheduler
    scheduler.step(val_metrics['f1'])

    # Log history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_metrics['loss'])
    history['val_acc'].append(val_metrics['acc'])
    history['val_dice'].append(val_metrics['dice'])
    history['val_iou'].append(val_metrics['iou'])
    history['val_f1'].append(val_metrics['f1'])
    history['lr'].append(current_lr)

    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'  Val   Loss: {val_metrics["loss"]:.4f} | Val Acc: {val_metrics["acc"]:.4f} | '
          f'Dice: {val_metrics["dice"]:.4f} | IoU: {val_metrics["iou"]:.4f} | F1: {val_metrics["f1"]:.4f}')

    # W&B logging
    if WANDB_ACTIVE:
        wandb.log({
            'epoch': epoch, 'lr': current_lr,
            'train/loss': train_loss, 'train/accuracy': train_acc,
            'val/loss': val_metrics['loss'], 'val/accuracy': val_metrics['acc'],
            'val/dice': val_metrics['dice'], 'val/iou': val_metrics['iou'],
            'val/f1': val_metrics['f1'],
        })

    # Checkpointing
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        best_epoch = epoch
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_f1': best_val_f1,
            'config': CONFIG,
        }, best_model_path)
        print(f'  ==> Saved best model (F1: {best_val_f1:.4f})')
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f'  Early stopping triggered after {CONFIG["patience"]} epochs without improvement.')
            break

    torch.cuda.empty_cache()

print('\\n' + '=' * 60)
print(f'Training finished. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}')
print('=' * 60)"""),
    ]


def cells_curves():
    return [
        md("""## 8. Training Curves"""),

        code("""# ── Training Curves ────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
epochs_range = range(1, len(history['train_loss']) + 1)

axes[0, 0].plot(epochs_range, history['train_loss'], label='Train')
axes[0, 0].plot(epochs_range, history['val_loss'], label='Val')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training & Validation Loss')
axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs_range, history['val_f1'], color='green')
axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--',
                    label=f'Best (epoch {best_epoch})')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('F1')
axes[0, 1].set_title('Validation F1')
axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs_range, history['val_dice'], label='Dice')
axes[1, 0].plot(epochs_range, history['val_iou'], label='IoU')
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Validation Segmentation Metrics')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(epochs_range, history['lr'], color='purple')
axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Training Summary', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Training curves saved.')"""),
    ]
