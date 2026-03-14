Plan: Create vK.11.0 Image Detection and Localisation.ipynb
Context
The user wants a new notebook vK.11.0 based on vK.10.6, incorporating findings from all audit documents and Docs v11 architecture specs. The approach is constructive generation via a Python builder script (generate_vk11.py) that reads vK.10.6 as JSON and builds a new notebook, preserving working cells and replacing/adding cells where improvements are needed.

Why this change is needed: vK.10.6 has excellent engineering infrastructure (CONFIG, AMP, seeding, DataParallel, 12-feature eval suite) and the best classification metrics in the vK.x series (AUC=0.91), but trains a 31.6M-param model from scratch on 8,829 images, capping segmentation at Tam-F1=0.22. v6.5 proved that a pretrained ResNet34 encoder achieves Tam-F1=0.41 with 24.4M params. The goal is to merge v6.5's proven architecture into vK.10.6's infrastructure, plus add ELA input, edge loss, and training pipeline fixes.

Critical rules from user:

Use vK.10.6 as baseline — don't redesign arbitrarily
Don't break working functionality (eval suite, data pipeline, viz)
Only modify architecture where audits clearly indicate a necessary fix
Deliverable
File: Notebooks/helper functions/generate_vk11.py — a constructive generator script Output: Notebooks/vK.11.0 Image Detection and Localisation.ipynb

Builder Script Structure
Following the established pattern from generate_vk10.py and build_vk106.py:

# generate_vk11.py
INPUT  = NOTEBOOKS_DIR / "vK.10.6 Image Detection and Localisation.ipynb"
OUTPUT = NOTEBOOKS_DIR / "vK.11.0 Image Detection and Localisation.ipynb"
Helper functions: load_notebook(), save_notebook(), make_md_cell(), make_code_cell()

Cell-by-Cell Plan (90 cells in vK.10.6 → ~95 cells in vK.11.0)
Section 1: Header & Setup (Cells 0–7) — MODIFY
Cell	vK.10.6	vK.11.0 Action
0	Title + TOC	Edit — update version to vK.11.0, update TOC to reflect new sections (ELA viz section)
1	Project Objectives	Edit — update version, add "pretrained encoder" and "ELA" to objectives
2	Objectives list	Edit — add ELA and edge loss objectives
3	Markdown intro	Edit — update version references
4-6	Imports + dataset discovery + debug	Preserve — working as-is
7	stderr suppression (os.dup2)	Remove — this is Bug #2 from audit; silently hides CUDA errors
Section 2: Configuration (Cells 8–13) — MAJOR REWRITE
Cell	vK.10.6	vK.11.0 Action
8	Markdown header	Edit — update version
9	CONFIG dict	Rewrite — new CONFIG (details below)
10	Hyperparameter summary table	Rewrite — must match CONFIG exactly (fix Bug #1: doc/code mismatch)
11	Reproducibility + device setup	Preserve
12	VRAM auto-scaling	Rewrite — fix Bug #1: align code and docs, adjust thresholds for SMP model (~1.1GB VRAM)
13	VRAM markdown docs	Rewrite — match actual code thresholds
New CONFIG dict:

CONFIG = {
    # Data
    'image_size': 256,
    'batch_size': 8,           # auto-adjusted by VRAM
    'num_workers': 4,
    'train_ratio': 0.70,

    # Model — SMP pretrained (from v6.5, Docs v11 I1)
    'architecture': 'TamperDetector',
    'encoder_name': 'resnet34',
    'encoder_weights': 'imagenet',
    'in_channels': 4,          # RGB + ELA (Docs v11 I2)
    'n_classes': 1,
    'n_labels': 2,
    'dropout': 0.5,

    # Optimizer — AdamW with differential LR (from v6.5)
    'optimizer': 'AdamW',
    'encoder_lr': 1e-4,
    'decoder_lr': 1e-3,
    'weight_decay': 1e-4,
    'max_grad_norm': 5.0,

    # Scheduler — ReduceLROnPlateau (from v8, replaces CosineAnnealing double-cycle bug)
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,
    'scheduler_monitor': 'val_tampered_f1',  # monitor tampered-only metric

    # Loss
    'alpha': 1.5,             # cls weight (preserved)
    'beta': 1.0,              # seg weight (preserved)
    'gamma': 0.3,             # edge loss weight (NEW — Docs v11 I7)
    'focal_gamma': 2.0,
    'seg_bce_weight': 0.5,
    'seg_dice_weight': 0.5,

    # Training
    'max_epochs': 50,
    'patience': 10,
    'checkpoint_every': 10,
    'accumulation_steps': 4,   # NEW — effective batch = batch_size * 4 (from v6.5)
    'encoder_freeze_epochs': 2, # NEW — protect pretrained BN stats (Docs v11)

    # Feature Flags
    'use_amp': True,
    'use_wandb': True,
    'seg_threshold': 0.5,

    # ELA
    'ela_quality': 90,         # JPEG re-save quality for ELA computation

    # Reproducibility
    'seed': 42,
}
Section 3: Data Discovery & Split (Cells 14–24) — MOSTLY PRESERVE
Cell	Action
14-19	Preserve — dataset discovery, metadata build (working)
20	Preserve — train/val/test split (70/15/15 stratified)
21-23	Preserve — dataset summary
24	Preserve — data leakage verification (path overlap assertions)
Section 4: Data Loading (Cells 25–35) — MODIFY FOR ELA
Cell	vK.10.6	vK.11.0 Action
25-26	Dependency imports	Edit — add import segmentation_models_pytorch as smp
27	Markdown for transforms	Edit — mention ELA channel
28	NEW	Insert — compute_ela() function (Docs v11 Section 4.1)
29	Train transforms	Edit — add A.VerticalFlip(p=0.3), keep existing augmentations. Handle 4-channel via additional_targets for ELA
30	Val/test transforms	Preserve (resize + normalize)
31	ImageMaskDataset class	Rewrite → ELAImageMaskDataset — add ELA computation in __getitem__, stack as 4th channel
32-34	DataLoader construction	Preserve (update class name reference)
35	Pre-training visualization	Edit — add ELA channel visualization panel (1 new subplot showing RGB vs ELA)
ELA integration in Dataset:

def __getitem__(self, idx):
    # ... load image (BGR), compute ELA, convert to RGB ...
    ela = compute_ela(image_bgr, quality=CONFIG['ela_quality'])  # grayscale
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Apply augmentations to image + mask (ELA transforms with image via additional_targets)
    augmented = self.transform(image=image, mask=mask, ela=ela)
    image_t = augmented['image']      # (3, H, W)
    ela_t = augmented['ela']          # (H, W)
    # Normalize ELA to [0, 1] and stack as 4th channel
    ela_norm = (ela_t.float() / 255.0).unsqueeze(0)  # (1, H, W)
    input_tensor = torch.cat([image_t, ela_norm], dim=0)  # (4, H, W)
    return input_tensor, mask, label
Section 5: Model Architecture (Cells 36–38) — MAJOR REWRITE
Cell	vK.10.6	vK.11.0 Action
36	Markdown (architecture description)	Rewrite — describe TamperDetector with SMP encoder
37	Model classes (DoubleConv, Down, Up, UNetWithClassifier)	Rewrite → TamperDetector class (Docs v11 Section 1.2). Delete DoubleConv/Down/Up/OutConv entirely.
38	Model instantiation + DataParallel + get_base_model()	Rewrite — instantiate TamperDetector, keep DataParallel + get_base_model() pattern
New model class (from Docs v11 05_Recommended_Final_Architecture.md):

class TamperDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.segmentor = smp.Unet(
            encoder_name=config['encoder_name'],
            encoder_weights=config['encoder_weights'],
            in_channels=config['in_channels'],
            classes=config['n_classes'],
        )
        encoder_out = self.segmentor.encoder.out_channels[-1]  # 512 for ResNet34
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config['dropout']),
            nn.Linear(256, config['n_labels']),
        )

    def forward(self, x):
        features = self.segmentor.encoder(x)
        cls_logits = self.classifier(features[-1])
        decoder_output = self.segmentor.decoder(*features)
        seg_logits = self.segmentor.segmentation_head(decoder_output)
        return cls_logits, seg_logits
Section 6: W&B (Cells 39–40) — MINOR EDIT
Cell	Action
39	Edit — update project name to "vK.11.0-...", update tags
40	Edit — update run name to "vK.11.0-..."
Section 7: Training Utilities (Cells 41–46) — MODIFY
Cell	vK.10.6	vK.11.0 Action
41	Markdown	Edit — mention edge loss, differential LR
42	Loss functions (FocalLoss, dice_loss, bce_loss) + scheduler	Rewrite — keep FocalLoss and dice_loss, add edge_loss() function (Docs v11 Section 2.3), change dice_loss to per-sample mode (from v8), change scheduler to ReduceLROnPlateau, change optimizer to AdamW with 4 param groups
43	Metrics (dice_coef, iou_coef, f1_coef, compute_metrics_split)	Preserve — working correctly
44	More metrics	Preserve
45	Checkpoint save/load	Preserve — get_base_model() unwrapping works for any nn.Module
46	Checkpoint helpers	Preserve
New edge_loss function:

def edge_loss(pred_logits, gt_masks):
    pred_prob = torch.sigmoid(pred_logits)
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32,
                           device=pred_prob.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(2, 3)
    gt_edge = (F.conv2d(gt_masks, sobel_x, padding=1).abs() +
               F.conv2d(gt_masks, sobel_y, padding=1).abs()).clamp(0, 1)
    pred_edge = (F.conv2d(pred_prob, sobel_x, padding=1).abs() +
                 F.conv2d(pred_prob, sobel_y, padding=1).abs()).clamp(0, 1)
    return F.binary_cross_entropy(pred_edge, gt_edge)
Per-sample Dice (from v8): Change dice_loss to compute per-sample then average, instead of batch-level.

Optimizer setup:

base_model = get_base_model(model)
optimizer = torch.optim.AdamW([
    {'params': base_model.segmentor.encoder.parameters(), 'lr': CONFIG['encoder_lr']},
    {'params': base_model.segmentor.decoder.parameters(), 'lr': CONFIG['decoder_lr']},
    {'params': base_model.segmentor.segmentation_head.parameters(), 'lr': CONFIG['decoder_lr']},
    {'params': base_model.classifier.parameters(), 'lr': CONFIG['decoder_lr']},
], weight_decay=CONFIG['weight_decay'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=CONFIG['scheduler_patience'],
    factor=CONFIG['scheduler_factor'], verbose=True
)
Section 8: Training Loop (Cells 47–51) — MODIFY
Cell	vK.10.6	vK.11.0 Action
47	Markdown	Edit — mention gradient accumulation, encoder freeze
48	train_one_epoch	Rewrite — add gradient accumulation (from v6.5 pattern), add edge loss, remove train-time seg logit accumulation (Bug #7: memory), move scheduler.step() out (ReduceLROnPlateau needs val metric)
49	evaluate	Edit — add edge loss to val loss computation
50	History init	Preserve
51	Main training loop	Rewrite — add encoder freeze/unfreeze logic, move scheduler.step(val_metric) here, keep early stopping on tampered_dice
Gradient accumulation pattern (from v6.5):

def train_one_epoch(epoch):
    model.train()
    accum_steps = CONFIG['accumulation_steps']
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (images, masks, labels) in enumerate(train_loader):
        with autocast('cuda', enabled=CONFIG['use_amp']):
            cls_logits, seg_logits = model(images)
            loss = compute_total_loss(cls_logits, seg_logits, labels, masks) / accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # Flush partial window
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=CONFIG['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
Encoder freeze logic in main loop:

if CONFIG['encoder_freeze_epochs'] > 0:
    freeze_encoder(model)
    print(f"Encoder frozen for first {CONFIG['encoder_freeze_epochs']} epochs")

for epoch in range(start_epoch, CONFIG['max_epochs']):
    if epoch == CONFIG['encoder_freeze_epochs']:
        unfreeze_encoder(model)
        print(f"Encoder unfrozen at epoch {epoch}")

    train_loss, train_acc, train_dice = train_one_epoch(epoch)
    val_metrics = evaluate(val_loader, len(val_dataset), 'Val')
    scheduler.step(val_metrics['tampered_f1'])  # ReduceLROnPlateau
    # ... checkpointing, early stopping ...
Section 9: Evaluation Suite (Cells 52–68) — MOSTLY PRESERVE
All 12 evaluation features from vK.10.6 are preserved. Minor edits only:

Cell	vK.10.6 Feature	vK.11.0 Action
52-53	Test eval with best checkpoint	Preserve
54-56	Training curves (4 plots)	Edit — update LR plot for ReduceLROnPlateau (no longer cosine)
57-58	Threshold optimization	Edit — increase sweep to 50 points: np.linspace(0.05, 0.80, 50) (from v6.5's denser sweep)
59-60	Pixel-level AUC	Preserve
61-62	Confusion matrix + ROC + PR curves	Preserve
63-64	Per-forgery-type breakdown	Preserve
65-66	Mask-size stratification	Preserve
67-68	Shortcut learning checks	Preserve
Section 10: Visualization (Cells 69–79) — MINOR EDIT + 1 NEW
Cell	Action
69-73	Preserve — denormalize, collect_samples, overlay views
74-75	Preserve — show_samples, show_image_and_mask
76	NEW
77-79	Preserve — submission prediction grid (4-panel)
Section 11: Advanced Analysis (Cells 80–85) — MOSTLY PRESERVE
Cell	Action
80-81	Preserve — failure case analysis
82-83	Edit — Grad-CAM target layer changes from down4.conv.block to segmentor.encoder.layer4 (SMP encoder)
84-85	Edit — robustness testing: use Albumentations-based degradation (v6.5 pattern) instead of inline OpenCV; fix fragile F1 formula (Bug #6) to use the standard compute_tam_f1 function
Section 12: Inference + Cleanup (Cells 86–89) — MINOR EDIT
Cell	Action
86-87	Edit — update predict_single_image to handle 4-channel input (add ELA computation)
88-89	Preserve — conclusion, W&B teardown
Bug Fixes (from Audit_vK10.6.md)
Bug	Fix Location
#1: CONFIG/docs mismatch	Cells 9-10 — generate both from same source data
#2: stderr suppression	Cell 7 — remove os.dup2(devnull, 2) entirely
#3: CONFIG['dropout'] unused	Cell 37 — TamperDetector reads from config
#4: CONFIG['train_ratio'] unused	Cell 20 — use test_size=1-CONFIG['train_ratio']
#5: Dual best_model path	Cell 51 — single checkpoint path only
#6: Fragile robustness F1	Cell 85 — use shared compute_tam_f1()
#7: Memory accumulation in train	Cell 48 — remove seg logit accumulation; compute train dice via running average
#9: Missing OutConv class	Cell 37 — N/A, SMP handles this
#10: Section numbering drift	All markdown cells — renumber consistently
CosineAnnealing double-cycle	Cell 42 — replaced with ReduceLROnPlateau
Files Modified/Created
File	Action
Notebooks/helper functions/generate_vk11.py	CREATE — constructive generator script (~1500-2000 lines)
Notebooks/vK.11.0 Image Detection and Localisation.ipynb	CREATE — generated output notebook (~95 cells)
Key Design Decisions
Keep dual-head architecture — vK.10.6's classifier head (on bottleneck) produced AUC=0.91, better than v6.5's max-pixel approach (AUC=0.87). The TamperDetector class retains both segmentation + classification heads.

256x256 not 384x384 — vK.10.6 uses 256x256 which runs faster on T4. v6.5 used 384x384 but needed batch_size=4. With 256x256, we can use batch_size=8-16 for better training stability. This is a deliberate tradeoff.

ReduceLROnPlateau over CosineAnnealing — Fixes the accidental double-cycle bug in vK.10.6. Monitored on val_tampered_f1 (the metric we care about). v8 proved this scheduler works.

Per-sample Dice from v8 — Fixes vK.10.6's batch-level Dice bias toward large masks. Small forensic regions get fair weighting.

ELA as 4th channel — SMP auto-adapts the first conv layer for >3 channels by averaging pretrained RGB weights. No architecture hacks needed.

Gradient accumulation — batch_size=8, accum=4, effective=32. Larger effective batch for stable gradients without VRAM pressure.

Encoder freeze — 2 epochs only. Protects pretrained BN statistics from random decoder/classifier gradients during warmup.

Verification Plan
After generating the notebook:

Structural check: Verify notebook loads in VS Code / Jupyter without JSON errors
Cell count: ~95 cells (90 from vK.10.6 + 5 new/split)
CONFIG consistency: Every CONFIG key referenced in code exists in CONFIG dict
Doc/code alignment: Markdown summary table matches CONFIG values exactly
Import check: All imports present (smp, cv2, torch, albumentations, etc.)
Model shape check: TamperDetector with in_channels=4 produces (B, 1, 256, 256) seg + (B, 2) cls
Loss computation: Total loss = αFocalLoss + β(w_bceBCE + w_diceDice) + γ*EdgeLoss — all terms present
Grad-CAM target: Points to segmentor.encoder.layer4 (not old down4.conv.block)
No vK.10.6 artifacts: No references to UNetWithClassifier, DoubleConv, Down, Up classes
Eval suite complete: All 12 features present + ELA visualization (13 total)
Plan approved. Let me start implementing. I'll begin by reading the vK.10.6 notebook to get the exact cell contents, then build the generator script.

Good, I have the complete cell map. Now let me read the critical cells I need to modify/preserve. Let me read them in parallel.

