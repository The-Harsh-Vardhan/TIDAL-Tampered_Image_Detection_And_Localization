import json

INPUT  = "c:/D Drive/Projects/FakeShield/Fakeshield Lite/vF.0.1_FakeShield_Lite_v0.1.ipynb"
OUTPUT = "c:/D Drive/Projects/FakeShield/Fakeshield Lite/vF.0.2_FakeShield_Lite_v0.2.ipynb"

with open(INPUT, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

def set_source(idx, lines):
    cells[idx]["source"] = lines

# ============================================================
# CELL 0: Title - version bump, note Kaggle runtime
# ============================================================
set_source(0, [
    "# vF.0.2 FakeShield-Lite v0.2 \u2014 Baseline\n",
    "# Tampered Image Detection & Localization\n",
    "\n",
    "---\n",
    "\n",
    "**Assignment:** Big Vision Internship \u2014 Tampered Image Detection & Localization \n",
    "**Model:** FakeShield-Lite (pruned from FakeShield, Xu et al., ICLR 2025) \n",
    "**Runtime:** Kaggle Notebook \u2014 GPU T4 (16 GB VRAM) \n",
    "**Dataset:** [CASIA Splicing Detection + Localization](https://www.kaggle.com/datasets/sagnikkayalcse52/casia-spicing-detection-localization) (connected as Kaggle input) \n",
    "**Version:** vF.0.2 \u2014 Kaggle-native baseline\n",
    "\n",
    "---"
])

# ============================================================
# CELL 2: Install deps - remove kagglehub (not needed),
#          remove torch (pre-installed on Kaggle)
# ============================================================
set_source(2, [
    "# ============================================================================\n",
    "# 1.1 Install Dependencies\n",
    "# ============================================================================\n",
    "# PyTorch and torchvision are pre-installed on Kaggle.\n",
    "# We only install packages not in the default Kaggle environment.\n",
    "!pip install -q transformers==4.37.2\n",
    "!pip install -q segment-anything\n",
    "!pip install -q albumentations>=1.3.1"
])

# ============================================================
# CELL 3: GPU check - update message for Kaggle
# ============================================================
set_source(3, [
    "# ============================================================================\n",
    "# 1.2 GPU Check\n",
    "# ============================================================================\n",
    "import torch\n",
    "\n",
    'print(f"PyTorch version : {torch.__version__}")\n',
    'print(f"CUDA available  : {torch.cuda.is_available()}")\n',
    "\n",
    "if torch.cuda.is_available():\n",
    '    print(f"GPU device      : {torch.cuda.get_device_name(0)}")\n',
    "    vram = torch.cuda.get_device_properties(0).total_mem / 1e9\n",
    '    print(f"VRAM            : {vram:.1f} GB")\n',
    "else:\n",
    '    print("WARNING: No GPU detected. Enable GPU via Settings > Accelerator > GPU T4 x2.")\n',
    "\n",
    'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
    'print(f"\\nUsing device    : {DEVICE}")'
])

# ============================================================
# CELL 7: Dataset description - note Kaggle input path
# ============================================================
set_source(7, [
    "### 3.1 Dataset Source\n",
    "\n",
    "We use the **CASIA Splicing Detection + Localization** dataset, connected directly as a\n",
    "Kaggle input. It is a corrected ground-truth version of CASIA 2.0 (from\n",
    "[SunnyHaze/CASIA2.0-Corrected-Groundtruth](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth)).\n",
    "\n",
    "On Kaggle, connected datasets are mounted at:\n",
    "```\n",
    "/kaggle/input/casia-spicing-detection-localization/\n",
    "```\n",
    "\n",
    "**Dataset structure:**\n",
    "```\n",
    "CASIA 2.0/\n",
    "  Au/     # Authentic images (~7,491 images, .jpg)\n",
    "  Tp/     # Tampered images  (~5,123 images, .jpg/.tif)\n",
    "  Gt/     # Corrected ground-truth binary masks (.png)\n",
    "```\n",
    "\n",
    "**Tampered image naming convention:**\n",
    "```\n",
    "Tp_D_CNN_M_N_sec00011_cha00085_11227.jpg\n",
    "|  |  |   | | |          |          +-- numeric ID\n",
    "|  |  |   | | +- source1 +- source2\n",
    "|  |  |   | +-- boundary (N=normal, B=blurred)\n",
    "|  |  |   +-- size (M=medium, S=small, L=large)\n",
    "|  |  +-- manipulation method code\n",
    "|  +-- D=different-image (splicing), S=same-image (copy-move)\n",
    "+-- Tp = tampered\n",
    "```"
])

# ============================================================
# CELL 8: Direct path to Kaggle input (no download needed)
# ============================================================
set_source(8, [
    "# ============================================================================\n",
    "# 3.1 Connect to Kaggle Dataset\n",
    "# ============================================================================\n",
    "# On Kaggle, the dataset is directly available under /kaggle/input/\n",
    "# No download or mounting required.\n",
    "\n",
    "import pathlib\n",
    "\n",
    "KAGGLE_INPUT = pathlib.Path(\"/kaggle/input/casia-spicing-detection-localization\")\n",
    "\n",
    "def find_dataset_root(base: pathlib.Path) -> str:\n",
    '    """Locate the folder containing Au/ and Tp/ inside the Kaggle input."""\n',
    '    if (base / "Au").is_dir() and (base / "Tp").is_dir():\n',
    "        return str(base)\n",
    '    for p in sorted(base.rglob("Au")):\n',
    "        if p.is_dir():\n",
    "            candidate = p.parent\n",
    '            if (candidate / "Tp").is_dir():\n',
    "                return str(candidate)\n",
    '    raise FileNotFoundError(f"Could not find Au/ + Tp/ under {base}")\n',
    "\n",
    "DATASET_DIR = find_dataset_root(KAGGLE_INPUT)\n",
    'print(f"Dataset root: {DATASET_DIR}")\n',
    "\n",
    "# List what we found\n",
    'for subdir in ["Au", "Tp", "Gt", "GT", "gt"]:\n',
    "    path = os.path.join(DATASET_DIR, subdir)\n",
    "    if os.path.isdir(path):\n",
    "        count = len(os.listdir(path))\n",
    '        print(f"  {subdir}/  : {count} files")'
])

# ============================================================
# CELL 17: SAM checkpoint - use /kaggle/working/ instead of /content/
# ============================================================
set_source(17, [
    "# ============================================================================\n",
    "# 5.3 Download Pretrained SAM ViT-B Weights\n",
    "# ============================================================================\n",
    "\n",
    'SAM_CHECKPOINT = "/kaggle/working/sam_vit_b_01ec64.pth"\n',
    "\n",
    "if not os.path.exists(SAM_CHECKPOINT):\n",
    '    print("Downloading SAM ViT-B checkpoint...")\n',
    "    !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \\\n",
    "          -O {SAM_CHECKPOINT}\n",
    '    print("Done.")\n',
    "else:\n",
    '    print(f"SAM checkpoint already present: {SAM_CHECKPOINT}")\n',
    "\n",
    'print(f"  Size: {os.path.getsize(SAM_CHECKPOINT) / 1e6:.0f} MB")'
])

# ============================================================
# CELL 30: Training run - update save dir to /kaggle/working/
# ============================================================
set_source(30, [
    "# ============================================================================\n",
    "# 9.2 Run Training\n",
    "# ============================================================================\n",
    "\n",
    'SAVE_DIR = "/kaggle/working/checkpoints"\n',
    "os.makedirs(SAVE_DIR, exist_ok=True)\n",
    "\n",
    "for epoch in range(NUM_EPOCHS):\n",
    '    print(f"\\nEpoch {epoch+1}/{NUM_EPOCHS}")\n',
    '    print("-" * 50)\n',
    "\n",
    "    train_loss = train_one_epoch(\n",
    "        model, train_loader, criterion, optimizer, scheduler, scaler, DEVICE)\n",
    "\n",
    "    val = validate(model, val_loader, criterion, DEVICE)\n",
    "\n",
    "    # Record history\n",
    '    history["train_loss"].append(train_loss)\n',
    '    history["val_loss"].append(val["loss"])\n',
    '    history["val_det_f1"].append(val["f1"])\n',
    '    history["val_iou"].append(val["iou"])\n',
    '    history["val_dice"].append(val["dice"])\n',
    "\n",
    '    print(f"  Train Loss : {train_loss:.4f}")\n',
    '    print(f"  Val Loss   : {val[\'loss\']:.4f}")\n',
    '    print(f"  Detection  \\u2014 ACC: {val[\'accuracy\']:.4f}  P: {val[\'precision\']:.4f}  "\n',
    '          f"R: {val[\'recall\']:.4f}  F1: {val[\'f1\']:.4f}")\n',
    '    print(f"  Localization \\u2014 IoU: {val[\'iou\']:.4f}  Dice: {val[\'dice\']:.4f}")\n',
    "\n",
    "    # Save best model\n",
    '    if val["iou"] > best_val_iou:\n',
    '        best_val_iou = val["iou"]\n',
    '        ckpt_path = os.path.join(SAVE_DIR, "best_model.pth")\n',
    "        torch.save({\n",
    '            "epoch": epoch,\n',
    '            "model_state_dict": model.state_dict(),\n',
    '            "val_metrics": val,\n',
    "        }, ckpt_path)\n",
    '        print(f"  >> New best model saved  (IoU={best_val_iou:.4f})")\n',
    "\n",
    'print("\\n" + "=" * 50)\n',
    'print(f"Training complete.  Best val IoU: {best_val_iou:.4f}")'
])

# ============================================================
# CELL 33: Load best checkpoint - use /kaggle/working/
# ============================================================
set_source(33, [
    "# ============================================================================\n",
    "# 10.1 Load Best Checkpoint for Evaluation\n",
    "# ============================================================================\n",
    "\n",
    'ckpt = torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location=DEVICE)\n',
    'model.load_state_dict(ckpt["model_state_dict"])\n',
    "model.eval()\n",
    'print(f"Loaded best checkpoint from epoch {ckpt[\'epoch\']+1}")\n',
    'print(f"  Val metrics at save: {ckpt[\'val_metrics\']}")'
])

# ============================================================
# CELL 38: Save final model - use /kaggle/working/
# ============================================================
set_source(38, [
    "# ============================================================================\n",
    "# 12.1 Save Final Weights\n",
    "# ============================================================================\n",
    "\n",
    'FINAL_PATH = "/kaggle/working/model_vF02_fakeshield_lite.pth"\n',
    "\n",
    "torch.save({\n",
    '    "model_state_dict": model.state_dict(),\n',
    '    "test_metrics": test_metrics,\n',
    '    "config": {\n',
    '        "img_size": IMG_SIZE,\n',
    '        "batch_size": BATCH_SIZE,\n',
    '        "epochs": NUM_EPOCHS,\n',
    '        "lr": LR,\n',
    "    },\n",
    "}, FINAL_PATH)\n",
    "\n",
    "size_mb = os.path.getsize(FINAL_PATH) / 1e6\n",
    'print(f"Model saved: {FINAL_PATH}  ({size_mb:.1f} MB)")\n',
    'print("This file will appear in the Output tab of your Kaggle notebook.")'
])

# ============================================================
# CELL 39: Reload demo - update path
# ============================================================
set_source(39, [
    "# ============================================================================\n",
    "# 12.2 Reload Model (demonstration)\n",
    "# ============================================================================\n",
    "\n",
    "# To reload in a fresh Kaggle session:\n",
    "#\n",
    "# model = FakeShieldLite(sam_checkpoint=SAM_CHECKPOINT).to(DEVICE)\n",
    '# ckpt = torch.load("/kaggle/working/model_vF02_fakeshield_lite.pth", map_location=DEVICE)\n',
    '# model.load_state_dict(ckpt["model_state_dict"])\n',
    "# model.eval()\n",
    "\n",
    'print("Reload code provided above (commented out).")'
])

# ============================================================
# CELL 40: Next steps - update version
# ============================================================
set_source(40, [
    "---\n",
    "## Section 13 \u2014 Next Steps\n",
    "\n",
    "This notebook is **vF.0.2 \u2014 Kaggle-native baseline**.\n",
    "\n",
    "### Dataset: CASIA Splicing Detection + Localization\n",
    "- Connected directly as Kaggle input (no download step)\n",
    "- Corrected ground-truth from [SunnyHaze/CASIA2.0-Corrected-Groundtruth](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth)\n",
    "- ~7,491 authentic + ~5,123 tampered images with corrected masks\n",
    "\n",
    "### Planned improvements for subsequent versions:\n",
    "\n",
    "| Version | Improvement | Rationale |\n",
    "|---|---|---|\n",
    "| vF.1 | Larger input resolution (384) | Better fine-grained mask quality |\n",
    "| vF.1 | Multi-dataset training (CASIA + Coverage + Columbia) | Improved generalisation |\n",
    "| vF.2 | Robustness testing (JPEG, resize, noise) | Assignment bonus points |\n",
    "| vF.2 | Ablation study (no SAM, no CLIP, no augmentation) | Demonstrates understanding |\n",
    "| vF.3 | Better encoder (CLIP ViT-L or DINOv2) | Higher detection accuracy |\n",
    "| vF.3 | Edge-aware loss | Better boundary delineation |\n",
    "\n",
    "### Key limitations of this baseline:\n",
    "\n",
    "1. **No text explanations** \u2014 FakeShield\u2019s LLM explainability is removed\n",
    "2. **SAM ViT-B** instead of ViT-H \u2014 lower segmentation quality\n",
    "3. **Single dataset** \u2014 original FakeShield trains on PS + DF + AIGC data\n",
    "4. **No domain tagging** \u2014 3-class DTG simplified to binary\n",
    "5. **Direct projection** instead of TCM \u2014 less sophisticated text-to-visual alignment\n",
    "\n",
    "---\n",
    "\n",
    "**End of vF.0.2 FakeShield-Lite Kaggle Baseline Notebook**"
])

# Write
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {OUTPUT}")

# Verify
with open(OUTPUT, "r", encoding="utf-8") as f:
    nb2 = json.load(f)
content = json.dumps(nb2, ensure_ascii=False)
print(f"Cells     : {len(nb2['cells'])}")
print(f"/content/ : {content.count('/content/')}  (should be 0)")
print(f"/kaggle/  : {content.count('/kaggle/')}")
print(f"kagglehub : {content.count('kagglehub')}  (should be 0)")
print(f"drive.mount: {content.count('drive.mount')}  (should be 0)")
