import json
import sys

INPUT  = "c:/D Drive/Projects/FakeShield/Fakeshield Lite/vF.0_FakeShield_Lite_v0.ipynb"
OUTPUT = "c:/D Drive/Projects/FakeShield/Fakeshield Lite/vF.0.1_FakeShield_Lite_v0.1.ipynb"

with open(INPUT, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]


def set_source(idx, lines):
    cells[idx]["source"] = lines


# ============================================================
# CELL 0: Title - version bump
# ============================================================
set_source(0, [
    "# vF.0.1 FakeShield-Lite v0.1 \u2014 Baseline\n",
    "# Tampered Image Detection & Localization\n",
    "\n",
    "---\n",
    "\n",
    "**Assignment:** Big Vision Internship \u2014 Tampered Image Detection & Localization \n",
    "**Model:** FakeShield-Lite (pruned from FakeShield, Xu et al., ICLR 2025) \n",
    "**Compute target:** Google Colab T4 GPU (16 GB VRAM) \n",
    "**Dataset:** [CASIA Splicing Detection + Localization](https://www.kaggle.com/datasets/sagnikkayalcse52/casia-spicing-detection-localization) (Kaggle, ~2.9 GB) \n",
    "**Version:** vF.0.1 \u2014 Baseline with Kaggle CASIA dataset integration\n",
    "\n",
    "---"
])

# ============================================================
# CELL 2: Install deps - add kagglehub
# ============================================================
set_source(2, [
    "# ============================================================================\n",
    "# 1.1 Install Dependencies\n",
    "# ============================================================================\n",
    "!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118\n",
    "!pip install -q transformers==4.37.2\n",
    "!pip install -q segment-anything\n",
    "!pip install -q albumentations>=1.3.1\n",
    "!pip install -q opencv-python-headless\n",
    "!pip install -q matplotlib scikit-learn tqdm\n",
    "!pip install -q kagglehub"
])

# ============================================================
# CELL 7: Dataset description
# ============================================================
set_source(7, [
    "### 3.1 Dataset Source\n",
    "\n",
    "We use the **CASIA Splicing Detection + Localization** dataset from Kaggle, which is a\n",
    "corrected ground-truth version of CASIA 2.0 (originally from\n",
    "[SunnyHaze/CASIA2.0-Corrected-Groundtruth](https://github.com/SunnyHaze/CASIA2.0-Corrected-Groundtruth)).\n",
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
    "```\n",
    "\n",
    "The dataset is downloaded automatically via `kagglehub` in the next cell."
])

# ============================================================
# CELL 8: Replace Drive mount with Kaggle download
# ============================================================
set_source(8, [
    "# ============================================================================\n",
    "# 3.1 Download Dataset from Kaggle\n",
    "# ============================================================================\n",
    "import kagglehub\n",
    "\n",
    "# Download the dataset (cached after first download)\n",
    "dataset_path = kagglehub.dataset_download(\n",
    '    "sagnikkayalcse52/casia-spicing-detection-localization"\n',
    ")\n",
    'print(f"Dataset downloaded to: {dataset_path}")\n',
    "\n",
    "# Locate the actual data root (may be nested inside the download)\n",
    "# The dataset contains Au/, Tp/, Gt/ folders\n",
    "import pathlib\n",
    "\n",
    "def find_dataset_root(base_path: str) -> str:\n",
    '    """Walk download directory to find the folder containing Au/ and Tp/."""\n',
    "    base = pathlib.Path(base_path)\n",
    "    # Check if Au/ is directly here\n",
    '    if (base / "Au").is_dir() and (base / "Tp").is_dir():\n',
    "        return str(base)\n",
    "    # Search one or two levels deep\n",
    '    for p in sorted(base.rglob("Au")):\n',
    "        if p.is_dir():\n",
    "            candidate = p.parent\n",
    '            if (candidate / "Tp").is_dir():\n',
    "                return str(candidate)\n",
    '    raise FileNotFoundError(f"Could not find Au/ + Tp/ under {base_path}")\n',
    "\n",
    "DATASET_DIR = find_dataset_root(dataset_path)\n",
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
# CELL 9: Mask discovery for CASIA Gt/ naming
# ============================================================
set_source(9, [
    "# ============================================================================\n",
    "# 3.2 Discover & Pair Images with Masks\n",
    "# ============================================================================\n",
    "#\n",
    "# CASIA 2.0 ground-truth naming conventions:\n",
    "#   Tampered image : Tp/Tp_D_CNN_M_N_sec00011_cha00085_11227.jpg\n",
    "#   Ground truth   : Gt/Tp_D_CNN_M_N_sec00011_cha00085_11227_gt.png\n",
    "#                 or Gt/Tp_D_CNN_M_N_sec00011_cha00085_11227.png\n",
    "# We handle both conventions.\n",
    "\n",
    'IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}\n',
    "\n",
    "def find_images(directory: str) -> List[str]:\n",
    '    """Recursively find all image files in a directory."""\n',
    "    images = []\n",
    "    if not os.path.isdir(directory):\n",
    "        return images\n",
    "    for root, _, files in os.walk(directory):\n",
    "        for f in files:\n",
    "            if os.path.splitext(f)[1].lower() in IMG_EXTS:\n",
    "                images.append(os.path.join(root, f))\n",
    "    return sorted(images)\n",
    "\n",
    "\n",
    "def find_mask_for_image(image_path: str, mask_dirs: List[str]) -> Optional[str]:\n",
    '    """\n',
    "    Find the corresponding ground-truth mask for a tampered image.\n",
    "    Handles CASIA naming conventions:\n",
    "      - same stem + _gt suffix\n",
    "      - same stem directly\n",
    "      - same stem with different extension\n",
    '    """\n',
    "    stem = os.path.splitext(os.path.basename(image_path))[0]\n",
    '    candidates = [stem, stem + "_gt", stem + "_mask"]\n',
    "    for mdir in mask_dirs:\n",
    "        for name in candidates:\n",
    '            for ext in [".png", ".bmp", ".jpg", ".tif"]:\n',
    "                p = os.path.join(mdir, name + ext)\n",
    "                if os.path.exists(p):\n",
    "                    return p\n",
    "    return None\n",
    "\n",
    "\n",
    "# Locate mask directories\n",
    "mask_dir_candidates = [\n",
    "    os.path.join(DATASET_DIR, d)\n",
    '    for d in ["Gt", "GT", "gt", "mask", "masks", "Mask"]\n',
    "    if os.path.isdir(os.path.join(DATASET_DIR, d))\n",
    "]\n",
    'print(f"Mask directories found: {mask_dir_candidates}")\n',
    "\n",
    "# --- Authentic images ---\n",
    'au_images = find_images(os.path.join(DATASET_DIR, "Au"))\n',
    "\n",
    "# --- Tampered images ---\n",
    'tp_images = find_images(os.path.join(DATASET_DIR, "Tp"))\n',
    "\n",
    "# Pair tampered images with their ground-truth masks\n",
    "tp_paired = []    # (image_path, mask_path)\n",
    "tp_no_mask = []\n",
    "for img_path in tp_images:\n",
    "    m = find_mask_for_image(img_path, mask_dir_candidates)\n",
    "    if m:\n",
    "        tp_paired.append((img_path, m))\n",
    "    else:\n",
    "        tp_no_mask.append(img_path)\n",
    "\n",
    'print(f"\\nAuthentic images      : {len(au_images)}")\n',
    'print(f"Tampered images       : {len(tp_images)}")\n',
    'print(f"  with mask           : {len(tp_paired)}")\n',
    'print(f"  without mask (skip) : {len(tp_no_mask)}")\n',
    "\n",
    "# Show a few filename examples\n",
    "if tp_paired:\n",
    '    print(f"\\nExample pairs:")\n',
    "    for img, msk in tp_paired[:3]:\n",
    '        print(f"  img: {os.path.basename(img)}")\n',
    '        print(f"  msk: {os.path.basename(msk)}")\n',
    "        print()"
])

# ============================================================
# CELL 40: Next steps - update version reference
# ============================================================
set_source(40, [
    "---\n",
    "## Section 13 \u2014 Next Steps\n",
    "\n",
    "This notebook is **vF.0.1 \u2014 baseline with Kaggle CASIA dataset integration**.\n",
    "\n",
    "### Dataset: CASIA Splicing Detection + Localization (Kaggle)\n",
    "- Source: `sagnikkayalcse52/casia-spicing-detection-localization`\n",
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
    "**End of vF.0.1 FakeShield-Lite Baseline Notebook**"
])

# ============================================================
# Write
# ============================================================
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Written: {OUTPUT}")

# Verify
with open(OUTPUT, "r", encoding="utf-8") as f:
    nb2 = json.load(f)
md = sum(1 for c in nb2["cells"] if c["cell_type"] == "markdown")
code = sum(1 for c in nb2["cells"] if c["cell_type"] == "code")
print(f"Cells: {len(nb2['cells'])}  (md={md}, code={code})")
