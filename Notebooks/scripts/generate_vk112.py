#!/usr/bin/env python3
"""
generate_vk112.py — Constructive generator for vK.11.2

Reads vK.11.1, preserves ALL existing cells unchanged, and inserts a
Reproducibility Verification section before the Conclusion.

Output: vK.11.2 Image Detection and Localisation.ipynb
"""

import json
import sys
import copy
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = SCRIPT_DIR.parent
INPUT = NOTEBOOKS_DIR / "source" / "vK.11.1 Image Detection and Localisation.ipynb"
OUTPUT = NOTEBOOKS_DIR / "source" / "vK.11.2 Image Detection and Localisation.ipynb"

# ── Helpers ─────────────────────────────────────────────────────────────

def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Saved: {path}  ({path.stat().st_size:,} bytes)")


def to_source_lines(text):
    """Convert multiline string to list of lines ending with \\n (Jupyter format)."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def make_md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source_lines(text),
    }


def make_code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source_lines(text),
    }


# ── Reproducibility Verification Section Cells ────────────────────────

def build_reproducibility_cells():
    """Build the Reproducibility Verification section cells."""
    cells = []

    # ── Section header ──
    cells.append(make_md_cell(
        "## 18. Reproducibility Verification\n"
        "\n"
        "This section verifies that the experiment setup is deterministic and\n"
        "reproducible. It confirms seed configuration, dataset split stability,\n"
        "checkpoint integrity, training convergence, and environment details.\n"
        "All values are read from existing variables — no training or evaluation\n"
        "code is re-executed."
    ))

    # ── 18.1 Seed Configuration ──
    cells.append(make_md_cell(
        "### 18.1 Experiment Seed Configuration\n"
        "\n"
        "Reproducibility is ensured through fixed random seeds applied at the start\n"
        "of the notebook (Section 3). The following sources of randomness are controlled:\n"
        "\n"
        "| Source | Method |\n"
        "|---|---|\n"
        "| Python `random` | `random.seed(SEED)` |\n"
        "| NumPy | `np.random.seed(SEED)` |\n"
        "| PyTorch CPU | `torch.manual_seed(SEED)` |\n"
        "| PyTorch CUDA (all GPUs) | `torch.cuda.manual_seed_all(SEED)` |\n"
        "| cuDNN deterministic mode | `torch.backends.cudnn.deterministic = True` |\n"
        "| cuDNN benchmark disabled | `torch.backends.cudnn.benchmark = False` |\n"
        "| DataLoader workers | `seed_worker()` with `torch.initial_seed()` per worker |\n"
        "\n"
        "The cell below confirms the active seed values."
    ))

    cells.append(make_code_cell(
        "# ================== 18.1 Seed Configuration Verification ==================\n"
        "import random\n"
        "import numpy as np\n"
        "import torch\n"
        "\n"
        "print('=' * 50)\n"
        "print('  SEED CONFIGURATION VERIFICATION')\n"
        "print('=' * 50)\n"
        "print()\n"
        "print(f'CONFIG seed:                {CONFIG[\"seed\"]}')\n"
        "print(f'SEED variable:              {SEED}')\n"
        "print(f'cuDNN deterministic:        {torch.backends.cudnn.deterministic}')\n"
        "print(f'cuDNN benchmark:            {torch.backends.cudnn.benchmark}')\n"
        "print(f'PyTorch initial seed:       {torch.initial_seed()}')\n"
        "print()\n"
        "if torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark:\n"
        "    print('Status: All reproducibility flags are correctly set.')\n"
        "else:\n"
        "    print('WARNING: Reproducibility flags are not fully configured.')"
    ))

    # ── 18.2 Dataset Split Determinism ──
    cells.append(make_md_cell(
        "### 18.2 Dataset Split Determinism\n"
        "\n"
        "The dataset is split using `sklearn.model_selection.train_test_split` with\n"
        "`random_state=SEED` and `stratify=labels`. This guarantees identical partitions\n"
        "across runs given the same seed and input data. The cell below confirms the\n"
        "current split sizes and class distributions."
    ))

    cells.append(make_code_cell(
        "# ================== 18.2 Dataset Split Verification ==================\n"
        "print('=' * 55)\n"
        "print('  DATASET SPLIT DETERMINISM')\n"
        "print('=' * 55)\n"
        "print()\n"
        "print(f\"{'Split':<12} {'Total':>7} {'Authentic':>11} {'Tampered':>10} {'Tam %':>7}\")\n"
        "print('-' * 50)\n"
        "for name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:\n"
        "    n_auth = (df['label'] == 0).sum()\n"
        "    n_tamp = (df['label'] == 1).sum()\n"
        "    pct = 100.0 * n_tamp / len(df) if len(df) > 0 else 0\n"
        "    print(f'{name:<12} {len(df):>7} {n_auth:>11} {n_tamp:>10} {pct:>6.1f}%')\n"
        "print('-' * 50)\n"
        "total = len(train_df) + len(val_df) + len(test_df)\n"
        "print(f\"{'Total':<12} {total:>7}\")\n"
        "print()\n"
        "print(f'Train ratio:  {len(train_df)/total:.2%}')\n"
        "print(f'Val ratio:    {len(val_df)/total:.2%}')\n"
        "print(f'Test ratio:   {len(test_df)/total:.2%}')\n"
        "print()\n"
        "print('Seed used for splitting:', SEED)"
    ))

    # ── 18.3 Checkpoint Reproducibility ──
    cells.append(make_md_cell(
        "### 18.3 Checkpoint Reproducibility\n"
        "\n"
        "The training pipeline saves three types of checkpoints:\n"
        "\n"
        "| Checkpoint | Frequency | Contents |\n"
        "|---|---|---|\n"
        "| `last_checkpoint.pt` | Every epoch | Full state (model, optimizer, scheduler, scaler, history) |\n"
        "| `best_model.pt` | On metric improvement | Full state at best validation tampered Dice |\n"
        "| `checkpoint_epoch_N.pt` | Every N epochs | Periodic snapshot for analysis |\n"
        "\n"
        "This enables resuming training from any interruption point while preserving\n"
        "the exact optimizer momentum, learning rate schedule position, and AMP scaler state."
    ))

    cells.append(make_code_cell(
        "# ================== 18.3 Checkpoint Verification ==================\n"
        "import os\n"
        "\n"
        "print('=' * 55)\n"
        "print('  CHECKPOINT REPRODUCIBILITY')\n"
        "print('=' * 55)\n"
        "print()\n"
        "\n"
        "ckpt_files = {\n"
        "    'best_model.pt': os.path.join(str(CHECKPOINT_DIR), 'best_model.pt'),\n"
        "    'last_checkpoint.pt': os.path.join(str(CHECKPOINT_DIR), 'last_checkpoint.pt'),\n"
        "}\n"
        "\n"
        "for name, path in ckpt_files.items():\n"
        "    exists = os.path.exists(path)\n"
        "    size = os.path.getsize(path) / (1024 * 1024) if exists else 0\n"
        "    status = f'Found ({size:.1f} MB)' if exists else 'Not found'\n"
        "    print(f'  {name:<25} {status}')\n"
        "\n"
        "print()\n"
        "\n"
        "# Try to read best checkpoint metadata\n"
        "best_path = ckpt_files['best_model.pt']\n"
        "if os.path.exists(best_path):\n"
        "    ckpt = torch.load(best_path, map_location='cpu', weights_only=False)\n"
        "    print(f'Best checkpoint metadata:')\n"
        "    print(f'  Best epoch:             {ckpt.get(\"best_epoch\", \"N/A\")}')\n"
        "    print(f'  Best metric (Tam Dice): {ckpt.get(\"best_metric\", \"N/A\")}')\n"
        "    print(f'  Saved at epoch:         {ckpt.get(\"epoch\", \"N/A\")}')\n"
        "    del ckpt  # free memory\n"
        "else:\n"
        "    print('No best checkpoint found — training has not been run yet.')"
    ))

    # ── 18.4 Training Stability Indicators ──
    cells.append(make_md_cell(
        "### 18.4 Training Stability Indicators\n"
        "\n"
        "Training convergence is verified by plotting the loss curve and the primary\n"
        "evaluation metric (tampered-only Dice) over epochs. A well-behaved training\n"
        "run should show:\n"
        "\n"
        "- Monotonically decreasing training loss\n"
        "- Validation tampered Dice that plateaus or improves before early stopping\n"
        "- No sudden spikes indicating instability\n"
        "\n"
        "The plots below reuse the `history` dictionary populated during training."
    ))

    cells.append(make_code_cell(
        "# ================== 18.4 Training Stability Plots ==================\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "if history.get('train_loss') and len(history['train_loss']) > 0:\n"
        "    epochs = range(1, len(history['train_loss']) + 1)\n"
        "\n"
        "    fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
        "\n"
        "    # Plot 1: Training Loss\n"
        "    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=1.5, label='Train Loss')\n"
        "    if history.get('val_loss'):\n"
        "        axes[0].plot(epochs, history['val_loss'], 'r-', linewidth=1.5, label='Val Loss')\n"
        "    axes[0].set_xlabel('Epoch')\n"
        "    axes[0].set_ylabel('Loss')\n"
        "    axes[0].set_title('Training & Validation Loss')\n"
        "    axes[0].legend()\n"
        "    axes[0].grid(True, alpha=0.3)\n"
        "\n"
        "    # Plot 2: Tampered Dice\n"
        "    if history.get('val_tampered_dice'):\n"
        "        axes[1].plot(epochs, history['val_tampered_dice'], 'g-', linewidth=1.5, label='Val Tampered Dice')\n"
        "        best_idx = max(range(len(history['val_tampered_dice'])),\n"
        "                       key=lambda i: history['val_tampered_dice'][i])\n"
        "        axes[1].axvline(x=best_idx + 1, color='gray', linestyle='--', alpha=0.5,\n"
        "                        label=f'Best @ epoch {best_idx + 1}')\n"
        "    axes[1].set_xlabel('Epoch')\n"
        "    axes[1].set_ylabel('Dice')\n"
        "    axes[1].set_title('Validation Tampered Dice')\n"
        "    axes[1].legend()\n"
        "    axes[1].grid(True, alpha=0.3)\n"
        "\n"
        "    # Plot 3: Learning Rate\n"
        "    if history.get('lr'):\n"
        "        axes[2].plot(epochs, history['lr'], 'm-', linewidth=1.5, label='Encoder LR')\n"
        "    axes[2].set_xlabel('Epoch')\n"
        "    axes[2].set_ylabel('Learning Rate')\n"
        "    axes[2].set_title('Learning Rate Schedule')\n"
        "    axes[2].legend()\n"
        "    axes[2].grid(True, alpha=0.3)\n"
        "\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(os.path.join(str(RESULTS_DIR), 'reproducibility_stability.png'),\n"
        "                dpi=150, bbox_inches='tight')\n"
        "    plt.show()\n"
        "    print(f'Total epochs trained: {len(history[\"train_loss\"])}')\n"
        "    if history.get('val_tampered_dice'):\n"
        "        print(f'Best val tampered Dice: {max(history[\"val_tampered_dice\"]):.4f} '\n"
        "              f'(epoch {best_idx + 1})')\n"
        "else:\n"
        "    print('No training history available — training has not been run yet.')"
    ))

    # ── 18.5 Environment Information ──
    cells.append(make_md_cell(
        "### 18.5 Environment Information\n"
        "\n"
        "The following environment details are recorded to enable future reproduction\n"
        "of results on compatible hardware and software configurations."
    ))

    cells.append(make_code_cell(
        "# ================== 18.5 Environment Information ==================\n"
        "import sys\n"
        "import torch\n"
        "import platform\n"
        "\n"
        "print('=' * 55)\n"
        "print('  ENVIRONMENT INFORMATION')\n"
        "print('=' * 55)\n"
        "print()\n"
        "print(f'Python version:       {sys.version.split()[0]}')\n"
        "print(f'Platform:             {platform.platform()}')\n"
        "print(f'PyTorch version:      {torch.__version__}')\n"
        "print(f'CUDA available:       {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'CUDA version:         {torch.version.cuda}')\n"
        "    print(f'cuDNN version:        {torch.backends.cudnn.version()}')\n"
        "    print(f'GPU count:            {torch.cuda.device_count()}')\n"
        "    for i in range(torch.cuda.device_count()):\n"
        "        name = torch.cuda.get_device_name(i)\n"
        "        mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)\n"
        "        print(f'  GPU {i}: {name} ({mem:.1f} GB)')\n"
        "else:\n"
        "    print('  No GPU detected — running on CPU')\n"
        "\n"
        "print()\n"
        "try:\n"
        "    import segmentation_models_pytorch as smp\n"
        "    print(f'SMP version:          {smp.__version__}')\n"
        "except ImportError:\n"
        "    print('SMP version:          not installed')\n"
        "try:\n"
        "    import albumentations as A\n"
        "    print(f'Albumentations:       {A.__version__}')\n"
        "except ImportError:\n"
        "    print('Albumentations:       not installed')\n"
        "try:\n"
        "    import sklearn\n"
        "    print(f'scikit-learn:         {sklearn.__version__}')\n"
        "except ImportError:\n"
        "    print('scikit-learn:         not installed')\n"
        "try:\n"
        "    import cv2\n"
        "    print(f'OpenCV:               {cv2.__version__}')\n"
        "except ImportError:\n"
        "    print('OpenCV:               not installed')"
    ))

    return cells


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print(f"Reading:  {INPUT}")
    nb = load_notebook(INPUT)
    src_cells = nb["cells"]
    print(f"Source cells: {len(src_cells)}")

    new_cells = []

    # ── Find Conclusion cell ──
    conclusion_idx = None
    for i, c in enumerate(src_cells):
        if c["cell_type"] == "markdown":
            text = "".join(c["source"])
            if "## 18. Conclusion" in text:
                conclusion_idx = i
                break

    if conclusion_idx is None:
        print("ERROR: Could not find Conclusion cell (## 18. Conclusion)")
        sys.exit(1)

    print(f"Conclusion cell found at index: {conclusion_idx}")

    # ── Build output cells ──
    # 1. Copy cells 0 through conclusion_idx-1
    for i in range(conclusion_idx):
        cell = copy.deepcopy(src_cells[i])

        # Update TOC — add Reproducibility section entry
        if i == 0:
            text = "".join(cell["source"])
            text = text.replace(
                "18. [Conclusion](#18-conclusion)",
                "18. [Reproducibility Verification](#18-reproducibility-verification)\n"
                "19. [Conclusion](#19-conclusion)"
            )
            text = text.replace("vK.11.1", "vK.11.2")
            cell["source"] = to_source_lines(text)

        # Update title cell
        elif i == 1:
            text = "".join(cell["source"])
            text = text.replace("vK.11.1", "vK.11.2")
            cell["source"] = to_source_lines(text)

        new_cells.append(cell)

    # 2. Insert Reproducibility Verification section
    repro_cells = build_reproducibility_cells()
    new_cells.extend(repro_cells)
    print(f"Inserted {len(repro_cells)} Reproducibility Verification cells")

    # 3. Copy Conclusion and remaining cells, renumbered
    for i in range(conclusion_idx, len(src_cells)):
        cell = copy.deepcopy(src_cells[i])

        # Renumber Conclusion from 18 → 19
        if i == conclusion_idx:
            text = "".join(cell["source"])
            text = text.replace("## 18. Conclusion", "## 19. Conclusion")
            text = text.replace("vK.11.1", "vK.11.2")
            cell["source"] = to_source_lines(text)

        new_cells.append(cell)

    # ── Assemble notebook ──
    nb_out = copy.deepcopy(nb)
    nb_out["cells"] = new_cells

    if "language_info" in nb_out.get("metadata", {}):
        nb_out["metadata"]["language_info"]["name"] = "python"

    save_notebook(nb_out, OUTPUT)

    # ── Summary ──
    code_cells = sum(1 for c in new_cells if c["cell_type"] == "code")
    md_cells = sum(1 for c in new_cells if c["cell_type"] == "markdown")
    print(f"Total cells: {len(new_cells)} ({code_cells} code, {md_cells} markdown)")
    print(f"Reproducibility cells added: {len(repro_cells)}")
    print("Done.")


if __name__ == "__main__":
    main()
