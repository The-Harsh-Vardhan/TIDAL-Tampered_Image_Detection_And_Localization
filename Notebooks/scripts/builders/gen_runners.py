#!/usr/bin/env python3
"""Generate 6 W&B runner notebooks for Kaggle deployment."""
import json, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "Runs")
DATASET_PATH = "/kaggle/input/vrpx-source-notebooks"

RUNNERS = {
    1: {
        "desc": "Heavy (P.7 50ep) + 3 Light",
        "experiments": [
            ("vR.P.7 Image Detection and Localisation.ipynb",   "vR.P.7",   "ELA Extended Training (50ep)",   "run01"),
            ("vR.P.0 Image Detection and Localisation.ipynb",   "vR.P.0",   "Baseline (no GT)",               "run01"),
            ("vR.P.1 Image Detection and Localisation.ipynb",   "vR.P.1",   "Dataset Fix Baseline",           "run01"),
            ("vR.P.1.5 Image Detection and Localisation.ipynb", "vR.P.1.5", "Speed Optimizations",            "run01"),
        ]
    },
    2: {
        "desc": "Heavy (P.8 40ep) + 3 Light",
        "experiments": [
            ("vR.P.8 Image Detection and Localisation.ipynb",   "vR.P.8",  "Progressive Unfreeze (40ep)",     "run01"),
            ("vR.P.2 Image Detection and Localisation.ipynb",   "vR.P.2",  "Gradual Encoder Unfreeze",        "run01"),
            ("vR.P.3.1 Image Detection and Localisation.ipynb", "vR.P.3",  "ELA Input (breakthrough)",        "run01"),
            ("vR.P.3.1 Image Detection and Localisation.ipynb", "vR.P.3",  "ELA Input (reproducibility)",     "run02"),
        ]
    },
    3: {
        "desc": "Heavy (P.12 50ep) + 3 Light",
        "experiments": [
            ("vR.P.12 Image Detection and Localisation.ipynb",  "vR.P.12", "ELA + Augmentation (50ep)",       "run01"),
            ("vR.P.4 Image Detection and Localisation.ipynb",   "vR.P.4",  "4ch RGB+ELA",                     "run01"),
            ("vR.P.5 Image Detection and Localisation.ipynb",   "vR.P.5",  "ResNet-50 Encoder",               "run01"),
            ("vR.P.6 Image Detection and Localisation.ipynb",   "vR.P.6",  "EfficientNet-B0",                 "run01"),
        ]
    },
    4: {
        "desc": "Heavy (P.11 50ep 512px) + 3 Light",
        "experiments": [
            ("vR.P.11 Image Detection and Localisation.ipynb",  "vR.P.11", "Hi-Res 512x512 (50ep)",           "run01"),
            ("vR.P.9 Image Detection and Localisation.ipynb",   "vR.P.9",  "Focal+Dice Loss",                 "run01"),
            ("vR.P.16 Image Detection and Localisation.ipynb",  "vR.P.16", "DCT Spatial Maps",                "run01"),
            ("vR.P.17 Image Detection and Localisation.ipynb",  "vR.P.17", "ELA+DCT 6ch Fusion",              "run01"),
        ]
    },
    5: {
        "desc": "3 Medium (CBAM x2 + Multi-Q ELA)",
        "experiments": [
            ("vR.P.10 Image Detection and Localisation.ipynb",  "vR.P.10", "CBAM Attention",                  "run01"),
            ("vR.P.10 Image Detection and Localisation.ipynb",  "vR.P.10", "CBAM Attention (repro)",          "run02"),
            ("vR.P.15 Image Detection and Localisation.ipynb",  "vR.P.15", "Multi-Quality ELA",               "run01"),
        ]
    },
    6: {
        "desc": "3 Medium (TTA x2 + P.18 eval)",
        "experiments": [
            ("vR.P.14 Image Detection and Localisation.ipynb",  "vR.P.14", "TTA 4-view",                      "run01"),
            ("vR.P.14 Image Detection and Localisation.ipynb",  "vR.P.14", "TTA 4-view (P.14b)",              "run02"),
            ("vR.P.18 Image Detection and Localisation.ipynb",  "vR.P.18", "JPEG Robustness (eval-only)",     "run01"),
        ]
    },
}


def make_runner(runner_id, config):
    cells = []

    # ── Markdown: title + experiment list ──
    exp_lines = [f"{i+1}. **{e[1]}** -- {e[2]} ({e[3]})  \n"
                 for i, e in enumerate(config["experiments"])]
    cells.append({
        "cell_type": "markdown", "metadata": {},
        "source": [
            f"# W&B Rerun — Runner {runner_id} of 6\n",
            f"\n",
            f"**Assignment:** {config['desc']}  \n",
            f"**Experiments:** {len(config['experiments'])}  \n",
            f"\n",
        ] + exp_lines + [
            f"\n",
            f"Each notebook runs sequentially via **papermill**, logging metrics\n",
            f"to the `tamper-detection-ablation` W&B project in real-time.\n",
        ]
    })

    # ── Cell 1: Setup ──
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "# ============================================================\n",
            f"# Runner {runner_id} of 6 -- Setup\n",
            "# ============================================================\n",
            "!pip install -q papermill wandb segmentation-models-pytorch albumentations\n",
            "\n",
            "import os, gc, time, json\n",
            "import torch\n",
            "import papermill as pm\n",
            "\n",
            f"RUNNER_ID = {runner_id}\n",
            f'DATASET_PATH = "{DATASET_PATH}"\n',
            'OUTPUT_DIR = "/kaggle/working/executed"\n',
            "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
            "\n",
            "# Verify GPU\n",
            "if torch.cuda.is_available():\n",
            "    gpu = torch.cuda.get_device_name(0)\n",
            "    mem = torch.cuda.get_device_properties(0).total_mem / 1e9\n",
            "    print(f'GPU: {gpu} ({mem:.1f} GB)')\n",
            "else:\n",
            "    print('WARNING: No GPU detected!')\n",
            "\n",
            "# Verify source notebooks exist\n",
            "if os.path.isdir(DATASET_PATH):\n",
            "    files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.ipynb')]\n",
            "    print(f'Source dataset: {len(files)} notebooks found')\n",
            "else:\n",
            "    print(f'ERROR: Dataset not found at {DATASET_PATH}')\n",
            "    print('Upload all vR.P.x notebooks as a Kaggle dataset named \"vrpx-source-notebooks\"')\n",
        ]
    })

    # ── Cell 2: Experiment queue ──
    q_lines = []
    for nb_file, version, desc, run_id in config["experiments"]:
        q_lines.append(f'    {{"file": "{nb_file}", "version": "{version}", '
                       f'"desc": "{desc}", "run_id": "{run_id}"}},\n')

    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "# ============================================================\n",
            f"# Runner {runner_id} -- Experiment Queue ({len(config['experiments'])} experiments)\n",
            "# ============================================================\n",
            "EXPERIMENTS = [\n",
        ] + q_lines + [
            "]\n",
            "\n",
            "print(f'\\nRunner {RUNNER_ID}: {len(EXPERIMENTS)} experiments queued')\n",
            "print('=' * 60)\n",
            "for i, exp in enumerate(EXPERIMENTS, 1):\n",
            "    src = os.path.join(DATASET_PATH, exp['file'])\n",
            "    exists = 'OK' if os.path.exists(src) else 'MISSING'\n",
            "    print(f'  {i}. [{exists:>7}] {exp[\"version\"]:12} {exp[\"desc\"]} ({exp[\"run_id\"]})')\n",
        ]
    })

    # ── Cell 3: Sequential execution ──
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "# ============================================================\n",
            f"# Runner {runner_id} -- Sequential Execution\n",
            "# ============================================================\n",
            "results = []\n",
            "run_start = time.time()\n",
            "\n",
            "for idx, exp in enumerate(EXPERIMENTS, 1):\n",
            "    nb_path = os.path.join(DATASET_PATH, exp['file'])\n",
            "    out_name = f\"{exp['version'].replace('.', '')}_{exp['run_id']}_output.ipynb\"\n",
            "    out_path = os.path.join(OUTPUT_DIR, out_name)\n",
            "\n",
            "    print(f'\\n{\"=\" * 70}')\n",
            "    print(f'  [{idx}/{len(EXPERIMENTS)}] {exp[\"version\"]} -- {exp[\"desc\"]} ({exp[\"run_id\"]})')\n",
            "    print(f'{\"=\" * 70}')\n",
            "\n",
            "    if not os.path.exists(nb_path):\n",
            "        print(f'  SKIP: {nb_path} not found')\n",
            "        results.append({**exp, 'status': 'NOT_FOUND', 'time_min': 0})\n",
            "        continue\n",
            "\n",
            "    t0 = time.time()\n",
            "    try:\n",
            "        pm.execute_notebook(\n",
            "            nb_path,\n",
            "            out_path,\n",
            "            kernel_name='python3',\n",
            "            cwd='/kaggle/working',\n",
            "        )\n",
            "        elapsed = (time.time() - t0) / 60\n",
            "        print(f'  SUCCESS in {elapsed:.1f} min')\n",
            "        results.append({**exp, 'status': 'SUCCESS', 'time_min': round(elapsed, 1)})\n",
            "\n",
            "    except pm.PapermillExecutionError as e:\n",
            "        elapsed = (time.time() - t0) / 60\n",
            "        err_msg = f'{e.ename}: {str(e.evalue)[:200]}'\n",
            "        print(f'  FAILED after {elapsed:.1f} min')\n",
            "        print(f'  {err_msg}')\n",
            "        results.append({**exp, 'status': 'FAILED', 'time_min': round(elapsed, 1), 'error': err_msg})\n",
            "\n",
            "    except Exception as e:\n",
            "        elapsed = (time.time() - t0) / 60\n",
            "        err_msg = str(e)[:200]\n",
            "        print(f'  ERROR after {elapsed:.1f} min: {err_msg}')\n",
            "        results.append({**exp, 'status': 'ERROR', 'time_min': round(elapsed, 1), 'error': err_msg})\n",
            "\n",
            "    # ── GPU cleanup between experiments ──\n",
            "    if torch.cuda.is_available():\n",
            "        torch.cuda.empty_cache()\n",
            "    gc.collect()\n",
            "    print(f'  GPU cache cleared')\n",
            "\n",
            "total_min = (time.time() - run_start) / 60\n",
            "print(f'\\nAll experiments processed. Total wall time: {total_min:.1f} min ({total_min/60:.1f} h)')\n",
        ]
    })

    # ── Cell 4: Summary ──
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "# ============================================================\n",
            f"# Runner {runner_id} -- Execution Summary\n",
            "# ============================================================\n",
            "print(f'\\n{\"=\" * 70}')\n",
            f"print(f'  RUNNER {runner_id} COMPLETE')\n",
            "print(f'{\"=\" * 70}\\n')\n",
            "\n",
            "gpu_total = sum(r['time_min'] for r in results)\n",
            "ok = [r for r in results if r['status'] == 'SUCCESS']\n",
            "fail = [r for r in results if r['status'] != 'SUCCESS']\n",
            "\n",
            "for r in results:\n",
            "    icon = 'OK' if r['status'] == 'SUCCESS' else r['status']\n",
            "    print(f\"  [{icon:>9}] {r['version']:12} {r['desc']:40} {r['time_min']:6.1f} min\")\n",
            "    if 'error' in r:\n",
            "        print(f\"             {r['error'][:100]}\")\n",
            "\n",
            "print(f'\\n  Succeeded: {len(ok)}/{len(results)}')\n",
            "if fail:\n",
            "    print(f'  Failed:    {len(fail)}/{len(results)}')\n",
            "print(f'  GPU time:  {gpu_total:.1f} min ({gpu_total/60:.1f} h)')\n",
            "\n",
            "# Save JSON summary\n",
            f"summary_path = os.path.join(OUTPUT_DIR, 'runner_{runner_id}_summary.json')\n",
            "with open(summary_path, 'w') as f:\n",
            "    json.dump(results, f, indent=2, default=str)\n",
            "print(f'\\n  Summary saved: {summary_path}')\n",
            "\n",
            "# List output notebooks\n",
            "print(f'\\n  Output notebooks:')\n",
            "for f_name in sorted(os.listdir(OUTPUT_DIR)):\n",
            "    if f_name.endswith('.ipynb'):\n",
            "        size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f_name)) / 1e6\n",
            "        print(f'    {f_name} ({size_mb:.1f} MB)')\n",
        ]
    })

    return {
        "cells": cells,
        "metadata": {
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [],
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": True
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


# ── Generate all 6 ──
for rid, cfg in RUNNERS.items():
    nb = make_runner(rid, cfg)
    path = os.path.join(OUT_DIR, f"wandb_runner_{rid}.ipynb")
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    print(f"  Created {path} ({len(cfg['experiments'])} experiments: {cfg['desc']})")

print(f"\nDone: 6 runner notebooks created in {OUT_DIR}/")
