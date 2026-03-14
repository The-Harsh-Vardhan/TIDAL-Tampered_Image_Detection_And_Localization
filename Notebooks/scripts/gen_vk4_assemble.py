"""
Assembles the vK.4 notebook from cell definitions in gen_vk4_cells*.py
"""
import json
import sys
sys.path.insert(0, '.')

from gen_vk4_cells import (cells_header, cells_env_setup, cells_wandb,
                            cells_dataset, cells_augmentation, cells_dataset_class)
from gen_vk4_cells2 import (cells_model, cells_loss_optimizer, cells_metrics,
                             cells_training, cells_curves)
from gen_vk4_cells3 import (cells_evaluation, cells_visualization, cells_gradcam,
                             cells_robustness, cells_shortcut, cells_save_artifacts,
                             cells_conclusion)


def build_cell(cell_type, source):
    """Build a notebook cell dict."""
    lines = source.split('\n')
    # Convert to notebook source format (each line ends with \n except last)
    nb_source = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            nb_source.append(line + '\n')
        else:
            nb_source.append(line)

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": nb_source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main():
    # Collect all cells in order
    all_cells = []
    all_cells.extend(cells_header())
    all_cells.extend(cells_env_setup())
    all_cells.extend(cells_wandb())
    all_cells.extend(cells_dataset())
    all_cells.extend(cells_augmentation())
    all_cells.extend(cells_dataset_class())
    all_cells.extend(cells_model())
    all_cells.extend(cells_loss_optimizer())
    all_cells.extend(cells_metrics())
    all_cells.extend(cells_training())
    all_cells.extend(cells_curves())
    all_cells.extend(cells_evaluation())
    all_cells.extend(cells_visualization())
    all_cells.extend(cells_gradcam())
    all_cells.extend(cells_robustness())
    all_cells.extend(cells_shortcut())
    all_cells.extend(cells_save_artifacts())
    all_cells.extend(cells_conclusion())

    # Build notebook
    nb_cells = [build_cell(ct, src) for ct, src in all_cells]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [
                    {
                        "sourceId": 0,
                        "sourceType": "datasetVersion",
                        "datasetSlug": "casia-spicing-detection-localization",
                        "isSourceIdPinned": False
                    }
                ],
                "isInternetEnabled": True,
                "isGpuEnabled": True
            }
        },
        "cells": nb_cells
    }

    out_path = "vK.4 Image Detection and Localisation.ipynb"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"Generated {out_path}")
    print(f"  Total cells: {len(nb_cells)}")
    md_count = sum(1 for ct, _ in all_cells if ct == 'markdown')
    code_count = sum(1 for ct, _ in all_cells if ct == 'code')
    print(f"  Markdown: {md_count}, Code: {code_count}")


if __name__ == '__main__':
    main()
