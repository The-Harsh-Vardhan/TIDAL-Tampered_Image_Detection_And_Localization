#!/usr/bin/env python3
"""
fix_wandb_logging.py — Patch W&B experiment tracking across all vK.11.x notebooks.

Adds:
  - Version-correct W&B project/name/tags (was hardcoded to vK.11.0)
  - Runtime config logging (device, GPU, param count)
  - Prediction visualization during training (every 5 epochs)
  - Model artifact + CSV upload after test eval
  - wandb.Table for final metrics
  - wandb.Image for 8 evaluation visualizations (threshold sweep, confusion matrix,
    mask-size, ELA, failure cases, Grad-CAM, robustness, dashboard)
"""

import json
import copy
import os
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent

NOTEBOOKS = [
    ("source/vK.11.0 Image Detection and Localisation.ipynb",                       "11.0"),
    ("source/vK.11.0 Image Detection and Localisation [Pretrained ResNet34].ipynb", "11.0"),
    ("source/vK.11.1 Image Detection and Localisation.ipynb",                       "11.1"),
    ("source/vK.11.2 Image Detection and Localisation.ipynb",                       "11.2"),
    ("source/vK.11.3 Image Detection and Localisation.ipynb",                       "11.3"),
    ("source/vK.11.4 Image Detection and Localisation.ipynb",                       "11.4"),
    ("source/vK.11.5 Image Detection and Localisation.ipynb",                       "11.5"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  Saved: {path.name} ({os.path.getsize(path):,} bytes)")


def to_source_lines(text):
    """Convert a string to Jupyter source-line list format."""
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def get_source(cell):
    """Get cell source as single string."""
    return "".join(cell.get("source", []))


def set_source(cell, text):
    """Set cell source from string."""
    cell["source"] = to_source_lines(text)


def find_cell(cells, *patterns):
    """Find first code cell whose source contains ALL given patterns."""
    for i, c in enumerate(cells):
        src = get_source(c)
        if all(p in src for p in patterns):
            return i
    return None


def replace_in_source(cells, idx, old, new):
    """Replace old→new in cell source. Returns True if replacement happened."""
    src = get_source(cells[idx])
    if old not in src:
        return False
    src = src.replace(old, new, 1)
    set_source(cells[idx], src)
    return True


def append_to_source(cells, idx, code):
    """Append code to end of cell source."""
    src = get_source(cells[idx]).rstrip("\n")
    src = src + "\n\n" + code
    set_source(cells[idx], src)
    return True


# ---------------------------------------------------------------------------
# PATCH 1: W&B Init — fix version strings + add runtime config
# ---------------------------------------------------------------------------

def patch_wandb_init(cells, version):
    idx = find_cell(cells, 'project="vK.11.0-tampered-image-detection-assignment"')
    if idx is None:
        # Check if already patched (different version)
        idx2 = find_cell(cells, f'project="vK.{version}-tampered-image-detection-assignment"')
        if idx2 is not None:
            print(f"    PATCH 1 (wandb init): already patched")
            return 0
        print(f"    PATCH 1 (wandb init): WARNING — anchor not found")
        return 0

    # For vK.11.0 the version strings match so check config.update instead
    src = get_source(cells[idx])
    if "wandb.config.update" in src:
        print(f"    PATCH 1 (wandb init): already patched")
        return 0

    changes = 0
    # 1. Project name
    if replace_in_source(cells, idx,
        'project="vK.11.0-tampered-image-detection-assignment"',
        f'project="vK.{version}-tampered-image-detection-assignment"'):
        changes += 1
    # 2. Run name
    if replace_in_source(cells, idx,
        'vK.11.0-smp-resnet34-ela-seed',
        f'vK.{version}-smp-resnet34-ela-seed'):
        changes += 1
    # 3. Tags
    if replace_in_source(cells, idx,
        'tags=["vk11.0",',
        f'tags=["vk{version}",'):
        changes += 1
    # 4. Offline name
    if replace_in_source(cells, idx,
        'name="vk11.0-offline"',
        f'name="vk{version}-offline"'):
        changes += 1

    # 5. Append runtime config update
    anchor = 'print(f"W&B active: {WANDB_ACTIVE}")'
    config_code = f'''print(f"W&B active: {{WANDB_ACTIVE}}")

if WANDB_ACTIVE:
    wandb.config.update({{
        'notebook_version': 'vK.{version}',
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'num_params': sum(p.numel() for p in get_base_model(model).parameters()),
    }}, allow_val_change=True)'''
    if replace_in_source(cells, idx, anchor, config_code):
        changes += 1

    print(f"    PATCH 1 (wandb init): {changes} replacements")
    return 1 if changes > 0 else 0


# ---------------------------------------------------------------------------
# PATCH 2: Training Loop — add prediction viz every 5 epochs
# ---------------------------------------------------------------------------

PREDICTION_VIZ_CODE = """\

        # --- W&B: log sample predictions every 5 epochs ---
        if epoch % 5 == 0 or epoch == start_epoch:
            try:
                import matplotlib.pyplot as _plt_v
                import numpy as _np_v
                _viz_images = []
                _v_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                _v_std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                with torch.no_grad():
                    for _vb_imgs, _vb_masks, _vb_labels in val_loader:
                        for _vi in range(_vb_imgs.size(0)):
                            if _vb_labels[_vi].item() == 1 and len(_viz_images) < 2:
                                _inp = _vb_imgs[_vi:_vi+1].to(device)
                                with autocast('cuda', enabled=CONFIG['use_amp']):
                                    _, _seg_out = model(_inp)
                                _pred = torch.sigmoid(_seg_out).cpu().squeeze().numpy()
                                _rgb = (_vb_imgs[_vi, :3] * _v_std + _v_mean).clamp(0, 1).permute(1, 2, 0).numpy()
                                _gt = _vb_masks[_vi].squeeze().numpy()
                                _ov = _rgb.copy()
                                _ov[_pred > 0.5] = _ov[_pred > 0.5] * 0.5 + _np_v.array([1., 0., 0.]) * 0.5
                                _fig_v, _ax_v = _plt_v.subplots(1, 4, figsize=(16, 4))
                                _ax_v[0].imshow(_rgb); _ax_v[0].set_title('Original'); _ax_v[0].axis('off')
                                _ax_v[1].imshow(_gt, cmap='gray'); _ax_v[1].set_title('GT Mask'); _ax_v[1].axis('off')
                                _ax_v[2].imshow(_pred, cmap='hot'); _ax_v[2].set_title('Predicted'); _ax_v[2].axis('off')
                                _ax_v[3].imshow(_ov); _ax_v[3].set_title('Overlay'); _ax_v[3].axis('off')
                                _fig_v.suptitle(f'Epoch {epoch} - Sample {len(_viz_images)+1}', fontsize=12)
                                _plt_v.tight_layout()
                                _viz_images.append(wandb.Image(_fig_v))
                                _plt_v.close(_fig_v)
                        if len(_viz_images) >= 2:
                            break
                if _viz_images:
                    wandb.log({'val_predictions': _viz_images}, commit=False)
            except Exception:
                pass"""


def patch_training_loop(cells):
    idx = find_cell(cells, '# ================== Main training loop ==================')
    if idx is None:
        print("    PATCH 2 (training loop viz): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])

    # Check already patched
    if 'val_predictions' in src:
        print("    PATCH 2 (training loop viz): already patched")
        return 0

    anchor = "'lr/decoder': optimizer.param_groups[1]['lr'],\n        })\n\n    # Build checkpoint"
    if anchor not in src:
        print("    PATCH 2 (training loop viz): WARNING — anchor not found")
        return 0

    new_text = "'lr/decoder': optimizer.param_groups[1]['lr'],\n        })" + PREDICTION_VIZ_CODE + "\n\n    # Build checkpoint"
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 2 (training loop viz): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 3: Test Eval — add artifact + table logging
# ---------------------------------------------------------------------------

ARTIFACT_CODE = """\

    # --- W&B: log model artifact ---
    try:
        _ckpt = os.path.join(str(KAGGLE_WORKING_DIR), 'best_model.pth')
        if os.path.exists(_ckpt):
            _art = wandb.Artifact('best-model', type='model',
                description=f'Best checkpoint from epoch {best_epoch}')
            _art.add_file(_ckpt)
            WANDB_RUN.log_artifact(_art)
            print('  W&B: model artifact logged')
    except Exception as _e:
        print(f'  W&B artifact skip: {_e}')

    # --- W&B: save training history CSV ---
    try:
        _hist = os.path.join(str(RESULTS_DIR), 'training_history.csv')
        if os.path.exists(str(_hist)):
            wandb.save(str(_hist))
            print('  W&B: training_history.csv saved')
    except Exception as _e:
        print(f'  W&B save skip: {_e}')

    # --- W&B: log final metrics table ---
    try:
        _tbl = wandb.Table(
            columns=['Metric', 'Value'],
            data=[
                ['accuracy', test_metrics['acc']],
                ['dice_all', test_metrics['dice']],
                ['tampered_dice', test_metrics['tampered_dice']],
                ['tampered_iou', test_metrics['tampered_iou']],
                ['tampered_f1', test_metrics['tampered_f1']],
                ['roc_auc', test_metrics['roc_auc']],
                ['best_epoch', float(best_epoch)],
            ]
        )
        wandb.log({'final_test_metrics_table': _tbl})
        print('  W&B: final metrics table logged')
    except Exception as _e:
        print(f'  W&B table skip: {_e}')"""


def patch_test_eval(cells):
    idx = find_cell(cells, 'FINAL_TEST_METRICS = test_metrics')
    if idx is None:
        print("    PATCH 3 (test eval artifacts): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])
    if 'wandb.Artifact' in src:
        print("    PATCH 3 (test eval artifacts): already patched")
        return 0

    anchor = "        'test/roc_auc': test_metrics['roc_auc'],\n    })"
    if anchor not in src:
        print("    PATCH 3 (test eval artifacts): WARNING — anchor not found")
        return 0

    src = src.replace(anchor, anchor + ARTIFACT_CODE, 1)
    set_source(cells[idx], src)
    print("    PATCH 3 (test eval artifacts): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 4: Threshold Sweep
# ---------------------------------------------------------------------------

def patch_threshold_sweep(cells):
    idx = find_cell(cells, 'Threshold Sweep on Validation Set')
    if idx is None:
        print("    PATCH 4 (threshold sweep): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src and "threshold_sweep" in src:
        print("    PATCH 4 (threshold sweep): already patched")
        return 0

    anchor = "plt.show()\n\ndef compute_metrics_at_threshold"
    if anchor not in src:
        print("    PATCH 4 (threshold sweep): WARNING — anchor not found")
        return 0

    new_text = """plt.show()

if WANDB_ACTIVE:
    wandb.log({'evaluation/threshold_sweep': wandb.Image(fig)})
    wandb.summary.update({'optimal_threshold': OPTIMAL_THRESHOLD})

def compute_metrics_at_threshold"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 4 (threshold sweep): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 5: Confusion Matrix + ROC/PR
# ---------------------------------------------------------------------------

def patch_confusion_matrix(cells):
    idx = find_cell(cells, 'Confusion Matrix', 'roc_curve')
    if idx is None:
        print("    PATCH 5 (confusion matrix): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src:
        print("    PATCH 5 (confusion matrix): already patched")
        return 0

    anchor = "plt.tight_layout()\nplt.show()"
    if anchor not in src:
        print("    PATCH 5 (confusion matrix): WARNING — anchor not found")
        return 0

    new_text = """plt.tight_layout()
plt.show()

if WANDB_ACTIVE:
    wandb.log({'evaluation/confusion_matrix_roc_pr': wandb.Image(fig)})"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 5 (confusion matrix): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 6: Mask-Size Stratified
# ---------------------------------------------------------------------------

def patch_mask_size(cells):
    # Find the CODE cell (not the markdown header)
    for i, c in enumerate(cells):
        src = get_source(c)
        if c.get("cell_type") == "code" and "Mask-Size Stratified" in src:
            idx = i
            break
    else:
        print("    PATCH 6 (mask-size): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src:
        print("    PATCH 6 (mask-size): already patched")
        return 0

    # This plt.show() is inside the `if bucket_names:` block (4-space indent)
    anchor = "    plt.tight_layout()\n    plt.show()"
    if anchor not in src:
        print("    PATCH 6 (mask-size): WARNING — anchor not found")
        return 0

    new_text = """    plt.tight_layout()
    plt.show()

    if WANDB_ACTIVE:
        wandb.log({'evaluation/mask_size_f1': wandb.Image(fig)})"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 6 (mask-size): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 7: ELA Visualization
# ---------------------------------------------------------------------------

def patch_ela_viz(cells):
    idx = find_cell(cells, 'show_ela_visualization', 'ELA')
    if idx is None:
        print("    PATCH 7 (ELA viz): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src:
        print("    PATCH 7 (ELA viz): already patched")
        return 0

    anchor = "    plt.show()\n\nshow_ela_visualization(test_loader, num_samples=3)"
    if anchor not in src:
        print("    PATCH 7 (ELA viz): WARNING — anchor not found")
        return 0

    new_text = """    plt.show()

    if WANDB_ACTIVE:
        wandb.log({'evaluation/ela_visualization': wandb.Image(fig)})

show_ela_visualization(test_loader, num_samples=3)"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 7 (ELA viz): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 8: Failure Cases
# ---------------------------------------------------------------------------

def patch_failure_cases(cells):
    idx = find_cell(cells, 'Failure Case Analysis', 'worst_order')
    if idx is None:
        print("    PATCH 8 (failure cases): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src:
        print("    PATCH 8 (failure cases): already patched")
        return 0

    # Cell ends with plt.tight_layout() / plt.show()
    anchor = "plt.tight_layout()\nplt.show()"
    if anchor not in src:
        print("    PATCH 8 (failure cases): WARNING — anchor not found")
        return 0

    new_text = """plt.tight_layout()
plt.show()

if WANDB_ACTIVE:
    wandb.log({'evaluation/failure_cases': wandb.Image(fig)})"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 8 (failure cases): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 9: Grad-CAM
# ---------------------------------------------------------------------------

def patch_gradcam(cells):
    idx = find_cell(cells, 'Grad-CAM Explainability')
    if idx is None:
        print("    PATCH 9 (Grad-CAM): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src:
        print("    PATCH 9 (Grad-CAM): already patched")
        return 0

    anchor = "plt.show()\n\ngrad_cam.remove_hooks()"
    if anchor not in src:
        print("    PATCH 9 (Grad-CAM): WARNING — anchor not found")
        return 0

    new_text = """plt.show()

if WANDB_ACTIVE:
    wandb.log({'evaluation/gradcam': wandb.Image(fig)})

grad_cam.remove_hooks()"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 9 (Grad-CAM): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 10: Robustness
# ---------------------------------------------------------------------------

def patch_robustness(cells):
    idx = find_cell(cells, 'Robustness Testing Suite')
    if idx is None:
        print("    PATCH 10 (robustness): WARNING — cell not found")
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src and "robustness" in src:
        print("    PATCH 10 (robustness): already patched")
        return 0

    anchor = "plt.savefig(os.path.join(PLOTS_DIR, 'robustness_results.png'), dpi=150, bbox_inches='tight')\nplt.show()"
    if anchor not in src:
        print("    PATCH 10 (robustness): WARNING — anchor not found")
        return 0

    new_text = """plt.savefig(os.path.join(PLOTS_DIR, 'robustness_results.png'), dpi=150, bbox_inches='tight')
plt.show()

if WANDB_ACTIVE:
    wandb.log({'evaluation/robustness': wandb.Image(fig)})
    try:
        _rob_table = wandb.Table(
            columns=['Condition', 'F1', 'Delta_from_Clean'],
            data=[[n, f, f - clean_f1] for n, f in robustness_results.items()]
        )
        wandb.log({'evaluation/robustness_table': _rob_table})
    except Exception:
        pass"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 10 (robustness): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 11: Dashboard Training Curves (vK.11.5 only)
# ---------------------------------------------------------------------------

def patch_dashboard_training_curves(cells):
    idx = find_cell(cells, 'Results Dashboard: Training Curves')
    if idx is None:
        # Not vK.11.5 — silently skip
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src:
        print("    PATCH 11 (dashboard curves): already patched")
        return 0

    anchor = "        plt.show()\n    else:"
    if anchor not in src:
        print("    PATCH 11 (dashboard curves): WARNING — anchor not found")
        return 0

    new_text = """        plt.show()

        if WANDB_ACTIVE:
            wandb.log({'dashboard/training_curves': wandb.Image(fig)})
    else:"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 11 (dashboard curves): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 12: Dashboard Localization (vK.11.5 only)
# ---------------------------------------------------------------------------

def patch_dashboard_localization(cells):
    idx = find_cell(cells, 'Results Dashboard: Example Localization')
    if idx is None:
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src:
        print("    PATCH 12 (dashboard localization): already patched")
        return 0

    anchor = "    plt.show()\n\n    # Clean up temporary variables"
    if anchor not in src:
        print("    PATCH 12 (dashboard localization): WARNING — anchor not found")
        return 0

    new_text = """    plt.show()

    if WANDB_ACTIVE:
        wandb.log({'dashboard/localization_example': wandb.Image(fig)})

    # Clean up temporary variables"""
    src = src.replace(anchor, new_text, 1)
    set_source(cells[idx], src)
    print("    PATCH 12 (dashboard localization): applied")
    return 1


# ---------------------------------------------------------------------------
# PATCH 13: Dashboard Metrics (vK.11.5 only)
# ---------------------------------------------------------------------------

def patch_dashboard_metrics(cells):
    idx = find_cell(cells, 'Results Dashboard: Metrics Summary')
    if idx is None:
        return 0

    src = get_source(cells[idx])
    if "wandb.log" in src:
        print("    PATCH 13 (dashboard metrics): already patched")
        return 0

    metrics_sync = """

# --- W&B: sync dashboard metrics ---
try:
    if WANDB_ACTIVE and 'FINAL_TEST_METRICS' in dir():
        _m = FINAL_TEST_METRICS
        wandb.log({'dashboard/metrics_table': wandb.Table(
            columns=['Metric', 'Value'],
            data=[
                ['accuracy', _m.get('acc', 0)],
                ['roc_auc', _m.get('roc_auc', 0)],
                ['tampered_dice', _m.get('tampered_dice', 0)],
                ['tampered_iou', _m.get('tampered_iou', 0)],
                ['tampered_f1', _m.get('tampered_f1', 0)],
            ]
        )})
except Exception:
    pass"""

    append_to_source(cells, idx, metrics_sync.lstrip("\n"))
    print("    PATCH 13 (dashboard metrics): applied")
    return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("W&B Logging Fix — vK.11.x Notebook Series")
    print("=" * 60)

    total_patched = 0

    for filename, version in NOTEBOOKS:
        path = NOTEBOOKS_DIR / filename
        if not path.exists():
            print(f"\nSKIP (not found): {filename}")
            continue

        print(f"\n--- {filename} (v{version}) ---")
        nb = load_notebook(path)
        cells = nb["cells"]
        cell_count_before = len(cells)

        changes = 0
        changes += patch_wandb_init(cells, version)
        changes += patch_training_loop(cells)
        changes += patch_test_eval(cells)
        changes += patch_threshold_sweep(cells)
        changes += patch_confusion_matrix(cells)
        changes += patch_mask_size(cells)
        changes += patch_ela_viz(cells)
        changes += patch_failure_cases(cells)
        changes += patch_gradcam(cells)
        changes += patch_robustness(cells)
        changes += patch_dashboard_training_curves(cells)
        changes += patch_dashboard_localization(cells)
        changes += patch_dashboard_metrics(cells)

        assert len(cells) == cell_count_before, \
            f"Cell count changed! {cell_count_before} -> {len(cells)}"

        if changes > 0:
            save_notebook(nb, path)
            total_patched += 1
            print(f"  Total patches applied: {changes}")
        else:
            print(f"  No changes needed")

    print(f"\n{'=' * 60}")
    print(f"Done. {total_patched}/{len(NOTEBOOKS)} notebooks patched.")


if __name__ == "__main__":
    main()
