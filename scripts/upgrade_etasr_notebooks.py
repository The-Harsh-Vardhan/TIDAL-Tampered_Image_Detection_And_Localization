#!/usr/bin/env python3
"""
Upgrade ETASR notebooks with infrastructure improvements.

Adds to all 11 notebooks in Notebooks vR.x/:
  - W&B experiment tracking (auth, init, per-epoch logging, final logging)
  - Mixed precision (mixed_float16 policy)
  - Gradient clipping (clipnorm=1.0)
  - Model checkpointing (best + latest)
  - Resume training from checkpoint
  - Standardized evaluation dashboard (2x2 grid)
  - Precision-Recall curve
  - 20-sample prediction visualization
  - GPU cleanup (clear_session + gc.collect)
  - wandb.finish()

Preserves: model architectures, ablation variables, preprocessing, splits.
"""

import json
import os
import re
import copy

NB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Notebooks vR.x')

# ============================================================
# Per-notebook configuration
# ============================================================
NOTEBOOK_CONFIGS = {
    'vR.ETASR Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr_etasr', 'run_id': 'run01', 'version': 'vR.ETASR',
        'has_test_set': False,
    },
    'vR.0 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr0', 'run_id': 'run01', 'version': 'vR.0',
        'has_test_set': True,
    },
    'vR.1 ETASR Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1', 'run_id': 'run01', 'version': 'vR.1',
        'has_test_set': False,
    },
    'vR.1.1 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1_1', 'run_id': 'run01', 'version': 'vR.1.1',
        'has_test_set': True,
    },
    'vR.1.2 \u2014 ETASR Run-01 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1_2', 'run_id': 'run01', 'version': 'vR.1.2',
        'has_test_set': True,
    },
    'vR.1.3 \u2014 ETASR Run-02 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1_3', 'run_id': 'run02', 'version': 'vR.1.3',
        'has_test_set': True,
    },
    'vR.1.4 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1_4', 'run_id': 'run01', 'version': 'vR.1.4',
        'has_test_set': True,
    },
    'vR.1.4 ETASR Run-01 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1_4', 'run_id': 'run01a', 'version': 'vR.1.4R',
        'has_test_set': True,
    },
    'vR.1.5 ETASR Run-01 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1_5', 'run_id': 'run01', 'version': 'vR.1.5',
        'has_test_set': True,
    },
    'vR.1.6 ETASR Run-01 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1_6', 'run_id': 'run01', 'version': 'vR.1.6',
        'has_test_set': True,
    },
    'vR.1.7 ETASR Run-01 Image Detection and Localisation.ipynb': {
        'experiment_id': 'vr1_7', 'run_id': 'run01', 'version': 'vR.1.7',
        'has_test_set': True,
    },
}

# ============================================================
# Helper functions
# ============================================================

def normalize_sources(cells):
    """Ensure all cell['source'] are lists of strings (some notebooks store as string)."""
    for cell in cells:
        if isinstance(cell.get('source'), str):
            cell['source'] = to_source_lines(cell['source'])
        elif cell.get('source') is None:
            cell['source'] = []


def src(cell):
    """Get cell source as a single string."""
    s = cell['source']
    if isinstance(s, str):
        return s
    return ''.join(s)


def find_cell_idx(cells, pattern, start=0):
    """Find first code cell whose source contains pattern, from start index."""
    for i in range(start, len(cells)):
        if cells[i]['cell_type'] == 'code' and pattern in src(cells[i]):
            return i
    return None


def to_source_lines(text):
    """Convert a multi-line string to notebook source list format."""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            if line:  # don't add empty trailing line
                result.append(line)
    return result


def make_code_cell(source_text):
    """Create a new code cell from a source string."""
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': to_source_lines(source_text),
    }


def make_md_cell(source_text):
    """Create a new markdown cell."""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': to_source_lines(source_text),
    }


def append_to_cell(cell, text):
    """Append text to a cell's source."""
    lines = cell['source']
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += '\n'
    lines.extend(to_source_lines(text))


def replace_in_cell(cell, old, new):
    """Replace text in a cell's source (join, replace, split back)."""
    full = ''.join(cell['source'])
    if old not in full:
        return False
    full = full.replace(old, new)
    cell['source'] = to_source_lines(full)
    return True


# ============================================================
# Template: W&B + infrastructure block (appended to imports cell)
# ============================================================

def wandb_init_block(config):
    return f'''
# ---- W&B Experiment Tracking ----
!pip install -q wandb
import wandb
from wandb.integration.keras import WandbCallback
import gc

WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_USERNAME = os.environ.get("WANDB_USERNAME", "")
USE_WANDB = bool(WANDB_API_KEY)

if USE_WANDB:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        entity=WANDB_USERNAME,
        project="tamper-detection-ablation",
        name="{config['experiment_id']}_{config['run_id']}",
        config={{
            "experiment_id": "{config['experiment_id']}",
            "run_id": "{config['run_id']}",
            "dataset": "CASIA2",
            "notebook": "{config['version']}",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "ela_quality": ELA_QUALITY,
            "seed": SEED,
        }}
    )
    print(f'W&B run: {{wandb.run.name}}')
else:
    print('W&B disabled (no WANDB_API_KEY found)')

# ---- Mixed Precision ----
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print(f'Mixed precision: {{tf.keras.mixed_precision.global_policy().name}}')

# ---- Checkpoint Paths ----
CHECKPOINT_DIR = '/kaggle/working/checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'{{VERSION}}_best.keras')
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, f'{{VERSION}}_latest.keras')
RESUME = True'''


# ============================================================
# Template: Checkpoint + Resume + Callbacks (injected before model.fit)
# ============================================================

CHECKPOINT_RESUME_BLOCK = '''
# ---- Checkpointing Callbacks ----
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1
)
checkpoint_latest = tf.keras.callbacks.ModelCheckpoint(
    LATEST_CHECKPOINT, save_best_only=False, verbose=0
)

# ---- Resume from Checkpoint ----
start_epoch = 0
if RESUME and os.path.exists(LATEST_CHECKPOINT):
    print(f'Resuming from checkpoint: {LATEST_CHECKPOINT}')
    model = tf.keras.models.load_model(LATEST_CHECKPOINT)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    # Estimate start_epoch from checkpoint modification time
    start_epoch = 0  # Will train from beginning but model weights are warm
    print('  Model weights restored. Continuing training.')
else:
    print('No checkpoint found — starting fresh.')

'''


# ============================================================
# Template: Standardized evaluation dashboard
# ============================================================

def eval_dashboard_cell(config):
    if config['has_test_set']:
        data_line = '_X_eval, _Y_eval_int, _eval_name = X_test, Y_test_int, "Test"'
    else:
        data_line = '_X_eval, _Y_eval_int, _eval_name = X_val, Y_val_int, "Validation"'

    return f'''# ============================================================
# [INFRA] Standardized Evaluation Dashboard
# ============================================================
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

{data_line}

_probs = model.predict(_X_eval, verbose=0)
_preds = np.argmax(_probs, axis=1)
_true = _Y_eval_int

_accuracy = accuracy_score(_true, _preds)
_prec_per, _rec_per, _f1_per, _ = precision_recall_fscore_support(_true, _preds, average=None)
_macro_f1 = precision_recall_fscore_support(_true, _preds, average='macro')[2]
_fpr, _tpr, _ = roc_curve(_true, _probs[:, 1])
_roc_auc = auc(_fpr, _tpr)
_cm = confusion_matrix(_true, _preds)
_pr_prec, _pr_rec, _ = precision_recall_curve(_true, _probs[:, 1])

print('=' * 70)
print(f'  STANDARDIZED EVALUATION — {{_eval_name}} Set')
print('=' * 70)
print(f'  Accuracy:  {{_accuracy:.4f}}  ({{_accuracy*100:.2f}}%)')
print(f'  Macro F1:  {{_macro_f1:.4f}}')
print(f'  ROC-AUC:   {{_roc_auc:.4f}}')
print(f'  Per-class: Precision  Recall  F1')
print(f'  Authentic: {{_prec_per[0]:.4f}}    {{_rec_per[0]:.4f}}  {{_f1_per[0]:.4f}}')
print(f'  Tampered:  {{_prec_per[1]:.4f}}    {{_rec_per[1]:.4f}}  {{_f1_per[1]:.4f}}')
print('=' * 70)

# ---- 2x2 Dashboard ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
axes[0, 0].plot(history.history['loss'], label='Train', color='#1f77b4', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val', color='#ff7f0e', linewidth=2)
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss', fontweight='bold'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# Accuracy curves
axes[0, 1].plot(history.history['accuracy'], label='Train', color='#2ca02c', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Val', color='#d62728', linewidth=2)
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy', fontweight='bold'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# ROC curve
axes[1, 0].plot(_fpr, _tpr, color='#1f77b4', linewidth=2, label=f'AUC={{_roc_auc:.4f}}')
axes[1, 0].plot([0, 1], [0, 1], '--', color='grey', linewidth=1)
axes[1, 0].set_xlabel('FPR'); axes[1, 0].set_ylabel('TPR')
axes[1, 0].set_title('ROC Curve', fontweight='bold'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
            xticklabels=['Au', 'Tp'], yticklabels=['Au', 'Tp'])
axes[1, 1].set_xlabel('Predicted'); axes[1, 1].set_ylabel('True')
axes[1, 1].set_title('Confusion Matrix', fontweight='bold')

plt.suptitle(f'{config["version"]} — Evaluation Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()
if USE_WANDB:
    wandb.log({{"evaluation_dashboard": wandb.Image(fig)}})
plt.show()'''


# ============================================================
# Template: Precision-Recall Curve
# ============================================================

PR_CURVE_CELL = '''# ============================================================
# [INFRA] Precision-Recall Curve
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(_pr_rec, _pr_prec, color='#2ca02c', linewidth=2)
ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
ax.set_title(f'Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
plt.tight_layout()
if USE_WANDB:
    wandb.log({"pr_curve": wandb.Image(fig)})
plt.show()'''


# ============================================================
# Template: 20-sample prediction visualization
# ============================================================

def prediction_viz_cell(config):
    if config['has_test_set']:
        x_var = 'X_test'
    else:
        x_var = 'X_val'

    return f'''# ============================================================
# [INFRA] Prediction Visualization (20 samples)
# ============================================================
class_names = ['Authentic', 'Tampered']
tp_idx = np.where(_true == 1)[0][:14]
au_idx = np.where(_true == 0)[0][:6]
_sample_idx = np.concatenate([tp_idx, au_idx])[:20]

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
for i, idx in enumerate(_sample_idx):
    ax = axes[i // 5, i % 5]
    ax.imshow({x_var}[idx])
    true_label = class_names[_true[idx]]
    pred_label = class_names[_preds[idx]]
    confidence = _probs[idx].max()
    correct = _true[idx] == _preds[idx]
    color = 'green' if correct else 'red'
    ax.set_title(f'True: {{true_label}}\\nPred: {{pred_label}} ({{confidence:.2f}})', color=color, fontsize=9)
    ax.axis('off')
# Hide unused subplots
for i in range(len(_sample_idx), 20):
    axes[i // 5, i % 5].axis('off')
plt.suptitle(f'{config["version"]} — Prediction Examples (20 samples)', fontsize=14, fontweight='bold')
plt.tight_layout()
if USE_WANDB:
    wandb.log({{"prediction_examples": wandb.Image(fig)}})
plt.show()'''


# ============================================================
# Template: Final W&B logging + GPU cleanup
# ============================================================

FINAL_LOGGING_CELL = '''# ============================================================
# [INFRA] Final W&B Logging + GPU Cleanup
# ============================================================
if USE_WANDB:
    wandb.log({
        "Final_Accuracy": float(_accuracy),
        "Final_MacroF1": float(_macro_f1),
        "Final_ROC_AUC": float(_roc_auc),
        "Final_Precision_Au": float(_prec_per[0]),
        "Final_Precision_Tp": float(_prec_per[1]),
        "Final_Recall_Au": float(_rec_per[0]),
        "Final_Recall_Tp": float(_rec_per[1]),
        "Final_F1_Au": float(_f1_per[0]),
        "Final_F1_Tp": float(_f1_per[1]),
    })
    # Save best model as W&B artifact
    if os.path.exists(BEST_MODEL_PATH):
        artifact = wandb.Artifact(f'{VERSION}_model', type='model')
        artifact.add_file(BEST_MODEL_PATH)
        wandb.log_artifact(artifact)
    wandb.finish()
    print('W&B run finished.')

# GPU cleanup
tf.keras.backend.clear_session()
gc.collect()
print('GPU memory released.')'''


# ============================================================
# Main upgrade function
# ============================================================

def upgrade_notebook(filepath, config):
    """Apply all infrastructure upgrades to a single notebook."""
    filename = os.path.basename(filepath)
    print(f'\n{"="*70}')
    print(f'Upgrading: {filename}')
    print(f'  experiment_id={config["experiment_id"]}, run_id={config["run_id"]}')
    print(f'{"="*70}')

    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    normalize_sources(cells)

    # Save original model build cell for validation
    build_idx = find_cell_idx(cells, 'def build_model')
    orig_build_src = src(cells[build_idx]) if build_idx is not None else None

    # ---- Find key cell indices ----
    imports_idx = find_cell_idx(cells, 'import tensorflow')
    compile_idx = find_cell_idx(cells, 'model.compile(')
    fit_idx = find_cell_idx(cells, 'model.fit(')
    save_idx = find_cell_idx(cells, 'model.save(')

    if imports_idx is None:
        print(f'  ERROR: Could not find imports cell. Skipping.')
        return False
    if compile_idx is None:
        print(f'  ERROR: Could not find compile cell. Skipping.')
        return False
    if fit_idx is None:
        print(f'  ERROR: Could not find fit cell. Skipping.')
        return False

    print(f'  imports_idx={imports_idx}, build_idx={build_idx}, '
          f'compile_idx={compile_idx}, fit_idx={fit_idx}, save_idx={save_idx}')

    # ================================================================
    # PHASE 1: Modify existing cells (does not change cell count)
    # ================================================================

    # 1a. Ensure roc_curve and auc are imported (some early notebooks lack them)
    imports_src = src(cells[imports_idx])
    if 'roc_curve' not in imports_src:
        replace_in_cell(cells[imports_idx],
            'from sklearn.metrics import (',
            'from sklearn.metrics import (\n    roc_curve, auc, precision_recall_curve,')
        # If single-line import style, try alternate
        if 'roc_curve' not in src(cells[imports_idx]):
            # Add as separate import
            append_to_cell(cells[imports_idx],
                '\nfrom sklearn.metrics import roc_curve, auc, precision_recall_curve\n')
        print('  + Added roc_curve/auc/precision_recall_curve imports')

    # 1b. Ensure VERSION is defined (vR.ETASR doesn't have it)
    if 'VERSION' not in imports_src:
        append_to_cell(cells[imports_idx],
            f"\nVERSION = '{config['version']}'\n")
        print(f'  + Added VERSION constant')

    # 1c. Append W&B + mixed precision + checkpoint paths to imports cell
    append_to_cell(cells[imports_idx], wandb_init_block(config))
    print('  + Appended W&B init + mixed precision + checkpoint paths')

    # 1d. Add dtype='float32' to final Dense layer for mixed precision stability
    if build_idx is not None:
        replaced = replace_in_cell(cells[build_idx],
            "Dense(2, activation='softmax')",
            "Dense(2, activation='softmax', dtype='float32')  # float32 for mixed precision")
        if replaced:
            print("  + Added dtype='float32' to output Dense layer")

    # 1e. Add clipnorm=1.0 to optimizer in compile cell
    compile_src = src(cells[compile_idx])
    if 'clipnorm' not in compile_src:
        replaced = replace_in_cell(cells[compile_idx],
            'Adam(learning_rate=LEARNING_RATE)',
            'Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)')
        if replaced:
            print('  + Added clipnorm=1.0 to Adam optimizer')
        else:
            print('  WARNING: Could not find Adam() call to add clipnorm')

    # 1f. Modify model.fit() cell: add checkpoint/resume logic and callbacks
    fit_cell = cells[fit_idx]
    fit_src = src(fit_cell)

    # Inject checkpoint + resume block BEFORE model.fit()
    # Find the line with 'history = model.fit(' and insert before it
    fit_lines = fit_cell['source']
    insert_pos = None
    for li, line in enumerate(fit_lines):
        if 'history = model.fit(' in line or 'model.fit(' in line:
            insert_pos = li
            break

    if insert_pos is not None:
        checkpoint_lines = to_source_lines(CHECKPOINT_RESUME_BLOCK)
        fit_cell['source'] = fit_lines[:insert_pos] + checkpoint_lines + fit_lines[insert_pos:]
        print('  + Injected checkpoint/resume block before model.fit()')

    # Add checkpoint callbacks + WandbCallback to the callbacks= argument
    fit_src_new = src(fit_cell)

    # Find existing callbacks pattern: callbacks=[early_stopping] or callbacks=[early_stopping, lr_scheduler]
    cb_match = re.search(r'callbacks=\[([^\]]+)\]', fit_src_new)
    if cb_match:
        existing_callbacks = cb_match.group(1).strip()
        new_callbacks = (f'{existing_callbacks}, checkpoint_best, checkpoint_latest'
                         f'] + ([WandbCallback(save_model=False)] if USE_WANDB else [])')
        old_str = f'callbacks=[{cb_match.group(1)}]'
        new_str = f'callbacks=[{new_callbacks}'
        replace_in_cell(fit_cell, old_str, new_str)
        print('  + Added checkpoint + WandB callbacks to model.fit()')

    # Add initial_epoch to model.fit() ONLY (not to callbacks)
    fit_src_final = src(fit_cell)
    if 'initial_epoch' not in fit_src_final:
        # Find model.fit( and its matching closing )
        fit_call_start = fit_src_final.find('model.fit(')
        if fit_call_start >= 0:
            depth = 0
            closing_paren = -1
            for ci in range(fit_call_start, len(fit_src_final)):
                if fit_src_final[ci] == '(':
                    depth += 1
                elif fit_src_final[ci] == ')':
                    depth -= 1
                    if depth == 0:
                        closing_paren = ci
                        break
            if closing_paren > 0:
                new_fit_src = (fit_src_final[:closing_paren].rstrip()
                               + ',\n    initial_epoch=start_epoch\n'
                               + fit_src_final[closing_paren:])
                fit_cell['source'] = to_source_lines(new_fit_src)
                print('  + Added initial_epoch=start_epoch to model.fit()')

    # ================================================================
    # PHASE 2: Insert new cells (insert from back to front)
    # ================================================================

    # Find insertion point: after existing visualizations, before save cell
    # Use save_idx as anchor, or end of notebook if save_idx not found
    if save_idx is not None:
        # Re-find save_idx since earlier insertions to fit cell may have shifted it
        # (fit cell modifications don't change cell count, only cell content)
        insert_before = save_idx
    else:
        insert_before = len(cells)

    # Insert in reverse order so indices don't shift
    new_cells = [
        make_md_cell('---\n\n## [INFRA] Standardized Evaluation & Visualization'),
        make_code_cell(eval_dashboard_cell(config)),
        make_code_cell(PR_CURVE_CELL),
        make_code_cell(prediction_viz_cell(config)),
        make_code_cell(FINAL_LOGGING_CELL),
    ]

    for cell in reversed(new_cells):
        cells.insert(insert_before, cell)

    print(f'  + Inserted {len(new_cells)} new cells before save cell (idx={insert_before})')

    # ================================================================
    # PHASE 3: Validation
    # ================================================================

    # Re-find build cell and verify architecture is unchanged
    new_build_idx = find_cell_idx(cells, 'def build_model')
    if new_build_idx is not None and orig_build_src is not None:
        new_build_src = src(cells[new_build_idx])
        # Only allowed change: dtype='float32' addition
        check_src = new_build_src.replace(", dtype='float32')  # float32 for mixed precision", ")")
        if check_src.strip() != orig_build_src.strip():
            print('  WARNING: Model architecture may have changed!')
        else:
            print('  VALIDATED: Model architecture unchanged')

    # Verify key patterns exist in upgraded notebook
    all_src = '\n'.join(src(c) for c in cells if c['cell_type'] == 'code')
    checks = {
        'wandb.init': 'W&B init',
        'mixed_float16': 'Mixed Precision',
        'clipnorm=1.0': 'Gradient Clipping',
        'ModelCheckpoint': 'Checkpointing',
        'RESUME': 'Resume Training',
        'Standardized Evaluation': 'Eval Dashboard',
        'Precision-Recall Curve': 'PR Curve',
        'Prediction Visualization': 'Prediction Viz',
        'wandb.finish()': 'W&B Finish',
        'gc.collect()': 'GPU Cleanup',
    }
    for pattern, name in checks.items():
        status = 'OK' if pattern in all_src else 'MISSING'
        if status == 'MISSING':
            print(f'  WARNING: {name} ({pattern}) not found!')

    # ================================================================
    # Write upgraded notebook
    # ================================================================
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f'  DONE: {len(cells)} cells (was {len(cells) - len(new_cells)})')
    return True


# ============================================================
# Main
# ============================================================

def main():
    print('ETASR Notebook Infrastructure Upgrade')
    print(f'Directory: {NB_DIR}')
    print(f'Notebooks: {len(NOTEBOOK_CONFIGS)}')

    if not os.path.isdir(NB_DIR):
        print(f'ERROR: Directory not found: {NB_DIR}')
        return

    success = 0
    failed = 0
    for filename, config in NOTEBOOK_CONFIGS.items():
        filepath = os.path.join(NB_DIR, filename)
        if not os.path.exists(filepath):
            print(f'\nWARNING: File not found: {filepath}')
            failed += 1
            continue
        try:
            if upgrade_notebook(filepath, config):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f'\nERROR processing {filename}: {e}')
            import traceback
            traceback.print_exc()
            failed += 1

    print(f'\n{"="*70}')
    print(f'COMPLETE: {success} upgraded, {failed} failed')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
