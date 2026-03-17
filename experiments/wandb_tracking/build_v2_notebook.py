"""Build w-b-leaderboard-v2.ipynb programmatically."""
import json, os

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"),
            "outputs": [], "execution_count": None}

cells = []

# ── Cell 0: Header ──────────────────────────────────────────────────────────
cells.append(md("""# W&B Leaderboard v2 — Tamper Detection Ablation Study

Publication-quality experiment analysis notebook. Pulls **all runs** from Weights & Biases
and generates 8 charts + markdown tables ready for **GitHub README** and **LaTeX report**.

### Outputs (`figures/` folder)
| # | File | Description |
|---|------|-------------|
| 1 | `ablation_progression.png` | F1/IoU progression across experiments |
| 2 | `impact_hierarchy.png` | Lollipop chart of per-change impact (pp) |
| 3 | `top_experiments_bar.png` | Top-15 ranked horizontal bar chart |
| 4 | `radar_comparison.png` | Multi-metric radar: baseline vs best |
| 5 | `training_curves.png` | Loss & F1 curves for top-5 runs |
| 6 | `feature_set_comparison.png` | Grouped bar by input representation |
| 7 | `precision_recall_scatter.png` | Bubble scatter with iso-F1 curves |
| 8 | `metric_heatmap.png` | Correlation heatmap across all metrics |
| — | `leaderboard_table.md` | Markdown table for README |
| — | `best_model_summary.txt` | Text card for email / report |"""))

# ── Cell 1: Installs & Imports ──────────────────────────────────────────────
cells.append(md("## 1 — Install & Imports"))
cells.append(code("""!pip install -q wandb pandas tabulate matplotlib seaborn

import os, re, warnings
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from tabulate import tabulate
from IPython.display import display, HTML, Markdown

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180, 'savefig.bbox': 'tight',
                     'axes.spines.top': False, 'axes.spines.right': False})"""))

# ── Cell 2: Config & Auth ───────────────────────────────────────────────────
cells.append(md("## 2 — Config & Auth"))
cells.append(code("""# ── Project constants ──
WANDB_PROJECT = 'Tampered Image Detection & Localization'
WANDB_ENTITY  = 'tampered-image-detection-and-localization'
OUTPUT_DIR    = 'figures'
DPI           = 180

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Authenticate ──
WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
else:
    wandb.login()

api = wandb.Api()
print(f'Authenticated. Entity: {WANDB_ENTITY}')"""))

# ── Cell 3: Fetch All Runs ──────────────────────────────────────────────────
cells.append(md("## 3 — Fetch All Runs & Build DataFrame"))
cells.append(code("""runs = api.runs(f'{WANDB_ENTITY}/{WANDB_PROJECT}')
print(f'Found {len(runs)} total runs')

records = []
for run in runs:
    cfg = run.config
    s = run.summary._json_dict
    records.append({
        'experiment': cfg.get('experiment', run.name or ''),
        'version':    cfg.get('version', run.name or ''),
        'change':     cfg.get('change', ''),
        'run_tag':    cfg.get('run', ''),
        'wandb_run_id': run.id,
        'state':      run.state,
        'feature_set':   cfg.get('feature_set', ''),
        'encoder':       cfg.get('encoder', ''),
        'input_type':    cfg.get('input_type', cfg.get('feature_set', '')),
        'in_channels':   cfg.get('in_channels', ''),
        'epochs':        cfg.get('epochs', ''),
        'learning_rate': cfg.get('learning_rate', ''),
        'batch_size':    cfg.get('batch_size', ''),
        'img_size':      cfg.get('img_size', ''),
        # pixel metrics
        'pixel_f1':        s.get('pixel_f1'),
        'pixel_iou':       s.get('pixel_iou'),
        'pixel_auc':       s.get('pixel_auc'),
        'pixel_precision': s.get('pixel_precision'),
        'pixel_recall':    s.get('pixel_recall'),
        # image metrics
        'image_accuracy':  s.get('image_accuracy'),
        'image_macro_f1':  s.get('image_macro_f1'),
        'image_roc_auc':   s.get('image_roc_auc'),
        # training
        'best_val_f1':     s.get('val_pixel_f1'),
        'final_train_loss': s.get('train_loss'),
        'final_val_loss':  s.get('val_loss'),
        'duration_min':    s.get('_wandb', {}).get('runtime', 0) / 60,
    })

df = pd.DataFrame(records)
print(f'Runs with pixel_f1: {df["pixel_f1"].notna().sum()} / {len(df)}')

# ── Helper: version sort key ──
def version_sort_key(v):
    m = re.search(r'P\\.(\\d+(?:\\.\\d+)*)', str(v))
    if m:
        parts = m.group(1).split('.')
        return sum(int(p) / (100 ** i) for i, p in enumerate(parts))
    return 999

# ── Leaderboard (all runs, sorted by F1) ──
leaderboard = (
    df[df['pixel_f1'].notna()]
    .sort_values('pixel_f1', ascending=False)
    .reset_index(drop=True)
)
leaderboard.index += 1
leaderboard.index.name = 'Rank'

# ── De-duplicated: best run per version ──
dedup = (
    df[df['pixel_f1'].notna()].copy()
    .sort_values('pixel_f1', ascending=False)
    .drop_duplicates(subset='version', keep='first')
)
dedup['sort_key'] = dedup['version'].apply(version_sort_key)
dedup = dedup.sort_values('sort_key').reset_index(drop=True)
print(f'Unique versions: {len(dedup)}')
dedup[['version', 'change', 'pixel_f1', 'pixel_iou', 'feature_set']].head(10)"""))

# ── Cell 4: Leaderboard Tables ──────────────────────────────────────────────
cells.append(md("## 4 — Leaderboard Tables"))
cells.append(code("""lb_cols = ['version', 'change', 'feature_set', 'pixel_f1', 'pixel_iou',
           'pixel_auc', 'image_accuracy', 'run_tag', 'state']

print('=' * 90)
print('  PIXEL F1 LEADERBOARD — Top 10')
print('=' * 90)
print(tabulate(leaderboard[lb_cols].head(10),
               headers='keys', tablefmt='pipe', floatfmt='.4f', showindex=True))

print()

# Image Accuracy
img_lb = (
    df[df['image_accuracy'].notna()]
    .sort_values('image_accuracy', ascending=False)
    .reset_index(drop=True)
)
img_lb.index += 1
img_cols = ['version', 'change', 'image_accuracy', 'image_macro_f1', 'pixel_f1']
print('=' * 90)
print('  IMAGE ACCURACY LEADERBOARD — Top 10')
print('=' * 90)
print(tabulate(img_lb[img_cols].head(10),
               headers='keys', tablefmt='pipe', floatfmt='.4f', showindex=True))

print()

# Feature Set Comparison
valid = df[df['pixel_f1'].notna()]
if valid['feature_set'].nunique() > 1:
    feat = valid.groupby('feature_set').agg(
        F1_max=('pixel_f1', 'max'), F1_mean=('pixel_f1', 'mean'),
        IoU_max=('pixel_iou', 'max'), runs=('pixel_f1', 'count')
    ).round(4).sort_values('F1_max', ascending=False)
    print('=' * 90)
    print('  FEATURE SET COMPARISON')
    print('=' * 90)
    print(tabulate(feat, headers='keys', tablefmt='pipe', floatfmt='.4f', showindex=True))"""))

# ── Cell 5: Markdown Export ──────────────────────────────────────────────────
cells.append(md("## 5 — Export Markdown Table for README"))
cells.append(code("""md_cols = ['version', 'change', 'pixel_f1', 'pixel_iou', 'pixel_auc']
md_table = tabulate(
    leaderboard[md_cols].head(15),
    headers=['Rank', 'Version', 'Change', 'Pixel F1', 'IoU', 'AUC'],
    tablefmt='pipe', floatfmt='.4f', showindex=True
)

md_path = os.path.join(OUTPUT_DIR, 'leaderboard_table.md')
with open(md_path, 'w') as f:
    f.write('## Experiment Leaderboard (Top 15 by Pixel F1)\\n\\n')
    f.write(md_table)
    f.write('\\n')
print(f'Saved: {md_path}')
display(Markdown(md_table))

# Best model summary card
best = leaderboard.iloc[0]
summary_lines = [
    f'Best Model: {best["version"]}',
    f'Change:     {best["change"]}',
    f'Feature:    {best["feature_set"]}',
    f'Pixel F1:   {best["pixel_f1"]:.4f}',
    f'IoU:        {best["pixel_iou"]:.4f}',
    f'Pixel AUC:  {best["pixel_auc"]:.4f}' if pd.notna(best.get('pixel_auc')) else '',
    f'Img Acc:    {best["image_accuracy"]:.4f}' if pd.notna(best.get('image_accuracy')) else '',
]
summary_txt = '\\n'.join(l for l in summary_lines if l)
txt_path = os.path.join(OUTPUT_DIR, 'best_model_summary.txt')
with open(txt_path, 'w') as f:
    f.write(summary_txt)
print(f'Saved: {txt_path}')
print(summary_txt)"""))

# ── Chart 1: Ablation Progression ────────────────────────────────────────────
cells.append(md("## Chart 1 — Ablation Progression"))
cells.append(code("""fig, ax1 = plt.subplots(figsize=(16, 6))

versions = dedup['version'].values
f1_vals  = dedup['pixel_f1'].values
iou_vals = dedup['pixel_iou'].values
x = np.arange(len(versions))

# Colors: green if improved over previous, red if regressed
colors = ['#4CAF50']  # first point neutral green
for i in range(1, len(f1_vals)):
    colors.append('#4CAF50' if f1_vals[i] >= f1_vals[i-1] else '#F44336')

# F1 line (left axis)
ax1.plot(x, f1_vals, '-', color='#1565C0', linewidth=1.5, alpha=0.4, zorder=1)
for i in range(len(x)):
    ax1.scatter(x[i], f1_vals[i], c=colors[i], s=70, edgecolors='black',
                linewidth=0.5, zorder=3)
ax1.set_ylabel('Pixel F1', color='#1565C0', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#1565C0')

# Delta annotations between consecutive points
for i in range(1, len(f1_vals)):
    delta = f1_vals[i] - f1_vals[i-1]
    if abs(delta) > 0.005:
        sign = '+' if delta > 0 else ''
        ax1.annotate(f'{sign}{delta:.3f}', xy=((x[i]+x[i-1])/2, (f1_vals[i]+f1_vals[i-1])/2),
                     fontsize=6, ha='center', va='bottom', color='gray', alpha=0.7)

# Best model annotation
best_idx = np.argmax(f1_vals)
ax1.annotate(f'{versions[best_idx]}\\nF1 = {f1_vals[best_idx]:.4f}',
             xy=(x[best_idx], f1_vals[best_idx]),
             xytext=(0, 20), textcoords='offset points', ha='center',
             fontsize=9, fontweight='bold', color='#D32F2F',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', edgecolor='#D32F2F'),
             arrowprops=dict(arrowstyle='->', color='#D32F2F'))
ax1.scatter(x[best_idx], f1_vals[best_idx], marker='*', s=250, c='gold',
            edgecolors='#D32F2F', linewidth=1, zorder=5)

# IoU line (right axis)
ax2 = ax1.twinx()
ax2.plot(x, iou_vals, 's--', color='#4CAF50', linewidth=1, markersize=4, alpha=0.5)
ax2.set_ylabel('Pixel IoU', color='#4CAF50', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#4CAF50')

ax1.set_xticks(x)
ax1.set_xticklabels(versions, rotation=55, ha='right', fontsize=7)
ax1.set_xlabel('Experiment Version')
ax1.set_title('Ablation Progression — Pixel F1 & IoU Across Experiments', fontsize=13, pad=15)
ax1.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'ablation_progression.png'), dpi=DPI)
print(f'Saved: {OUTPUT_DIR}/ablation_progression.png')
plt.show()"""))

# ── Chart 2: Impact Hierarchy Lollipop ───────────────────────────────────────
cells.append(md("## Chart 2 — Impact Hierarchy (Lollipop Chart)"))
cells.append(code("""# Build impact data from the de-duplicated leaderboard
# Find the key experiments and compute their delta from the ELA baseline (P.3)
baseline_version = 'vR.P.3'
baseline_row = dedup[dedup['version'] == baseline_version]
if len(baseline_row) == 0:
    baseline_row = dedup[dedup['version'].str.contains('P.3', na=False)]
baseline_f1 = baseline_row['pixel_f1'].values[0] if len(baseline_row) > 0 else 0.692

# Impact entries: (label, delta_pp, category)
# We compute these dynamically from the leaderboard
impact_data = []
key_versions = {
    'vR.P.19': 'Multi-Q RGB ELA 9ch',
    'vR.P.15': 'Multi-Q ELA (grayscale)',
    'vR.P.10': 'CBAM attention',
    'vR.P.7':  'Extended training (50ep)',
    'vR.P.8':  'Progressive unfreeze',
    'vR.P.12': 'Data augmentation',
    'vR.P.9':  'Focal+Dice loss',
    'vR.P.16': 'DCT spatial map',
    'vR.P.14': 'Test-Time Augmentation',
    'vR.P.27': 'JPEG compression aug',
    'vR.P.28': 'Cosine annealing LR',
}

for ver, label in key_versions.items():
    row = dedup[dedup['version'] == ver]
    if len(row) > 0:
        delta = (row['pixel_f1'].values[0] - baseline_f1) * 100  # pp
        impact_data.append((label + f' ({ver})', delta))

# Sort by absolute impact
impact_data.sort(key=lambda x: abs(x[1]), reverse=True)

if len(impact_data) > 0:
    labels = [d[0] for d in impact_data]
    deltas = [d[1] for d in impact_data]
    bar_colors = ['#4CAF50' if d >= 0 else '#F44336' for d in deltas]

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.45)))
    y_pos = np.arange(len(labels))

    # Lollipop: stem + dot
    ax.hlines(y=y_pos, xmin=0, xmax=deltas, color=bar_colors, linewidth=2.5, alpha=0.8)
    ax.scatter(deltas, y_pos, c=bar_colors, s=100, edgecolors='black', linewidth=0.5, zorder=5)

    # Value labels
    for i, (d, lbl) in enumerate(zip(deltas, labels)):
        sign = '+' if d > 0 else ''
        offset = 0.5 if d >= 0 else -0.5
        ha = 'left' if d >= 0 else 'right'
        ax.text(d + offset, i, f'{sign}{d:.2f}pp', va='center', ha=ha, fontsize=9, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Impact vs ELA Baseline (P.3) in percentage points', fontsize=11)
    ax.set_title('Impact Hierarchy — Change Contribution vs ELA Baseline', fontsize=13, pad=15)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'impact_hierarchy.png'), dpi=DPI)
    print(f'Saved: {OUTPUT_DIR}/impact_hierarchy.png')
    plt.show()
else:
    print('Not enough data for impact hierarchy.')"""))

# ── Chart 3: Top Experiments Bar ─────────────────────────────────────────────
cells.append(md("## Chart 3 — Top Experiments Ranked Bar Chart"))
cells.append(code("""plot_df = leaderboard.head(15).iloc[::-1].copy()

if len(plot_df) > 0:
    fig, ax = plt.subplots(figsize=(12, max(5, len(plot_df) * 0.4)))

    # Color gradient: red (worst) -> green (best)
    norm = plt.Normalize(plot_df['pixel_f1'].min(), plot_df['pixel_f1'].max())
    cmap = plt.cm.RdYlGn
    bar_colors = [cmap(norm(v)) for v in plot_df['pixel_f1']]

    bars = ax.barh(range(len(plot_df)), plot_df['pixel_f1'], color=bar_colors,
                   edgecolor='gray', linewidth=0.5)

    # Highlight best (last bar = top of reversed list)
    bars[-1].set_edgecolor('#D32F2F')
    bars[-1].set_linewidth(2)

    # Value labels
    for i, (v, ver) in enumerate(zip(plot_df['pixel_f1'], plot_df['version'])):
        if pd.notna(v):
            ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=8, fontweight='bold')

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([f'{v}  ({c[:30]})' for v, c in
                        zip(plot_df['version'], plot_df['change'].fillna(''))], fontsize=8)
    ax.set_xlabel('Pixel F1', fontsize=11)
    ax.set_title('Top 15 Experiments by Pixel F1', fontsize=13, pad=15)
    ax.set_xlim(right=plot_df['pixel_f1'].max() * 1.08)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'top_experiments_bar.png'), dpi=DPI)
    print(f'Saved: {OUTPUT_DIR}/top_experiments_bar.png')
    plt.show()"""))

# ── Chart 4: Radar Chart ────────────────────────────────────────────────────
cells.append(md("## Chart 4 — Multi-Metric Radar: Baseline vs Best"))
cells.append(code("""# Pick 3 key models for comparison
radar_versions = ['vR.P.1', 'vR.P.15', 'vR.P.19']
radar_labels   = ['P.1 (RGB baseline)', 'P.15 (Multi-Q gray ELA)', 'P.19 (Multi-Q RGB ELA)']
radar_colors   = ['#F44336', '#FF9800', '#4CAF50']
radar_metrics  = ['pixel_f1', 'pixel_iou', 'pixel_auc', 'pixel_precision', 'pixel_recall']
metric_labels  = ['Pixel F1', 'Pixel IoU', 'Pixel AUC', 'Precision', 'Recall']

radar_rows = []
for v in radar_versions:
    row = dedup[dedup['version'] == v]
    if len(row) > 0:
        vals = [row[m].values[0] if pd.notna(row[m].values[0]) else 0 for m in radar_metrics]
        radar_rows.append(vals)
    else:
        radar_rows.append([0] * len(radar_metrics))

if any(sum(r) > 0 for r in radar_rows):
    N = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for vals, label, color in zip(radar_rows, radar_labels, radar_colors):
        vals_plot = vals + vals[:1]
        ax.fill(angles, vals_plot, alpha=0.15, color=color)
        ax.plot(angles, vals_plot, 'o-', linewidth=2, markersize=6, label=label, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Comparison: Baseline vs Best', fontsize=13, pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'radar_comparison.png'), dpi=DPI)
    print(f'Saved: {OUTPUT_DIR}/radar_comparison.png')
    plt.show()
else:
    print('Radar versions not found in data.')"""))

# ── Chart 5: Training Curves ────────────────────────────────────────────────
cells.append(md("## Chart 5 — Training Curves (Top 5 Runs)"))
cells.append(code("""top5 = leaderboard.head(5)
palette = sns.color_palette('tab10', n_colors=len(top5))

if len(top5) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (_, row) in enumerate(top5.iterrows()):
        try:
            run = api.run(f'{WANDB_ENTITY}/{WANDB_PROJECT}/{row["wandb_run_id"]}')
            history = run.history(pandas=True)
            if history.empty:
                continue
            label = row['version']
            color = palette[i]

            if 'train_loss' in history.columns:
                v = history[history['train_loss'].notna()]
                axes[0].plot(v.get('epoch', v.index), v['train_loss'],
                             label=label, color=color, linewidth=1.5)
            if 'val_loss' in history.columns:
                v = history[history['val_loss'].notna()]
                axes[1].plot(v.get('epoch', v.index), v['val_loss'],
                             label=label, color=color, linewidth=1.5)
            if 'val_pixel_f1' in history.columns:
                v = history[history['val_pixel_f1'].notna()]
                axes[2].plot(v.get('epoch', v.index), v['val_pixel_f1'],
                             label=label, color=color, linewidth=1.5)
        except Exception as e:
            print(f'  Skipped {row["version"]}: {e}')

    for ax, title, ylabel in zip(axes,
        ['Train Loss', 'Validation Loss', 'Val Pixel F1'],
        ['Loss', 'Loss', 'Pixel F1']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle('Training Curves — Top 5 Experiments', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=DPI)
    print(f'Saved: {OUTPUT_DIR}/training_curves.png')
    plt.show()
else:
    print('No runs for training curves.')"""))

# ── Chart 6: Feature Set Grouped Bar ─────────────────────────────────────────
cells.append(md("## Chart 6 — Feature Set Comparison (Grouped Bar)"))
cells.append(code("""valid = df[df['pixel_f1'].notna()]

if valid['feature_set'].nunique() > 1:
    feat = valid.groupby('feature_set').agg(
        F1_max=('pixel_f1', 'max'),
        IoU_max=('pixel_iou', 'max'),
        runs=('pixel_f1', 'count')
    ).sort_values('F1_max', ascending=True).reset_index()

    fig, ax = plt.subplots(figsize=(12, max(4, len(feat) * 0.5)))
    y = np.arange(len(feat))
    bar_h = 0.35

    bars_f1 = ax.barh(y - bar_h/2, feat['F1_max'], bar_h, label='Pixel F1',
                       color='#1565C0', edgecolor='white')
    bars_iou = ax.barh(y + bar_h/2, feat['IoU_max'], bar_h, label='Pixel IoU',
                        color='#4CAF50', edgecolor='white')

    # Value labels
    for bar_set in [bars_f1, bars_iou]:
        for bar in bar_set:
            w = bar.get_width()
            if pd.notna(w) and w > 0:
                ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{w:.4f}', va='center', fontsize=8)

    # Highlight best
    best_idx = feat['F1_max'].idxmax()
    bars_f1[best_idx].set_edgecolor('#D32F2F')
    bars_f1[best_idx].set_linewidth(2)

    ax.set_yticks(y)
    ax.set_yticklabels(feat['feature_set'], fontsize=9)
    ax.set_xlabel('Metric Value', fontsize=11)
    ax.set_title('Best Performance by Input Representation (Feature Set)', fontsize=13, pad=15)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'feature_set_comparison.png'), dpi=DPI)
    print(f'Saved: {OUTPUT_DIR}/feature_set_comparison.png')
    plt.show()
else:
    print('Not enough distinct feature sets.')"""))

# ── Chart 7: Precision-Recall Scatter ────────────────────────────────────────
cells.append(md("## Chart 7 — Precision–Recall Scatter with Iso-F1 Curves"))
cells.append(code("""metric_df = df[df['pixel_f1'].notna()].copy()
has_pr = metric_df['pixel_precision'].notna() & metric_df['pixel_recall'].notna()
pr_df = metric_df[has_pr]

if len(pr_df) > 1:
    fig, ax = plt.subplots(figsize=(10, 8))

    # Iso-F1 curves
    for f1_val in [0.4, 0.5, 0.6, 0.7, 0.8]:
        r = np.linspace(0.01, 1.0, 200)
        p = (f1_val * r) / (2 * r - f1_val)
        mask = (p > 0) & (p <= 1)
        ax.plot(r[mask], p[mask], '--', color='gray', alpha=0.3, linewidth=0.8)
        # Label the curve
        idx = np.argmin(np.abs(r - 0.95))
        if mask[idx]:
            ax.text(r[idx], p[idx], f'F1={f1_val}', fontsize=7, color='gray', alpha=0.5)

    # Scatter: size = F1, color = feature_set
    feat_sets = pr_df['feature_set'].unique()
    cmap_cat = plt.cm.get_cmap('Set1', max(len(feat_sets), 3))

    for i, fs in enumerate(feat_sets):
        subset = pr_df[pr_df['feature_set'] == fs]
        sizes = (subset['pixel_f1'] * 300).clip(lower=40)
        ax.scatter(subset['pixel_recall'], subset['pixel_precision'],
                   s=sizes, c=[cmap_cat(i)] * len(subset),
                   label=fs if fs else 'Unknown', alpha=0.7,
                   edgecolors='black', linewidth=0.5)

    # Version labels for top models
    top_pr = pr_df.nlargest(10, 'pixel_f1')
    for _, r in top_pr.iterrows():
        ax.annotate(r['version'], (r['pixel_recall'], r['pixel_precision']),
                    fontsize=7, ha='center', va='bottom', alpha=0.8,
                    xytext=(0, 5), textcoords='offset points')

    ax.set_xlabel('Pixel Recall', fontsize=11)
    ax.set_ylabel('Pixel Precision', fontsize=11)
    ax.set_title('Precision–Recall Trade-off by Feature Set (bubble size = F1)', fontsize=13, pad=15)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(title='Feature Set', loc='lower left', fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'precision_recall_scatter.png'), dpi=DPI)
    print(f'Saved: {OUTPUT_DIR}/precision_recall_scatter.png')
    plt.show()
else:
    print('Not enough precision/recall data.')"""))

# ── Chart 8: Metric Heatmap ─────────────────────────────────────────────────
cells.append(md("## Chart 8 — Metric Correlation Heatmap"))
cells.append(code("""hm_cols = ['pixel_f1', 'pixel_iou', 'pixel_auc', 'pixel_precision',
           'pixel_recall', 'image_accuracy', 'image_macro_f1']
hm_data = metric_df[hm_cols].dropna(axis=1, how='all')

if hm_data.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = hm_data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=-1, vmax=1, ax=ax, square=True, mask=mask,
                linewidths=0.5, linecolor='white')
    ax.set_title('Metric Correlation Heatmap', fontsize=13, pad=15)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'metric_heatmap.png'), dpi=DPI)
    print(f'Saved: {OUTPUT_DIR}/metric_heatmap.png')
    plt.show()
else:
    print('Not enough data for heatmap.')"""))

# ── Summary & Export ─────────────────────────────────────────────────────────
cells.append(md("## Summary & W&B Artifact Upload"))
cells.append(code("""valid = df[df['pixel_f1'].notna()]
print('=' * 70)
print('  EXPERIMENT SUMMARY')
print('=' * 70)
print(f'  Total runs:          {len(df)}')
print(f'  Completed (F1 logged): {len(valid)}')
print(f'  Unique versions:     {valid["version"].nunique()}')
print(f'  Total GPU time:      {valid["duration_min"].sum():.0f} min ({valid["duration_min"].sum()/60:.1f} hrs)')
print()
print(f'  Best Pixel F1:  {valid["pixel_f1"].max():.4f}  ({valid.loc[valid["pixel_f1"].idxmax(), "version"]})')
print(f'  Mean Pixel F1:  {valid["pixel_f1"].mean():.4f}')
print(f'  Best Pixel IoU: {valid["pixel_iou"].max():.4f}')
print('=' * 70)

# ── List all saved files ──
print('\\n--- Saved Files ---')
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f'  {f:40s}  {size/1024:.1f} KB')"""))

cells.append(code("""# ── Upload to W&B as artifact ──
export_files = [os.path.join(OUTPUT_DIR, f) for f in os.listdir(OUTPUT_DIR)]

try:
    wandb.finish()  # close any prior run
except:
    pass

try:
    with wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name='leaderboard_v2_export',
        job_type='analysis',
    ):
        artifact = wandb.Artifact('leaderboard_v2', type='results')
        for f in export_files:
            if os.path.exists(f):
                artifact.add_file(f)
                print(f'  Added: {f}')
        wandb.log_artifact(artifact)

        # Log leaderboard as W&B Table
        wandb.log({'leaderboard_v2': wandb.Table(
            dataframe=leaderboard[lb_cols].head(20))})
        print('\\nLeaderboard v2 artifact uploaded to W&B.')
except Exception as e:
    print(f'W&B upload skipped: {e}')

print('\\n=== All done! ===')"""))

# ── Build the notebook JSON ──────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

# Fix: split source lines properly (each line must end with \n except last)
for cell in notebook["cells"]:
    raw = "\n".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    lines = raw.split("\n")
    cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "w-b-leaderboard-v2.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
print(f"Total cells: {len(cells)}")
