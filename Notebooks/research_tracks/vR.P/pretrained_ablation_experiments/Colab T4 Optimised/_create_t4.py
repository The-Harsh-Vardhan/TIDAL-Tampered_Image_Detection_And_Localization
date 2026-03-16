"""Create Colab T4 Optimised versions of all 6 Final Notebooks.

Changes vs original Final Notebooks:
- IMAGE_SIZE: 384 → 256  (2.25x fewer pixels per forward pass)
- BATCH_SIZE: 16  → 32   (fits 16GB VRAM at 256x256)
- NUM_WORKERS: 2  → 4    (more parallelism, safe since dataset is local)
- P.30.1/P.30.4: EPOCHS 50→30, PATIENCE 10→7
- P.30.2:        EPOCHS 40→25, PATIENCE  7→5
- W&B run name prefixed with [T4] instead of [FINAL]
- Separate checkpoint/results dirs so T4 runs don't overwrite full-res runs
"""
import json, os, shutil, re

SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Final Notebooks')
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT_DIR, exist_ok=True)

NOTEBOOKS = ['vR.P.19', 'vR.P.30', 'vR.P.30.1', 'vR.P.30.2', 'vR.P.30.3', 'vR.P.30.4']

# Per-notebook epoch / patience overrides
EPOCH_OVERRIDES = {
    'vR.P.30.1': (30, 7),
    'vR.P.30.2': (25, 5),
    'vR.P.30.4': (30, 7),
}


def sub(src, old, new):
    if old not in src:
        print(f'    WARNING: "{old}" not found')
    return src.replace(old, new)


for version in NOTEBOOKS:
    fname    = f'{version} Image Detection and Localisation.ipynb'
    src_path = os.path.join(SOURCE_DIR, fname)
    dst_path = os.path.join(OUT_DIR,    fname)

    shutil.copy2(src_path, dst_path)

    with open(dst_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # ---- Cell 2: config --------------------------------------------------------
    c2 = ''.join(nb['cells'][2]['source'])

    # IMAGE_SIZE
    c2 = re.sub(r'IMAGE_SIZE\s*=\s*384', 'IMAGE_SIZE = 256', c2)

    # BATCH_SIZE
    c2 = re.sub(r'BATCH_SIZE\s*=\s*16', 'BATCH_SIZE = 32', c2)

    # NUM_WORKERS
    c2 = re.sub(r'NUM_WORKERS\s*=\s*\d+', 'NUM_WORKERS = 4', c2)

    # EPOCHS / PATIENCE
    epochs, patience = EPOCH_OVERRIDES.get(version, (None, None))
    if epochs:
        c2 = re.sub(r'EPOCHS\s*=\s*\d+',   f'EPOCHS = {epochs}',   c2)
        c2 = re.sub(r'PATIENCE\s*=\s*\d+', f'PATIENCE = {patience}', c2)

    # Separate persistence dirs so T4 runs don't clobber full-res checkpoints
    c2 = c2.replace(
        "CHECKPOINT_DIR = '/content/drive/MyDrive/BigVision/checkpoints'",
        "CHECKPOINT_DIR = '/content/drive/MyDrive/BigVision/checkpoints_t4'"
    )
    c2 = c2.replace(
        "RESULTS_DIR = '/content/drive/MyDrive/BigVision/results'",
        "RESULTS_DIR = '/content/drive/MyDrive/BigVision/results_t4'"
    )
    c2 = c2.replace(
        "LOGS_DIR = '/content/drive/MyDrive/BigVision/logs'",
        "LOGS_DIR = '/content/drive/MyDrive/BigVision/logs_t4'"
    )

    # W&B run name: [FINAL] → [T4]
    c2 = c2.replace("name=f'[FINAL] {VERSION}'", "name=f'[T4] {VERSION}'")

    # W&B tags: replace 'final' with 't4-optimised'
    c2 = c2.replace("tags=['final',", "tags=['t4-optimised',")

    # W&B group
    c2 = c2.replace(
        "group='TIDAL-Final-Ablation'",
        "group='TIDAL-T4-Optimised'"
    )

    # notebook_type config key
    c2 = c2.replace("'notebook_type': 'final'", "'notebook_type': 't4-optimised'")

    # Add T4 optimisation note after SEED line (or near top of config block)
    c2 = c2.replace(
        "VERSION = 'vR.P.",
        "# --- Colab T4 Optimised: IMAGE_SIZE=256, BATCH_SIZE=32 ---\nVERSION = 'vR.P."
    )

    nb['cells'][2]['source'] = [c2]

    # ---- Cell 0: update platform note in header --------------------------------
    c0 = ''.join(nb['cells'][0]['source'])
    c0 = c0.replace('Google Colab (T4/A100 GPU)', 'Google Colab T4 (Optimised: 256px, BS=32)')

    # Add T4 optimised row to metadata table if Platform row exists
    if '| **Platform**' in c0:
        c0 = c0.replace(
            '| **Platform** | Google Colab T4 (Optimised: 256px, BS=32) |',
            '| **Platform** | Google Colab T4 (Optimised: 256px, BS=32) |\n| **Optimised for** | T4 GPU: IMAGE_SIZE=256, BATCH_SIZE=32, ~2.5x faster than 384px |'
        )

    nb['cells'][0]['source'] = [c0]

    # ---- Cell 13: update training config table ---------------------------------
    c13 = ''.join(nb['cells'][13]['source'])

    # Fix image size value in table
    c13 = c13.replace('| **Image size** | 384x384 |', '| **Image size** | 256x256 *(T4 optimised)* |')
    c13 = c13.replace('| **Batch size** | 16 |', '| **Batch size** | 32 *(T4 optimised)* |')

    # Fix epochs/patience if overridden
    if epochs:
        c13 = re.sub(r'\| \*\*Epochs\*\* \|.*?\|', f'| **Epochs** | {epochs} max *(T4 optimised)* |', c13)
        c13 = re.sub(r'\| \*\*Early stopping\*\* \|.*?\|',
                     f'| **Early stopping** | patience={patience}, monitor=val_loss |', c13)

    nb['cells'][13]['source'] = [c13]

    # ---- HF upload: tag as t4-optimised ----------------------------------------
    c27 = ''.join(nb['cells'][27]['source'])
    c27 = c27.replace(
        "path_in_repo=f'checkpoints/{_fname}'",
        "path_in_repo=f'checkpoints/t4/{_fname}'"
    )
    nb['cells'][27]['source'] = [c27]

    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    ep_str = f', EPOCHS={epochs}' if epochs else ''
    print(f'Created: {fname}  (256px, BS=32{ep_str})')

print('\nDone!')
