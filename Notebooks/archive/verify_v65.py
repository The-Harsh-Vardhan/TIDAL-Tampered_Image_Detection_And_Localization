import json, os

base = os.path.dirname(os.path.abspath(__file__))
for name in ['tamper_detection_v6.5_kaggle.ipynb', 'tamper_detection_v6.5_colab.ipynb']:
    path = os.path.join(base, name)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    src = ''.join(''.join(c['source']) for c in cells)
    md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
    code_count = sum(1 for c in cells if c['cell_type'] == 'code')
    tag = 'Kaggle' if 'kaggle' in name else 'Colab'
    print(f'{tag}: {len(cells)} cells ({md_count} md, {code_count} code)')

    features = [
        'CONFIG = {', "'use_amp'", "'use_multi_gpu'", "'use_wandb'",
        'def setup_device', 'def setup_model', 'def train_one_epoch',
        'def validate_model', 'DataParallel', 'class BCEDiceLoss',
        'class GradCAM', 'v6.5', "enabled=CONFIG", 'GradScaler',
    ]
    for feat in features:
        ok = feat in src
        print(f'  [{"OK" if ok else "FAIL"}] {feat}')

    if 'kaggle' in name:
        print(f'  [{"OK" if "/kaggle/" in src else "FAIL"}] Kaggle paths')
        print(f'  [{"OK" if "drive.mount" not in src else "FAIL"}] No Drive mount')
    else:
        print(f'  [{"OK" if "drive.mount" in src else "FAIL"}] Drive mount')
        print(f'  [{"OK" if "kaggle datasets download" in src else "FAIL"}] Kaggle API')
    print()
