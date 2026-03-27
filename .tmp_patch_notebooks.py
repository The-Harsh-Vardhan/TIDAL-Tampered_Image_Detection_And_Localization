import json
import re
from pathlib import Path

roots = [
    Path('Notebooks/tracks/vR.P/final_upgraded'),
    Path('Notebooks/tracks/vR.P/colab'),
    Path('Notebooks/tracks/vR.P/source'),
]

helper = '''def _get_secret(name, aliases=None):
    aliases = aliases or [name]

    # 1) Environment variables
    for key in [name] + [a for a in aliases if a != name]:
        val = os.getenv(key)
        if val:
            return val

    # 2) Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient
        _usc = UserSecretsClient()
        for key in [name] + [a for a in aliases if a != name]:
            try:
                val = _usc.get_secret(key)
                if val:
                    return val
            except Exception:
                pass
    except Exception:
        pass

    # 3) Colab userdata
    try:
        from google.colab import userdata as _colab_ud
        for key in [name] + [a for a in aliases if a != name]:
            try:
                val = _colab_ud.get(key)
                if val:
                    return val
            except Exception:
                pass
    except Exception:
        pass

    return None'''


def patch_source(src: str) -> tuple[str, bool]:
    old = src

    need_helper = any(
        token in src
        for token in [
            'WANDB_API_KEY', 'HF_TOKEN', 'KAGGLE_USERNAME', 'KAGGLE_API_KEY',
            'kaggle_secrets', 'google.colab import userdata'
        ]
    )

    if need_helper and 'def _get_secret(' not in src:
        if 'if USE_WANDB:' in src:
            src = src.replace('if USE_WANDB:', helper + '\n\nif USE_WANDB:', 1)
        else:
            # prepend helper near top of code cell
            src = helper + '\n\n' + src

    # W&B step style in vR.P.19+
    src = re.sub(
        r"""# Step 1: Try Colab userdata secret\n\s*_wandb_ok = False\n\s*try:\n\s*from google\.colab import userdata as _colab_ud\n\s*_key = _colab_ud\.get\('WANDB_API_KEY'\)\n\s*_wandb_ok = _try_wandb_init\(_key\)\n\s*except Exception:\n\s*pass\n""",
        """# Step 1: Try automatic secret detection (env -> Kaggle -> Colab)
    _wandb_ok = False
    _wandb_key = _get_secret('WANDB_API_KEY', ['WANDB_API_KEY', 'wandb_api_key', 'wandb_api'])
    if _wandb_key:
        _wandb_ok = _try_wandb_init(_wandb_key)
""",
        src,
        flags=re.MULTILINE,
    )

    # Older W&B login snippets
    src = re.sub(
        r"""try:\n\s*from google\.colab import userdata\n\s*_key = userdata\.get\('WANDB_API_KEY'\)\n\s*wandb\.login\(key=_key\)\n\s*except Exception:\n\s*wandb\.login\(\)""",
        """_wandb_key = _get_secret('WANDB_API_KEY', ['WANDB_API_KEY', 'wandb_api_key', 'wandb_api'])
        if _wandb_key:
            wandb.login(key=_wandb_key, relogin=True)
        else:
            wandb.login()""",
        src,
        flags=re.MULTILINE,
    )

    # Kaggle-only W&B snippets
    src = re.sub(
        r"""from kaggle_secrets import UserSecretsClient\n\s*_key = UserSecretsClient\(\)\.get_secret\("WANDB_API_KEY"\)\n\s*wandb\.login\(key=_key\)""",
        """_wandb_key = _get_secret('WANDB_API_KEY', ['WANDB_API_KEY', 'wandb_api_key', 'wandb_api'])
    if _wandb_key:
        wandb.login(key=_wandb_key, relogin=True)
    else:
        wandb.login()""",
        src,
        flags=re.MULTILINE,
    )

    # Variant with wandb_api_key variable
    src = re.sub(
        r"""from kaggle_secrets import UserSecretsClient\n\s*wandb_api_key = UserSecretsClient\(\)\.get_secret\("WANDB_API_KEY"\)""",
        """wandb_api_key = _get_secret('WANDB_API_KEY', ['WANDB_API_KEY', 'wandb_api_key', 'wandb_api'])""",
        src,
        flags=re.MULTILINE,
    )

    # Kaggle credentials from Colab-only
    src = re.sub(
        r"""# Configure Kaggle credentials from Colab secrets\n\s*try:\n\s*from google\.colab import userdata\n\s*os\.environ\['KAGGLE_USERNAME'\] = userdata\.get\('KAGGLE_USERNAME'\)\n\s*os\.environ\['KAGGLE_KEY'\] = userdata\.get\('KAGGLE_API_KEY'\)\n\s*except Exception:\n\s*print\("WARNING: Could not load Kaggle credentials from Colab secrets\."\)\n\s*print\("Please set KAGGLE_USERNAME and KAGGLE_KEY in Colab secrets,"\)\n\s*print\("or upload your kaggle\.json to ~/.kaggle/kaggle\.json"\)""",
        """# Configure Kaggle credentials from env/Kaggle/Colab secrets
    _ku = _get_secret('KAGGLE_USERNAME', ['KAGGLE_USERNAME', 'kaggle_username'])
    _kk = _get_secret('KAGGLE_KEY', ['KAGGLE_KEY', 'KAGGLE_API_KEY', 'kaggle_key', 'kaggle_api_key'])
    if _ku and _kk:
        os.environ['KAGGLE_USERNAME'] = _ku
        os.environ['KAGGLE_KEY'] = _kk
    else:
        print("WARNING: Could not load Kaggle credentials from env/Kaggle/Colab secrets.")
        print("Set KAGGLE_USERNAME and KAGGLE_KEY, or upload ~/.kaggle/kaggle.json")""",
        src,
        flags=re.MULTILINE,
    )

    # HF token from Colab-only
    src = re.sub(
        r"""from huggingface_hub import HfApi, login\n\s*from google\.colab import userdata\n\n\s*_hf_token = userdata\.get\('HF_TOKEN'\)\n\s*login\(token=_hf_token\)""",
        """from huggingface_hub import HfApi, login

        _hf_token = _get_secret('HF_TOKEN', ['HF_TOKEN', 'HUGGINGFACE_TOKEN', 'huggingface_token'])
        if not _hf_token:
            raise RuntimeError('HF_TOKEN not found in env/Kaggle/Colab secrets')
        login(token=_hf_token)""",
        src,
        flags=re.MULTILINE,
    )

    # Message cleanup
    src = src.replace('Ensure HF_TOKEN is set in Colab secrets', 'Ensure HF_TOKEN is set in env/Kaggle/Colab secrets')
    src = src.replace('Add KAGGLE_USERNAME and KAGGLE_KEY to Colab secrets', 'Add KAGGLE_USERNAME and KAGGLE_KEY to env/Kaggle/Colab secrets')

    return src, src != old


files = []
for root in roots:
    files.extend(sorted(root.glob('*.ipynb')))

changed_files = 0
changed_cells = 0

for path in files:
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    file_changed = False
    for cell in data.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        new_src, changed = patch_source(src)
        if changed:
            cell['source'] = new_src.splitlines(keepends=True)
            file_changed = True
            changed_cells += 1

    if file_changed:
        with path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.write('\n')
        changed_files += 1
        print(f'patched: {path.as_posix()}')

print(f'patched_files={changed_files}')
print(f'patched_cells={changed_cells}')
