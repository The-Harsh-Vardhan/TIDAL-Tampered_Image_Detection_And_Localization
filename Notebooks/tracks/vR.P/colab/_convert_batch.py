"""Convert Kaggle notebooks to Google Colab compatible versions.

Applies to: vR.P.30, vR.P.30.2, vR.P.30.3, vR.P.30.4

Modifies:
- Cell 0: Platform field in markdown header
- Cell 2: Persistence paths, W&B secrets, pip installs, HF_REPO_ID
- Cell 4: Dataset discovery (Google Drive primary, Kaggle API fallback)
- Cell 27: HuggingFace Hub upload after W&B block
"""
import json
import os
import shutil

SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'source')
COLAB_DIR  = os.path.dirname(os.path.abspath(__file__))

TARGETS = ['vR.P.30', 'vR.P.30.2', 'vR.P.30.3', 'vR.P.30.4']

NEW_CELL4 = '''# ============================================================
# 2.1 Dataset Path Discovery (Colab Version)
# ============================================================
# Priority: Google Drive (pre-uploaded) > Kaggle API download
# Expected Google Drive layout:
#   /content/drive/MyDrive/BigVision/datasets/CASIA2/
#       IMAGE/  (contains Au/ and Tp/)
#       MASK/   (contains Au/ and Tp/)

from google.colab import drive

# --- Mount Google Drive ---
GDRIVE_MOUNT = '/content/drive'
if not os.path.ismount(GDRIVE_MOUNT):
    drive.mount(GDRIVE_MOUNT)

GDRIVE_DATASET_DIR = '/content/drive/MyDrive/BigVision/datasets/CASIA2'

def find_dataset():
    """Search for Au/ and Tp/ directories.

    Search order:
      1. Google Drive (GDRIVE_DATASET_DIR)
      2. /content/datasets (Kaggle API download location)
      3. /kaggle/input (if somehow running on Kaggle)

    Returns: (dataset_root, au_dir, tp_dir, gt_mask_dir_or_None)
    """
    search_roots = [
        GDRIVE_DATASET_DIR,
        '/content/datasets',
        '/kaggle/input',
    ]
    candidates = []

    for base in search_roots:
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if 'Au' in dirnames and 'Tp' in dirnames:
                candidates.append((
                    dirpath,
                    os.path.join(dirpath, 'Au'),
                    os.path.join(dirpath, 'Tp')
                ))

    if not candidates:
        return None, None, None, None

    # Separate IMAGE vs MASK candidates
    image_candidates = [c for c in candidates if 'mask' not in c[0].lower()]
    mask_candidates  = [c for c in candidates if 'mask' in c[0].lower()]

    if image_candidates:
        explicit_image = [c for c in image_candidates if 'image' in c[0].lower()]
        chosen = explicit_image[0] if explicit_image else image_candidates[0]
    else:
        chosen = candidates[0]

    gt_dir = None
    if mask_candidates:
        gt_dir = mask_candidates[0][0]

    return chosen[0], chosen[1], chosen[2], gt_dir


# --- Try finding the dataset ---
DATASET_ROOT, AU_DIR, TP_DIR, GT_DIR_ROOT = find_dataset()

# --- Fallback: Download via Kaggle API if not found ---
if DATASET_ROOT is None:
    print("Dataset not found on Google Drive. Downloading via Kaggle API...")
    KAGGLE_DATASET = 'sagnikkayalcse52/casia-spicing-detection-localization'
    DOWNLOAD_DIR = '/content/datasets'

    # Configure Kaggle credentials from Colab secrets
    try:
        from google.colab import userdata
        os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    except Exception:
        print("WARNING: Could not load Kaggle credentials from Colab secrets.")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY in Colab secrets,")
        print("or upload your kaggle.json to ~/.kaggle/kaggle.json")

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
    subprocess.run([
        'kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET,
        '-p', DOWNLOAD_DIR, '--unzip'
    ], check=True)

    # Retry discovery after download
    DATASET_ROOT, AU_DIR, TP_DIR, GT_DIR_ROOT = find_dataset()

if DATASET_ROOT is None:
    raise FileNotFoundError(
        'Could not find Au/ and Tp/ directories.\\n'
        'Please either:\\n'
        f'  1. Upload CASIA2 dataset to Google Drive at: {GDRIVE_DATASET_DIR}\\n'
        '  2. Add KAGGLE_USERNAME and KAGGLE_KEY to Colab secrets'
    )

# --- Resolve GT mask directory ---
GT_DIR = None
if GT_DIR_ROOT is not None:
    gt_tp_dir = os.path.join(GT_DIR_ROOT, 'Tp')
    if os.path.isdir(gt_tp_dir):
        gt_files = [f for f in os.listdir(gt_tp_dir)
                    if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.bmp'}]
        if gt_files:
            GT_DIR = GT_DIR_ROOT
            print(f'GT mask structure detected: {GT_DIR}')
            print(f'  MASK/Au: {len(os.listdir(os.path.join(GT_DIR, "Au")))} files')
            print(f'  MASK/Tp: {len(gt_files)} mask files')

if GT_DIR is None:
    search_base = os.path.dirname(DATASET_ROOT)
    for root, dirs, files in os.walk(search_base):
        for d in dirs:
            if any(pat in d.lower() for pat in ['groundtruth', 'gt', 'mask']):
                candidate = os.path.join(root, d)
                if any(Path(f).suffix.lower() in {'.jpg','.jpeg','.png','.tif','.bmp'}
                       for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))):
                    GT_DIR = candidate
                    break
        if GT_DIR:
            break

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

print(f'\\nDataset root:  {DATASET_ROOT}')
print(f'Authentic dir: {AU_DIR}  ({len(os.listdir(AU_DIR))} files)')
print(f'Tampered dir:  {TP_DIR}  ({len(os.listdir(TP_DIR))} files)')
if GT_DIR:
    print(f'GT mask dir:   {GT_DIR}')
else:
    print(f'GT mask dir:   NOT FOUND \\u2014 will generate pseudo-masks from ELA')
'''

HF_UPLOAD_BLOCK = r'''

# --- HuggingFace Hub: Upload Checkpoints ---
USE_HF_HUB = True

if USE_HF_HUB:
    try:
        from huggingface_hub import HfApi, login
        from google.colab import userdata

        _hf_token = userdata.get('HF_TOKEN')
        login(token=_hf_token)
        api = HfApi()

        # Create repo if it doesn't exist
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, private=True)

        _uploaded = []
        for _fpath, _fname in [
            (BEST_MODEL_PATH, f'{VERSION}_best_model.pt'),
            (LATEST_CHECKPOINT, f'{VERSION}_latest_checkpoint.pt'),
        ]:
            if os.path.exists(_fpath):
                api.upload_file(
                    path_or_fileobj=_fpath,
                    path_in_repo=f'checkpoints/{_fname}',
                    repo_id=HF_REPO_ID,
                    commit_message=f'Upload {_fname} ({VERSION})',
                )
                _uploaded.append(_fname)

        if _uploaded:
            print(f'HuggingFace Hub: uploaded {", ".join(_uploaded)} to {HF_REPO_ID}')
        else:
            print('HuggingFace Hub: no checkpoint files found to upload')

    except Exception as e:
        print(f'HuggingFace Hub upload failed ({e}), continuing without upload')
        print('Ensure HF_TOKEN is set in Colab secrets')
'''

for version in TARGETS:
    fname = f'{version} Image Detection and Localisation.ipynb'
    src_path  = os.path.join(SOURCE_DIR, fname)
    dest_path = os.path.join(COLAB_DIR,  fname)

    shutil.copy2(src_path, dest_path)

    with open(dest_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # --- Cell 0: Platform ---
    c0 = ''.join(nb['cells'][0]['source'])
    c0 = c0.replace('Kaggle (T4/P100 GPU)', 'Google Colab (T4/A100 GPU)')
    nb['cells'][0]['source'] = [c0]

    # --- Cell 2: Setup ---
    c2 = ''.join(nb['cells'][2]['source'])

    # pip installs
    c2 = c2.replace(
        '!pip install -q wandb',
        '!pip install -q wandb\n!pip install -q huggingface_hub'
    )

    # Persistence paths
    c2 = c2.replace("CHECKPOINT_DIR = '/kaggle/working/checkpoints'",
                    "CHECKPOINT_DIR = '/content/drive/MyDrive/BigVision/checkpoints'")
    c2 = c2.replace("RESULTS_DIR = '/kaggle/working/results'",
                    "RESULTS_DIR = '/content/drive/MyDrive/BigVision/results'")
    c2 = c2.replace("LOGS_DIR = '/kaggle/working/logs'",
                    "LOGS_DIR = '/content/drive/MyDrive/BigVision/logs'")

    # W&B secrets
    c2 = c2.replace('from kaggle_secrets import UserSecretsClient',
                    'from google.colab import userdata')
    c2 = c2.replace('_key = UserSecretsClient().get_secret("WANDB_API_KEY")',
                    "_key = userdata.get('WANDB_API_KEY')")

    # Run ID default
    c2 = c2.replace(
        "RUN_ID = f'run{_run_match.group(1)}' if _run_match else 'run01'",
        "RUN_ID = f'run{_run_match.group(1)}' if _run_match else 'colab01'"
    )

    # HF repo config
    c2 = c2.replace(
        "DATASET_NAME = 'CASIA2'",
        "DATASET_NAME = 'CASIA2'\nHF_REPO_ID = 'the-harsh-vardhan/TIDAL'  # HuggingFace repo for model weights"
    )

    nb['cells'][2]['source'] = [c2]

    # --- Cell 4: Dataset discovery ---
    nb['cells'][4]['source'] = [NEW_CELL4]

    # --- Cell 27: HF upload ---
    c27 = ''.join(nb['cells'][27]['source'])
    c27 += HF_UPLOAD_BLOCK
    nb['cells'][27]['source'] = [c27]

    with open(dest_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f'Created: {fname}')

print('\nDone!')
