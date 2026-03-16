"""
Environment-aware path configuration for sweep modules.

Detects Kaggle vs local execution and sets paths accordingly.
Locally, set DATASET_ROOT env var to the directory containing the
CASIA v2.0 dataset (parent of Au/ and Tp/ directories).
"""

import os

if os.path.isdir('/kaggle/input'):
    DATA_SEARCH_ROOTS = ['/kaggle/input']
    CHECKPOINT_DIR = '/kaggle/working/checkpoints'
    RESULTS_DIR = '/kaggle/working/results'
    LOGS_DIR = '/kaggle/working/logs'
else:
    DATA_SEARCH_ROOTS = [
        os.environ.get('DATASET_ROOT', './data'),
    ]
    CHECKPOINT_DIR = os.environ.get('CHECKPOINT_DIR', './sweep_outputs/checkpoints')
    RESULTS_DIR = os.environ.get('RESULTS_DIR', './sweep_outputs/results')
    LOGS_DIR = os.environ.get('LOGS_DIR', './sweep_outputs/logs')

for _d in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)
