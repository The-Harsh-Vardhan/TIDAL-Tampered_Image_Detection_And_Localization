"""Papermill notebook execution with timing and error handling."""

import os
import time
import papermill as pm


def execute_experiment(nb_path, out_path):
    """Execute a notebook via papermill.

    Returns:
        tuple: (status, elapsed_min, error_msg_or_None)
        status is one of: 'SUCCESS', 'FAILED', 'ERROR', 'NOT_FOUND'
    """
    if not os.path.exists(nb_path):
        return ('NOT_FOUND', 0.0, f'{nb_path} not found')

    t0 = time.time()
    try:
        pm.execute_notebook(
            nb_path,
            out_path,
            kernel_name='python3',
            cwd='/kaggle/working',
        )
        elapsed = (time.time() - t0) / 60
        return ('SUCCESS', round(elapsed, 1), None)

    except pm.PapermillExecutionError as e:
        elapsed = (time.time() - t0) / 60
        err = f'{e.ename}: {str(e.evalue)[:200]}'
        return ('FAILED', round(elapsed, 1), err)

    except Exception as e:
        elapsed = (time.time() - t0) / 60
        return ('ERROR', round(elapsed, 1), str(e)[:200])
