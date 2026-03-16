"""W&B authentication using Kaggle Secrets."""

import os


def setup_wandb():
    """Login to W&B using the WANDB_API_KEY Kaggle Secret.

    The source notebooks handle wandb.init() / wandb.finish() internally.
    This function only handles authentication.
    """
    import wandb

    api_key = os.environ.get('WANDB_API_KEY')
    if api_key:
        wandb.login(key=api_key)
        print(f'W&B authenticated (key ...{api_key[-4:]})')
    else:
        print('WARNING: WANDB_API_KEY not found in environment.')
        print('Set it as a Kaggle Secret: Settings > Secrets > WANDB_API_KEY')
        wandb.login()
