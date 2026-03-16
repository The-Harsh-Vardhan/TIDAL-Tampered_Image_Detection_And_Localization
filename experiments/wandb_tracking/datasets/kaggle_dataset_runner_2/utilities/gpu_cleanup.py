"""GPU memory cleanup between experiments."""

import gc
import torch


def cleanup_gpu():
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
