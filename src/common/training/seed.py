"""Global RNG seeding."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs.

    Sets the `random`, NumPy, and CPU/CUDA Torch seeds plus `PYTHONHASHSEED`, and
    optionally forces deterministic cuDNN behaviour.

    Parameters
    ----------
    seed : int
        Seed value applied to every RNG and to `PYTHONHASHSEED`.
    deterministic : bool
        If True, enable deterministic cuDNN and disable its benchmark autotuner.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
