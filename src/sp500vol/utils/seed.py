"""Global seeding for reproducibility.

Every training run must call `seed_everything(cfg.seed)` before any randomness.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducibility.

    Notes:
        - Does NOT make GPU runs bit-exact across hardware. Use deterministic
          algorithms only when the cost is acceptable; see PyTorch docs.
        - PYTHONHASHSEED is set as a process env var; only affects subprocesses.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
