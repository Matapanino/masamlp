"""Seeding utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int | None) -> None:
    """Seed python, NumPy, and torch (all devices). No-op when seed is None."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU and all CUDA/MPS devices
