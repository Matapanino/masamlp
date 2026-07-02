"""Device-resident tensor bundle used by the Trainer."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass
class TabularData:
    """One split's tensors. ``x_num``/``x_cat`` may have zero columns but are
    always 2-D; ``y`` is whatever the objective's ``prepare_target`` built."""

    x_num: Tensor
    x_cat: Tensor
    y: Tensor | None = None
    weight: Tensor | None = None

    def __post_init__(self) -> None:
        if self.x_num.shape[0] != self.x_cat.shape[0]:
            raise ValueError("x_num and x_cat row counts differ")

    def __len__(self) -> int:
        return self.x_num.shape[0]

    def to(self, device: torch.device) -> TabularData:
        return replace(
            self,
            x_num=self.x_num.to(device),
            x_cat=self.x_cat.to(device),
            y=self.y.to(device) if self.y is not None else None,
            weight=self.weight.to(device) if self.weight is not None else None,
        )

    def slice(self, idx: Tensor) -> TabularData:
        return replace(
            self,
            x_num=self.x_num[idx],
            x_cat=self.x_cat[idx],
            y=self.y[idx] if self.y is not None else None,
            weight=self.weight[idx] if self.weight is not None else None,
        )
