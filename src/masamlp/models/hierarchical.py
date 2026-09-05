"""WP3 (2026-09-05): smooth numeric embeddings plus hierarchical random effects.

Hierarchy coordinates are in the fitted preprocessor's numeric space, BEFORE
learnable scaling. Build support and boundaries inside the fitting boundary.
The existing numeric embedding is the smooth root; coarse-to-fine zero-start
residuals modify its token without increasing the RealMLP trunk width.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from masamlp.core.training_terms import ModelRegularizer, TrainingTerms
from masamlp.models.base import FeatureEmbedding
from masamlp.models.realmlp import RealMLPNet


class ResidualLevel(nn.Module):
    """One supported interval or exact-value table; unknown entries contribute zero."""

    def __init__(self, spec: dict, width: int, support_power: float) -> None:
        super().__init__()
        self.exact = "values" in spec
        key = "values" if self.exact else "boundaries"
        if set(spec) != {key, "counts"}:
            raise ValueError("a level needs counts and exactly one of values or boundaries")
        points = torch.as_tensor(spec[key], dtype=torch.float32)
        counts = torch.as_tensor(spec["counts"], dtype=torch.float32)
        size = len(points) if self.exact else len(points) + 1
        if (points.ndim != 1 or counts.ndim != 1 or len(counts) != size or size < 1
                or not torch.isfinite(points).all() or not torch.isfinite(counts).all()
                or (points[1:] <= points[:-1]).any() or (counts < 0).any()
                or not (counts > 0).any()):
            raise ValueError("level coordinates must increase with valid nonnegative counts")
        self.register_buffer("points", points)
        self.register_buffer("counts", counts)
        self.register_buffer("supported", counts > 0)
        precision = torch.where(counts > 0, counts.clamp_min(1).pow(-support_power), 0.)
        self.register_buffer("precision", precision)
        self.weight = nn.Parameter(torch.zeros(size, width))  # no RNG consumption
        self.penalized_size = int((counts > 0).sum()) * width

    def forward(self, x: Tensor) -> Tensor:
        # Static-shape gathers, no host branch on device data. Nonfinite values
        # never enter a residual; the root's usual imputation remains external.
        index = torch.searchsorted(self.points, x.contiguous(), right=not self.exact)
        if self.exact:
            index = index.clamp_max(len(self.points) - 1)
            known = x == self.points[index]
        else:
            known = torch.ones_like(x, dtype=torch.bool)
        known = known & torch.isfinite(x) & self.supported[index]
        return F.embedding(index, self.weight) * known.unsqueeze(-1)


class HierarchicalEmbedding(nn.Module):
    """Decorate the existing embedding, preserving all backbone output coordinates."""

    def __init__(self, base: FeatureEmbedding, hierarchies: list[dict],
                 support_power: float) -> None:
        super().__init__()
        self.base = base
        self.n_num, self.d_out = base.n_num, base.d_out
        self.residuals = nn.ModuleList()
        self.routes: list[tuple[int, int, int]] = []
        if base.num_embedding is None:
            raise ValueError("hierarchical embeddings require a smooth numeric embedding")
        embedded = (list(range(base.n_num)) if base.num_embedding_idx is None
                    else base.num_embedding_idx)
        seen = set()
        for h in hierarchies:
            if set(h) != {"feature", "levels"}:
                raise ValueError("each hierarchy needs feature and levels")
            feature = h["feature"]
            if (isinstance(feature, bool) or not isinstance(feature, int)
                    or feature not in embedded or feature in seen):
                raise ValueError("hierarchy feature must uniquely name an embedded numeric column")
            seen.add(feature)
            slot = embedded.index(feature)
            offset = sum(base.feature_chunk_sizes[:slot])
            width = base.feature_chunk_sizes[slot]
            levels = h["levels"]
            if not levels or "values" not in levels[-1]:
                raise ValueError("a hierarchy must finish with an exact values level")
            previous = None
            for j, level in enumerate(levels):
                if j < len(levels) - 1 and "boundaries" not in level:
                    raise ValueError("only the final hierarchy level may use exact values")
                table = ResidualLevel(level, width, support_power)
                if not table.exact:
                    points = set(table.points.tolist())
                    if previous is not None and not previous.issubset(points):
                        raise ValueError("coarse boundaries must be contained in finer boundaries")
                    previous = points
                self.residuals.append(table)
                self.routes.append((feature, offset, width))
        if not self.residuals:
            raise ValueError("hierarchies must contain at least one feature")
        self.penalty_normalizer = sum(t.penalized_size for t in self.residuals)

    @property
    def scaling(self):
        return self.base.scaling

    @property
    def num_embedding(self):
        return self.base.num_embedding

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        out = self.base(x_num, x_cat)
        for table, (feature, offset, width) in zip(self.residuals, self.routes, strict=True):
            out = out + F.pad(table(x_num[:, feature]), (offset, self.d_out - offset - width))
        return out


class HierarchicalRealMLPNet(RealMLPNet):
    """RealMLP with final-parameter hierarchical shrinkage through WP2.

    ``backbone_params`` takes ordinary RealMLP constructor options. Embedding
    options remain shared top-level model_params. ``shrinkage=0`` is the free
    residual-table control; residuals always have zero optimizer weight decay.
    """

    def __init__(self, embedding: FeatureEmbedding, out_dim: int,
                 hierarchies: list[dict], shrinkage: float = 1.,
                 support_power: float = 1., backbone_params: dict | None = None) -> None:
        if not math.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        if not math.isfinite(support_power) or support_power < 0:
            raise ValueError("support_power must be finite and nonnegative")
        super().__init__(embedding, out_dim, **(backbone_params or {}))
        self.shrinkage = shrinkage
        self.embedding = HierarchicalEmbedding(embedding, hierarchies, support_power)

    def param_groups(self) -> list[dict]:
        tables = [t.weight for t in self.embedding.residuals]
        table_ids = {id(p) for p in tables}
        groups = [{**g, "params": [p for p in g["params"] if id(p) not in table_ids]}
                  for g in super().param_groups()]
        return [g for g in groups if g["params"]] + [
            {"params": tables, "lr_factor": 1., "wd_factor": 0.},
        ]

    def training_terms(self, batch, raw: Tensor) -> TrainingTerms:
        if self.shrinkage == 0:
            return TrainingTerms()
        penalty = sum((t.weight.float().square() * t.precision[:, None]).sum()
                      for t in self.embedding.residuals)
        return TrainingTerms(regularizers=(ModelRegularizer(
            penalty, normalizer=self.embedding.penalty_normalizer, coefficient=self.shrinkage,
        ),))
