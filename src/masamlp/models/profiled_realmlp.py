"""Alternating source-function profiling with a training-orthogonal remainder.

The Trainer refreshes weighted projection sufficient statistics on training X
before each epoch and after restoring prediction weights. Forward uses saved
coefficients: neither query rows nor query batch sizes define the projection.
Orthogonality holds at refreshes, up to the stated pseudoinverse tolerance;
coefficients are held fixed while the next epoch adjusts all live functions.
"""
from __future__ import annotations

import math
from copy import deepcopy

import torch
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding
from masamlp.models.realmlp import RealMLPNet


class ProfiledRealMLPNet(nn.Module):
    """RealMLP source towers plus a dense residual RealMLP with separate embeddings.

    ``joint`` refreshes the remainder's projection onto an intercept and the
    learned source bases. ``unprojected`` keeps the same decomposition without
    subtraction. ``frozen`` freezes the sources after the shared warmup, then
    fits the projected remainder using the ordinary objective with fixed
    source logits. No labels enter projection fitting.
    """

    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        source_groups: list[list[int]] | None = None,
        source_hidden_sizes: tuple[int, ...] | list[int] = (128, 128, 16),
        profile_mode: str = "joint",
        source_warmup_epochs: int = 0,
        projection_rtol: float = 1e-10,
        **realmlp_params,
    ) -> None:
        super().__init__()
        if profile_mode not in ("joint", "unprojected", "frozen"):
            raise ValueError("profile_mode must be joint, unprojected or frozen")
        if not isinstance(source_warmup_epochs, int) or source_warmup_epochs < 0:
            raise ValueError("source_warmup_epochs must be a nonnegative integer")
        if not 0 < projection_rtol < 1:
            raise ValueError("projection_rtol must be in (0, 1)")
        if not source_hidden_sizes or any(w < 1 for w in source_hidden_sizes):
            raise ValueError("source_hidden_sizes must contain positive widths")
        for key in ("tower_groups", "first_layer_groups", "linear_skip_idx"):
            if realmlp_params.get(key) is not None:
                raise ValueError(f"profiled_realmlp does not combine with {key}")
        if source_groups is None:
            source_groups = [list(range(len(embedding.feature_chunk_sizes)))]
        source_embedding = deepcopy(embedding)
        self.remainder = RealMLPNet(embedding, out_dim, **realmlp_params)
        source_params = {**realmlp_params, "hidden_sizes": source_hidden_sizes,
                         "tower_groups": source_groups}
        self.sources = RealMLPNet(source_embedding, out_dim, **source_params)
        self.n_sources = len(source_groups)
        self.basis_width = source_hidden_sizes[-1]
        self.profile_mode = profile_mode
        self.source_warmup_epochs = source_warmup_epochs
        self.projection_rtol = projection_rtol
        self.needs_data_init = self.sources.needs_data_init
        self._source_frozen = False
        self._warmup = source_warmup_epochs > 0
        size = 1 + self.n_sources * self.basis_width
        self.register_buffer("projection", torch.zeros(size, out_dim, dtype=torch.float64))
        self.register_buffer("profile_refreshes", torch.zeros((), dtype=torch.int64))
        # Accumulation state is temporary, excluded from serialization.
        self.register_buffer(
            "_gram", torch.zeros(size, size, dtype=torch.float64), persistent=False)
        self.register_buffer(
            "_rhs", torch.zeros(size, out_dim, dtype=torch.float64), persistent=False)
        self.register_buffer("_active", torch.tensor(not self._warmup))

    @property
    def output_layer(self):
        # The objective's initial prior belongs to the additive sources.
        return self.sources.output_layer

    def param_groups(self):
        return self.remainder.param_groups() + self.sources.param_groups()

    @torch.no_grad()
    def data_init(self, x_num: Tensor, x_cat: Tensor) -> None:
        self.remainder.data_init(x_num, x_cat)
        self.sources.data_init(x_num, x_cat)

    def set_schedule_t(self, t: float) -> None:
        self.remainder.set_schedule_t(t)
        self.sources.set_schedule_t(t)

    def set_training_epoch(self, epoch: int) -> None:
        self._warmup = epoch < self.source_warmup_epochs
        self._active.fill_(not self._warmup)
        if self.profile_mode == "frozen" and not self._warmup:
            self._source_frozen = True
            self.sources.requires_grad_(False)
            self.sources.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self._source_frozen:
            self.sources.eval()
        return self

    def _source_basis(self, x_num: Tensor, x_cat: Tensor):
        h = self.sources.embedding(x_num, x_cat)
        if self.sources.front_scale is not None:
            h = self.sources.front_scale(h)
        if self.sources.towers is None:
            bases = [self.sources.trunk(h)]
        else:
            bases = [tower(h) for tower in self.sources.towers]
        weight = self.sources.output_layer.weight
        contributions = [b @ weight[i*self.basis_width:(i+1)*self.basis_width]
                         / math.sqrt(self.basis_width) for i, b in enumerate(bases)]
        # Split the shared intercept evenly for component diagnostics only.
        source = torch.stack(contributions, dim=1)
        source = source + self.sources.output_layer.bias / self.n_sources
        basis = torch.cat([torch.ones_like(bases[0][:, :1]), *bases], dim=1)
        return source, basis

    def decompose(self, x_num: Tensor, x_cat: Tensor):
        """Return (source contributions, remainder, projection basis)."""
        source, basis = self._source_basis(x_num, x_cat)
        if self.training and self._warmup:
            residual = torch.zeros_like(source[:, 0])
        else:
            raw = self.remainder(x_num, x_cat)
            residual = raw
            if self.profile_mode != "unprojected":
                residual = raw - basis @ self.projection.to(basis.dtype)
            residual = residual * self._active.to(residual.dtype)
        return source, residual, basis

    def forward(self, x_num: Tensor, x_cat: Tensor):
        source, residual, _ = self.decompose(x_num, x_cat)
        return source.sum(dim=1) + residual

    def needs_prediction_state(self) -> bool:
        return self.profile_mode != "unprojected" and not self._warmup

    @torch.no_grad()
    def reset_prediction_state(self) -> None:
        self._gram.zero_()
        self._rhs.zero_()

    @torch.no_grad()
    def update_prediction_state(self, x_num: Tensor, x_cat: Tensor,
                                weight: Tensor | None) -> None:
        _, basis = self._source_basis(x_num, x_cat)
        b = basis.double()
        raw = self.remainder(x_num, x_cat).double()
        weighted = b if weight is None else b * weight.double().reshape(-1, 1)
        self._gram.add_(b.T @ weighted)
        self._rhs.add_(weighted.T @ raw)

    @torch.no_grad()
    def finalize_prediction_state(self) -> None:
        # Double sufficient statistics and a rank-aware Moore-Penrose solve
        # permit duplicate/constant source bases without ridge bias.
        self.projection.copy_(torch.linalg.pinv(
            self._gram, rtol=self.projection_rtol, hermitian=True) @ self._rhs)
        self.profile_refreshes.add_(1)
