"""GANDALF — Joseph & Raj 2022, "GANDALF: Gated Adaptive Network for Deep
Automated Learning of Features" (arXiv:2207.08548).

Clean-room reimplementation following the MIT-licensed reference in
pytorch_tabular. The backbone is a Gated Feature Learning Unit (GFLU): each
stage selects features through a learnable sparse mask (t-softmax with a
learnable per-stage temperature, initialized so ~``feature_sparsity`` of the
mass starts near zero; entmax15/sparsemax are drop-in alternatives) and
updates a feature-space hidden state with GRU-style gates. The head is a
single linear layer (the reference's extra learnable output offset ``T0`` is
subsumed by masaMLP's objective-driven bias initialization).

The GFLU operates on the embedded feature vector, so the ``num_embedding``
zoo and categorical embeddings compose with it. The reference batch-norms
raw continuous inputs; masaMLP's preprocessing already scales them, so
``input_batch_norm`` defaults to False (enable it for strict parity).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from masamlp.models.base import FeatureEmbedding
from masamlp.models.layers import entmax15, sparsemax, t_softmax, t_softmax_initial_t

_MASK_FUNCTIONS = ("t_softmax", "entmax15", "sparsemax")


class GatedFeatureLearningUnit(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_stages: int,
        mask_function: str = "t_softmax",
        feature_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mask_function not in _MASK_FUNCTIONS:
            raise ValueError(f"mask_function must be one of {_MASK_FUNCTIONS}")
        self.n_features = n_features
        self.n_stages = n_stages
        self.mask_function = mask_function
        # Beta-distributed mask logits with random concentrations per stage,
        # as in the reference (drawn from the torch RNG so runs are seeded).
        concentrations = 0.5 + 9.5 * torch.rand(n_stages, 2)
        masks = torch.stack(
            [
                torch.distributions.Beta(c0, c1).sample((n_features,))
                for c0, c1 in concentrations
            ]
        )
        self.feature_masks = nn.Parameter(masks)
        self.t = (
            nn.Parameter(
                t_softmax_initial_t(masks, feature_sparsity),
                requires_grad=learnable_sparsity,
            )
            if mask_function == "t_softmax"
            else None
        )
        self.w_in = nn.ModuleList(
            nn.Linear(2 * n_features, 2 * n_features) for _ in range(n_stages)
        )
        self.w_out = nn.ModuleList(
            nn.Linear(2 * n_features, n_features) for _ in range(n_stages)
        )
        self.dropout = nn.Dropout(dropout)

    def masks(self) -> Tensor:
        """The (n_stages, n_features) sparse feature masks."""
        if self.mask_function == "t_softmax":
            return t_softmax(self.feature_masks, torch.relu(self.t), dim=-1)
        fn = entmax15 if self.mask_function == "entmax15" else sparsemax
        return fn(self.feature_masks, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        h = x
        masks = self.masks()
        for stage in range(self.n_stages):
            feature = masks[stage] * x
            gates = self.w_in[stage](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(gates[:, : self.n_features])
            r = torch.sigmoid(gates[:, self.n_features :])
            h_new = torch.tanh(self.w_out[stage](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1.0 - z) * h + z * h_new)
        return h


class GandalfNet(nn.Module):
    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        n_stages: int = 6,
        mask_function: str = "t_softmax",
        feature_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        dropout: float = 0.0,
        input_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.input_norm = nn.BatchNorm1d(embedding.d_out) if input_batch_norm else None
        self.gflu = GatedFeatureLearningUnit(
            embedding.d_out,
            n_stages,
            mask_function=mask_function,
            feature_sparsity=feature_sparsity,
            learnable_sparsity=learnable_sparsity,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(embedding.d_out, out_dim)

    def feature_importances(self) -> Tensor:
        """Mask mass per embedded feature, summed over stages (the
        reference's ``feature_importance_``)."""
        return self.gflu.masks().sum(dim=0).detach()

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        x = self.embedding(x_num, x_cat)
        if self.input_norm is not None:
            x = self.input_norm(x)
        return self.output_layer(self.gflu(x))
