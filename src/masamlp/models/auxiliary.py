"""Masked ordinal auxiliary likelihood with a shared latent (WP4, 2026-09-05).

Routing indices refer to PREPROCESSED numeric/categorical columns. The caller
must explicitly partition every input into parents or excluded columns, including
all descendants of the auxiliary response. This generic module cannot infer
causal provenance from names. A separate categorical target carrier is excluded
from both prediction branches, enabling shuffled-label controls without changing
the observed response available to the primary branch.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from masamlp.core.training_terms import RowTerm, TrainingTerms
from masamlp.data.dataset import TabularData
from masamlp.models.base import FeatureEmbedding
from masamlp.models.realmlp import NTPLinear, RealMLPNet


def _partition(parent, excluded, size, name):
    for values in (parent, excluded):
        if (not isinstance(values, (list, tuple))
                or any(type(i) is not int or not 0 <= i < size for i in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name}: expected unique, in-range integer indices")
    if set(parent) & set(excluded) or set(parent) | set(excluded) != set(range(size)):
        raise ValueError(f"{name}: parents and excluded must partition every input")
    return list(parent)


class AuxiliaryOrdinalNet(nn.Module):
    """Two RealMLP branches; binary logits plus opt-in WP2 ordinal row terms.

    ``embedding_kind='tokens'`` requests the existing configuration-factory
    protocol; this model constructs flat embeddings, not attention tokens.
    The latent joins the primary hidden representation at the binary head.
    Auxiliary dropout is zero so the recomputed hook uses the same latent as
    forward, without storing graphs or consuming extra dropout RNG draws.
    """

    embedding_kind = "tokens"

    def __init__(
        self,
        embedding_config: dict,
        out_dim: int,
        parent_num_idx: list[int],
        parent_cat_idx: list[int],
        excluded_num_idx: list[int],
        excluded_cat_idx: list[int],
        auxiliary_target_cat_idx: int,
        auxiliary_class_order: list[int],
        auxiliary_weight: float = 0.1,
        latent_dim: int = 8,
        auxiliary_hidden_sizes: tuple[int, ...] | list[int] = (32, 32),
        backbone_params: dict | None = None,
    ) -> None:
        super().__init__()
        if out_dim != 1:
            raise ValueError("auxiliary_ordinal requires a binary one-logit primary output")
        if not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
            raise ValueError("auxiliary_weight must be finite and nonnegative")
        if type(latent_dim) is not int or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        config = dict(embedding_config)
        cards = config['cat_cardinalities']
        nums = _partition(parent_num_idx, excluded_num_idx, config['n_num'], 'numeric routing')
        cats = _partition(parent_cat_idx, excluded_cat_idx, len(cards), 'categorical routing')
        if (type(auxiliary_target_cat_idx) is not int
                or auxiliary_target_cat_idx not in excluded_cat_idx):
            raise ValueError("auxiliary target must be an excluded categorical column")
        k = cards[auxiliary_target_cat_idx] - 1
        if (k < 2 or any(type(i) is not int for i in auxiliary_class_order)
                or sorted(auxiliary_class_order) != list(range(1, k + 1))):
            raise ValueError("auxiliary_class_order must order all nonmissing category codes 1..K")
        self.auxiliary_weight = auxiliary_weight
        self.auxiliary_target_cat_idx = auxiliary_target_cat_idx
        self.register_buffer('_parent_num', torch.tensor(nums, dtype=torch.long))
        self.register_buffer('_parent_cat', torch.tensor(cats, dtype=torch.long))
        ev_cats = [i for i in range(len(cards)) if i != auxiliary_target_cat_idx]
        self.register_buffer('_ev_cat', torch.tensor(ev_cats, dtype=torch.long))
        lookup = torch.zeros(k + 1, dtype=torch.long)
        for rank, code in enumerate(auxiliary_class_order):
            lookup[code] = rank
        self.register_buffer('_class_rank', lookup)
        self.auxiliary = RealMLPNet(
            FeatureEmbedding(len(nums), [cards[i] for i in cats], num_scaling=True),
            latent_dim, hidden_sizes=auxiliary_hidden_sizes, activation='selu',
            zero_init_output=False,
        )
        self.ev = RealMLPNet(
            FeatureEmbedding(**{**config, 'cat_cardinalities': [cards[i] for i in ev_cats]}),
            out_dim, **dict(backbone_params or {}),
        )
        if self.ev.linear_skip_idx is not None:
            raise ValueError("auxiliary_ordinal does not support backbone linear_skip_idx")
        self.ev.output_layer = NTPLinear(
            self.ev.output_layer.in_features + latent_dim, 1,
            zero_init=self.ev.zero_init_output,
        )
        self.ordinal_head = nn.Linear(latent_dim, 1, bias=False)
        self.cutpoint_start = nn.Parameter(torch.tensor(0.))
        self.cutpoint_gaps = nn.Parameter(torch.zeros(k - 2))
        self.needs_data_init = self.ev.needs_data_init

    @property
    def output_layer(self):
        return self.ev.output_layer

    def auxiliary_latent(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.auxiliary(x_num.index_select(1, self._parent_num),
                              x_cat.index_select(1, self._parent_cat))

    def _ev_hidden(self, x_num, x_cat):
        h = self.ev.embedding(x_num, x_cat.index_select(1, self._ev_cat))
        if self.ev.front_scale is not None:
            h = self.ev.front_scale(h)
        return self.ev.trunk(h)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        return self.output_layer(torch.cat((self._ev_hidden(x_num, x_cat),
                                            self.auxiliary_latent(x_num, x_cat)), dim=1))

    def cutpoints(self) -> Tensor:
        gaps = F.softplus(self.cutpoint_gaps.float()) + 1e-4
        return torch.cat((self.cutpoint_start.float().reshape(1),
                          self.cutpoint_start.float() + gaps.cumsum(0)))

    def auxiliary_log_probs(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        latent = self.auxiliary_latent(x_num, x_cat)
        # Disable outer AMP for the complete ordinal calculation, including its head.
        with torch.autocast(device_type=latent.device.type, enabled=False):
            score = F.linear(latent.float(), self.ordinal_head.weight.float())
            z = self.cutpoints()[None, :] - score
            gaps = F.softplus(self.cutpoint_gaps.float()) + 1e-4
            # log(sigmoid(upper)-sigmoid(lower)), stable even in saturated tails.
            interior = (F.logsigmoid(z[:, 1:]) + F.logsigmoid(-z[:, :-1])
                        + torch.log(-torch.expm1(-gaps))[None, :])
            return torch.cat((F.logsigmoid(z[:, :1]), interior,
                              F.logsigmoid(-z[:, -1:])), dim=1)

    def training_terms(self, batch: TabularData, raw: Tensor) -> TrainingTerms:
        if self.auxiliary_weight == 0:
            return TrainingTerms()
        codes = batch.x_cat[:, self.auxiliary_target_cat_idx]
        log_probs = self.auxiliary_log_probs(batch.x_num, batch.x_cat)
        loss = -log_probs.gather(1, self._class_rank[codes, None]).squeeze(1)
        # Reserved unknown/missing code 0 has zero auxiliary contribution. The
        # trainer retains its usual denominator over all primary row weights.
        loss = torch.where(codes != 0, loss, torch.zeros_like(loss))
        return TrainingTerms(auxiliary=(RowTerm(loss, self.auxiliary_weight),))

    def set_schedule_t(self, t: float) -> None:
        self.ev.set_schedule_t(t)

    @torch.no_grad()
    def data_init(self, x_num: Tensor, x_cat: Tensor) -> None:
        if not self.needs_data_init:
            return
        was_training = self.training
        self.eval()
        h = self.ev.embedding(x_num, x_cat.index_select(1, self._ev_cat))
        if self.ev.front_scale is not None:
            h = self.ev.front_scale(h)
        for module in self.ev.trunk:
            h = module.data_init_(h) if isinstance(module, NTPLinear) else module(h)
        if not self.ev.zero_init_output:
            self.output_layer.data_init_(torch.cat((h, self.auxiliary_latent(x_num, x_cat)), 1))
        self.train(was_training)

    def param_groups(self) -> list[dict]:
        return self.ev.param_groups() + self.auxiliary.param_groups() + [
            {'params': list(self.ordinal_head.parameters()), 'lr_factor': 1.0},
            {'params': [self.cutpoint_start, self.cutpoint_gaps],
             'lr_factor': 1.0, 'wd_factor': 0.0},
        ]
