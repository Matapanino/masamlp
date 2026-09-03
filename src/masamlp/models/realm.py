"""RealM — the RealMLP backbone under TabM-style **full BatchEnsemble**.

Two published ideas, composed:

* **RealMLP** (Holzmüller et al. 2024, arXiv:2407.04491) supplies the
  backbone — a learnable diagonal scaling layer, neural-tangent-parametrized
  linear layers, parametric activations, flat_cos-scheduled dropout, PBLD
  numeric embeddings, and the data-driven ``std+he5`` init.
* **TabM** (Gorishniy et al. 2024, arXiv:2410.24210) supplies the
  ensembling — ``k`` members share one weight matrix ``W`` per layer and
  differ only through rank-1 adapters, a per-member bias and a per-member
  output head. Members train as independent predictors (the trainer averages
  the per-member losses) and predictions are averaged on the prediction
  scale.

:mod:`masamlp.models.tabm` already implements *TabM-mini*: a single
multiplicative adapter on the shared embedding, over a plain ReLU MLP. RealM
is the **full** BatchEnsemble variant instead — every layer carries its own
input adapter ``r`` and output adapter ``s`` — over the RealMLP trunk, so
member representations keep diverging with depth rather than being fixed at
the input. The point of the architecture is that diversity is **learned
jointly** on a shared feature extractor, where the outer ``n_ens`` axis buys
diversity with k independent fits and k times the compute.

Per layer, the member-``i`` weight is ``W ⊙ (s_i r_iᵀ)``, but it is **never
materialized**: the forward pass broadcasts the adapters around one shared
matmul::

    h = h * r                     # (n, k, d_in), broadcast over rows
    h = (h @ W) / sqrt(d_in)      # one matmul for every member
    h = h * s + b                 # (n, k, d_out), broadcast over rows

``k`` is fixed in the config and every shape is static: no ``torch.vmap``, no
Python loop over members, one XLA graph per batch shape.

Contract notes (masaMLP, ADR 0005):
- ``forward`` returns per-member outputs ``(n, k, out_dim)``. The trainer
  flattens members into rows for the loss, so the loss is the **mean of the
  per-member losses**, and ``apply_transform`` averages members on the
  prediction scale — which is also what the per-epoch validation metric and
  therefore the (single, **joint**) best epoch are computed on.
- ``output_layer`` is the per-member head; its ``(k, out_dim)`` bias receives
  the ``(out_dim,)`` class-prior init by broadcast.

Shapes and memory: one ``(n, k, d)`` activation at ``n=2048, k=32, d=384`` is
25.2M values (~96 MiB fp32). Parameter memory stays near one backbone, but
**compute is ~k member-forwards** — start at ``k=8``/``16``. See
``docs/realm.md``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from masamlp.core.trainer import flat_cos
from masamlp.models.base import FeatureEmbedding
from masamlp.models.layers import ScalingLayer
from masamlp.models.realmlp import (
    _ACTIVATIONS,
    _INIT_MODES,
    ParametricActivation,
    ScheduledDropout,
    _heplus_bias,
)

#: ``member_init`` values for the **first** adapter — the one that turns the
#: shared embedding into k different member representations.  ``"auto"``
#: follows TabM's reference rule (random signs without a numeric embedding,
#: normal with one); ``"ones"`` is the degenerate parity mode: every member
#: starts *and stays* identical, which is only meaningful at ``k=1`` (where
#: it makes RealM bit-identical to ``realmlp``) or as a control.
_MEMBER_INITS = ("auto", "normal", "signs", "ones")


class BatchEnsembleLinear(nn.Module):
    """One BatchEnsemble layer in the neural-tangent parametrization.

    ``weight`` ``(d_in, d_out)`` is **shared** by all ``k`` members; each
    member owns an input adapter ``r`` ``(k, d_in)``, an output adapter ``s``
    ``(k, d_out)`` and a bias ``(k, d_out)``, so member ``i`` computes
    ``((x_i ⊙ r_i) @ W)/sqrt(d_in) ⊙ s_i + b_i`` — algebraically
    ``x_i @ (W ⊙ (r_i s_iᵀ))/sqrt(d_in) + b_i`` without ever forming the
    ``k`` weight matrices.

    The draws mirror :class:`~masamlp.models.realmlp.NTPLinear` exactly —
    ``randn(d_in, d_out)`` then ``randn(d_out)``, same shapes, same order —
    and ``r``/``s`` start at one with the bias copied to every member, so a
    ``k=1`` layer is bit-identical to the NTP layer it replaces.
    """

    def __init__(
        self, in_features: int, out_features: int, k: int, zero_init: bool = False
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.k = k
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        # One draw, copied to every member: at init the members differ only
        # through the first adapter, exactly as TabM's reference does.
        bias = factor * torch.randn(out_features)
        self.bias = nn.Parameter(bias.expand(k, out_features).clone())
        self.r = nn.Parameter(torch.ones(k, in_features))
        self.s = nn.Parameter(torch.ones(k, out_features))

    def forward(self, x: Tensor) -> Tensor:
        """``(n, k, d_in)`` -> ``(n, k, d_out)``."""
        pre = (1.0 / math.sqrt(self.in_features)) * ((x * self.r) @ self.weight)
        return pre * self.s + self.bias

    @torch.no_grad()
    def data_init_(self, x: Tensor) -> Tensor:
        """pytabkit's ``std`` weight + ``he+5`` bias init on ``(n, k, d_in)``
        activations, returning this layer's output so the caller can walk the
        trunk.

        The shared weight is fitted on the members **flattened into rows**
        ``(n*k, d_in)``: one weight serves every member, so its per-unit
        std must be estimated over all member representations, not over one.
        The resulting ``(d_out,)`` bias is copied to every member.
        """
        n, k, d_in = x.shape
        flat = (x * self.r).reshape(n * k, d_in)
        w = torch.randn_like(self.weight)
        pre = (flat @ w) / math.sqrt(d_in)
        std = pre.std(dim=0, correction=0, keepdim=True).clamp_min(1e-30)
        self.weight.copy_(w / std)
        pre = pre / std
        self.bias.copy_(_heplus_bias(pre))          # (d_out,) -> (k, d_out)
        return pre.reshape(n, k, -1) * self.s + self.bias


class EnsembleNTPHead(nn.Module):
    """``k`` independent NTP output heads over shared per-member features:
    ``(n, k, d)`` -> ``(n, k, out)`` via ``weight`` ``(k, d, out)``.

    Unlike :class:`~masamlp.models.tabm.EnsembleHead` (uniform init, no NTP
    scaling) this is the ensemble form of
    :class:`~masamlp.models.realmlp.NTPLinear`, down to the draw order, so a
    ``k=1`` head is bit-identical to RealMLP's output layer.
    """

    def __init__(self, d: int, out_dim: int, k: int, zero_init: bool = False) -> None:
        super().__init__()
        self.d = d
        self.k = k
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(k, d, out_dim))
        bias = factor * torch.randn(out_dim)
        self.bias = nn.Parameter(bias.expand(k, out_dim).clone())

    def forward(self, x: Tensor) -> Tensor:
        pre = torch.einsum("nkd,kdo->nko", x, self.weight)
        return (1.0 / math.sqrt(self.d)) * pre + self.bias

    @torch.no_grad()
    def data_init_(self, x: Tensor) -> Tensor:
        """``std`` + ``he+5`` on the flattened ``(n*k, d)`` activations, with
        the fitted ``(d, out)`` weight and ``(out,)`` bias copied to every
        member — the heads then diverge through training, not through k
        independent draws."""
        n, k, d = x.shape
        flat = x.reshape(n * k, d)
        w = torch.randn(d, self.weight.shape[-1], device=x.device, dtype=x.dtype)
        pre = (flat @ w) / math.sqrt(d)
        std = pre.std(dim=0, correction=0, keepdim=True).clamp_min(1e-30)
        self.weight.copy_((w / std).expand(k, -1, -1))
        pre = pre / std
        self.bias.copy_(_heplus_bias(pre))
        return pre.reshape(n, k, -1) + self.bias


class RealMNet(nn.Module):
    """RealMLP under full BatchEnsemble — see the module docstring.

    Every knob of :class:`~masamlp.models.realmlp.RealMLPNet` carries over
    with the same meaning; ``k``, ``member_init``, ``adapter_std`` and
    ``adapter_lr_factor`` are the ensembling half.
    """

    #: ``ens_mode="vectorized"`` stacks whole models and vmaps them — the
    #: wrong axis for an architecture that already vectorizes its members,
    #: and it freezes ``ScheduledDropout``'s non-persistent schedule buffer.
    #: The outer ``n_ens`` axis still composes in the default loop mode.
    supports_vectorized = False
    supports_member_batches = True

    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
        k: int = 8,
        hidden_sizes: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "mish",
        dropout: float = 0.0,
        dropout_schedule: str = "none",
        use_parametric_act: bool = False,
        act_lr_factor: float = 0.1,
        plr_lr_factor: float = 1.0,
        init_mode: str = "ntp",
        zero_init_output: bool = True,
        scale_position: str = "input",
        scale_lr_factor: float = 6.0,
        first_layer_lr_factor: float = 1.0,
        bias_lr_factor: float = 0.1,
        member_init: str = "auto",
        adapter_std: float = 0.5,
        adapter_lr_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f"k must be a positive int, got {k!r}")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {activation!r}. Expected {set(_ACTIVATIONS)}")
        if dropout_schedule not in ("none", "flat_cos"):
            raise ValueError(f"Unknown dropout_schedule {dropout_schedule!r}")
        if init_mode not in _INIT_MODES:
            raise ValueError(f"Unknown init_mode {init_mode!r}. Expected one of {_INIT_MODES}")
        if scale_position not in ("input", "first_layer"):
            raise ValueError(
                f"Unknown scale_position {scale_position!r}. Expected 'input' or 'first_layer'"
            )
        if member_init not in _MEMBER_INITS:
            raise ValueError(
                f"Unknown member_init {member_init!r}. Expected one of {_MEMBER_INITS}"
            )
        self.k = k
        self.embedding = embedding
        self.dropout_schedule = dropout_schedule
        self.act_lr_factor = act_lr_factor
        self.plr_lr_factor = plr_lr_factor
        self.init_mode = init_mode
        self.needs_data_init = init_mode != "ntp"
        self.zero_init_output = zero_init_output
        self.scale_position = scale_position
        self.scale_lr_factor = scale_lr_factor
        self.first_layer_lr_factor = first_layer_lr_factor
        self.bias_lr_factor = bias_lr_factor
        self.adapter_lr_factor = adapter_lr_factor
        # TabM's reference rule: random signs when the numeric features enter
        # raw, normal when they go through a numeric embedding first (which
        # PBLD does), because an embedded feature block is already a learned,
        # sign-sensitive representation.
        if member_init == "auto":
            member_init = "normal" if embedding.num_embedding is not None else "signs"
        self.member_init = member_init
        self.adapter_std = adapter_std
        self.front_scale: nn.Module | None = None
        if scale_position == "first_layer" and embedding.scaling is not None:
            embedding.scaling = None
        # The first adapter is where one shared representation becomes k
        # different ones; it sits AFTER the embedding and the front scale and
        # BEFORE the first feature-mixing layer.
        self.first_adapter = nn.Parameter(
            self._init_first_adapter(k, embedding.d_out, member_init, adapter_std)
        )
        fn = _ACTIVATIONS[activation]
        act_cls = {"mish": nn.Mish, "selu": nn.SELU, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        d_in = embedding.d_out
        for width in hidden_sizes:
            layers.append(BatchEnsembleLinear(d_in, width, k))
            layers.append(ParametricActivation(width, fn) if use_parametric_act else act_cls())
            if dropout > 0:
                # Masks are drawn over the whole (n, k, d) tensor, so members
                # see independent dropout — the regularizer decorrelates them.
                layers.append(
                    ScheduledDropout(dropout)
                    if dropout_schedule == "flat_cos"
                    else nn.Dropout(dropout)
                )
            d_in = width
        self.trunk = nn.Sequential(*layers)
        self.output_layer = EnsembleNTPHead(d_in, out_dim, k, zero_init=zero_init_output)
        if scale_position == "first_layer":
            self.front_scale = ScalingLayer(embedding.d_out)

    @staticmethod
    def _init_first_adapter(k: int, d: int, member_init: str, std: float) -> Tensor:
        if member_init == "ones":
            return torch.ones(k, d)
        if member_init == "signs":
            return torch.randint(0, 2, (k, d), dtype=torch.float32) * 2.0 - 1.0
        return torch.empty(k, d).normal_(1.0, std)

    def set_schedule_t(self, t: float) -> None:
        """Trainer hook: update scheduled dropout with training progress."""
        if self.dropout_schedule == "flat_cos":
            factor = flat_cos(t)
            for module in self.modules():
                if isinstance(module, ScheduledDropout):
                    module.set_factor(factor)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        member_batches = x_num.ndim == 3
        if member_batches:
            n, k = x_num.shape[:2]
            if k != self.k or x_cat.ndim != 3 or x_cat.shape[:2] != (n, k):
                raise ValueError(f"member batches must have leading shape (n, {self.k})")
            h = self.embedding(x_num.flatten(0, 1), x_cat.flatten(0, 1)).unflatten(
                0, (n, k)
            )
        else:
            h = self.embedding(x_num, x_cat)         # (n, e)   shared
        if self.front_scale is not None:
            h = self.front_scale(h)                  # (n, e)   shared
        if not member_batches:
            h = h.unsqueeze(1)
        h = h * self.first_adapter                    # (n, k, e) per member
        return self.output_layer(self.trunk(h))      # (n, k, out_dim)

    @torch.no_grad()
    def data_init(self, x_num: Tensor, x_cat: Tensor) -> None:
        """Estimator hook: pytabkit's data-driven RealMLP-TD init, walked over
        the **member** activations (see :meth:`BatchEnsembleLinear.data_init_`).
        A no-op under the default ``init_mode="ntp"``."""
        if self.init_mode != "std+he5":
            return
        was_training = self.training
        self.eval()                      # dropout must be identity during the walk
        h = self.embedding(x_num, x_cat)
        if self.front_scale is not None:
            h = self.front_scale(h)
        h = h.unsqueeze(1) * self.first_adapter
        for module in self.trunk:
            h = module.data_init_(h) if isinstance(module, BatchEnsembleLinear) else module(h)
        if not self.zero_init_output:
            self.output_layer.data_init_(h)
        if was_training:
            self.train()

    def param_groups(self) -> list[dict]:
        """RealMLP's per-group pricing, plus one group for the ensembling
        parameters.

        Unchanged from :meth:`RealMLPNet.param_groups
        <masamlp.models.realmlp.RealMLPNet.param_groups>`: the scaling layer
        at ``scale_lr_factor``, shared weights at 1x (times
        ``first_layer_lr_factor`` on the first layer), parametric activations
        at ``act_lr_factor``, numeric embeddings at ``plr_lr_factor``, and no
        weight decay on any bias.

        New: every **per-member** parameter — the first adapter, each layer's
        ``r``/``s``, and the head weight — forms one group at
        ``adapter_lr_factor``. The per-member biases keep RealMLP's bias
        treatment (``bias_lr_factor``, zero weight decay) with
        ``adapter_lr_factor`` applied on top, so at ``adapter_lr_factor=1.0``
        every group's effective (lr, wd) factor equals RealMLP's.
        """
        layers = [m for m in self.trunk if isinstance(m, BatchEnsembleLinear)]
        first = layers[0] if layers else None
        rest = layers[1:]
        af = self.adapter_lr_factor
        adapters = [self.first_adapter, self.output_layer.weight]
        for m in layers:
            adapters += [m.r, m.s]
        weights = [m.weight for m in rest]
        biases = [m.bias for m in rest] + [self.output_layer.bias]
        scale = (
            list(self.embedding.scaling.parameters())
            if self.embedding.scaling is not None
            else []
        )
        if self.front_scale is not None:
            scale = scale + list(self.front_scale.parameters())
        act = [m.alpha for m in self.modules() if isinstance(m, ParametricActivation)]
        plr = (
            list(self.embedding.num_embedding.parameters())
            if self.embedding.num_embedding is not None
            else []
        )
        first_w = [first.weight] if first is not None else []
        first_b = [first.bias] if first is not None else []
        assigned = {
            id(p) for p in adapters + weights + biases + scale + act + plr + first_w + first_b
        }
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in assigned]
        ff = self.first_layer_lr_factor
        bf = self.bias_lr_factor * af
        groups = [
            {"params": scale, "lr_factor": self.scale_lr_factor},
            *(
                [
                    {"params": first_w, "lr_factor": ff},
                    {"params": first_b, "lr_factor": bf * ff, "wd_factor": 0.0},
                    {"params": weights, "lr_factor": 1.0},
                    {"params": biases, "lr_factor": bf, "wd_factor": 0.0},
                ]
                if ff != 1.0
                else [
                    {"params": first_w + weights, "lr_factor": 1.0},
                    {"params": first_b + biases, "lr_factor": bf, "wd_factor": 0.0},
                ]
            ),
            {"params": adapters, "lr_factor": af},
            {"params": act, "lr_factor": self.act_lr_factor},
            {"params": plr, "lr_factor": self.plr_lr_factor},
            {"params": other, "lr_factor": 1.0},
        ]
        return [g for g in groups if g["params"]]
