"""RealMLP — Holzmüller et al. 2024, "Better by Default: Strong Pre-Tuned
MLPs and Boosted Trees on Tabular Data" (arXiv:2407.04491).

The architecture follows the author's MIT-licensed standalone TD-S reference
(dholzmueller/realmlp-td-s_standalone): a learnable per-feature scaling
layer (via ``FeatureEmbedding``'s ``num_scaling``), neural-tangent-
parametrized linear layers, SELU/Mish activations, and a zero-initialized
output layer. The RealMLP-TD extras (per pytabkit's Apache-2.0
implementation) are available as options: parametric activations
(``x + (act(x) - x) * alpha`` with per-unit learnable ``alpha``, trained at
``act_lr_factor``), flat_cos-scheduled dropout, PBLD numeric embeddings with
their own learning-rate factor, and — at the trainer level — flat_cos
weight decay with zero decay on biases.

The matching training recipes live in :mod:`masamlp.presets`
(``realmlp_params`` for TD-S, ``realmlp_td_params`` for TD).

Since 0.9.0 the remaining TD fidelity knobs are options, all defaulting to
the 0.8.0 behaviour: ``init_mode="std+he5"`` for pytabkit's data-driven
layerwise init, ``zero_init_output=False`` for TD's non-zero output layer,
``scale_position="first_layer"`` to put the diagonal gain on the first
layer's input (where TD puts it) rather than on the raw numerics (where
TD-S puts it), and explicit ``scale_lr_factor`` / ``first_layer_lr_factor``
/ ``bias_lr_factor``.

0.9.1 adds the opt-in **input routing** half of the architecture:
``linear_skip_idx`` puts a zero-initialized linear map from a subset of the
numeric inputs straight onto the logits
(``raw = trunk(x) + x_skip @ W_skip + b_skip``), trained in its own param
group at ``linear_skip_lr_factor`` with no weight decay. Its counterpart on
the input side is :class:`~masamlp.models.base.FeatureEmbedding`'s
``num_embedding_idx``, which restricts the numeric embedding to a subset of
the columns. Both default to ``None`` = the 0.9.0 model, byte for byte.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn

from masamlp.core.trainer import flat_cos
from masamlp.models.base import FeatureEmbedding, _resolve_column_idx
from masamlp.models.layers import ScalingLayer

_ACTIVATIONS: dict[str, Callable[[Tensor], Tensor]] = {
    "mish": torch.nn.functional.mish,
    "selu": torch.selu,
    "relu": torch.relu,
}


#: ``init_mode`` values.  ``"ntp"`` is the standalone TD-S reference's
#: ``N(0, 1)`` draw for both weight and bias — masaMLP's behaviour through
#: 0.8.0.  ``"std+he5"`` is pytabkit's RealMLP-**TD** default
#: (``weight_init_mode="std"``, ``bias_init_mode="he+5"``): a sequential,
#: layer-by-layer, *data-driven* init that needs a batch of training rows.
_INIT_MODES = ("ntp", "std+he5")


def _heplus_bias(pre: Tensor, n_simplex: int = 5) -> Tensor:
    """pytabkit's ``he+5`` bias: for every unit, minus a random Exp(1)-simplex
    convex combination of ``n_simplex`` sampled pre-activations of that unit.

    This puts each unit's activation threshold at a random data point, so
    roughly half the units start on the active side of the nonlinearity.
    """
    n_rows, n_units = pre.shape
    idx = torch.randint(0, n_rows, (n_units, n_simplex), device=pre.device)
    w = torch.distributions.Exponential(1.0).sample((n_units, n_simplex)).to(pre.device)
    w = w / w.sum(dim=1, keepdim=True)
    cols = torch.arange(n_units, device=pre.device)
    picked = torch.stack([pre[idx[:, i], cols] for i in range(n_simplex)], dim=1)
    return -(picked * w.to(pre.dtype)).sum(dim=1)


class NTPLinear(nn.Module):
    """Linear layer in the neural-tangent parametrization: weights are drawn
    from N(0, 1) and the forward pass scales by ``1/sqrt(in_features)``,
    which shifts the effective per-layer learning rates.

    Construction always uses the ``"ntp"`` draw (``W, b ~ N(0, 1)``).  The
    data-driven ``"std+he5"`` init is applied afterwards by
    :meth:`RealMLPNet.data_init`, which needs the activations reaching this
    layer and therefore cannot run at construction time.
    """

    def __init__(self, in_features: int, out_features: int, zero_init: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        factor = 0.0 if zero_init else 1.0
        self.weight = nn.Parameter(factor * torch.randn(in_features, out_features))
        self.bias = nn.Parameter(factor * torch.randn(out_features))

    def forward(self, x: Tensor) -> Tensor:
        return (1.0 / math.sqrt(self.in_features)) * (x @ self.weight) + self.bias

    @torch.no_grad()
    def data_init_(self, x: Tensor) -> Tensor:
        """pytabkit's ``std`` weight + ``he+5`` bias init, given the batch of
        activations ``x`` that reaches this layer.  Returns this layer's
        output on ``x`` so the caller can walk the trunk.

        ``std``: redraw ``W ~ N(0, 1)`` and divide each output column by the
        sample std of that column's pre-activation, so every unit's
        pre-activation has **unit std** on the init batch
        (``pytabkit/models/nn_models/nn.py:166-169``).
        """
        w = torch.randn_like(self.weight)
        pre = (x @ w) / math.sqrt(self.in_features)
        std = pre.std(dim=0, correction=0, keepdim=True).clamp_min(1e-30)
        self.weight.copy_(w / std)
        pre = pre / std
        self.bias.copy_(_heplus_bias(pre))
        return pre + self.bias


class ParametricActivation(nn.Module):
    """RealMLP-TD: ``x + (act(x) - x) * alpha`` with per-unit learnable
    ``alpha`` (init 1 — plain activation), so each unit can interpolate
    between identity and the nonlinearity."""

    def __init__(self, n_units: int, fn: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.ones(n_units))

    def forward(self, x: Tensor) -> Tensor:
        return x + (self.fn(x) - x) * self.alpha


class ScheduledDropout(nn.Module):
    """Dropout whose probability is scaled by a factor the trainer updates
    per step (RealMLP-TD's flat_cos dropout schedule).

    The probability lives in a 0-dim buffer (non-persistent — absent from
    state_dicts, moves with ``.to()``): ``F.dropout`` takes ``p`` as an op
    attribute, so a per-step Python float would bake a new constant into
    every compiled XLA graph and recompile every step; a tensor operand
    keeps one graph on every device."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.base_p = p
        self.register_buffer("_keep", torch.tensor(1.0 - p), persistent=False)

    def set_factor(self, factor: float) -> None:
        self._keep.fill_(1.0 - self.base_p * factor)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        keep = self._keep.to(x.dtype)
        return x * (torch.rand_like(x) < keep) / keep


class RealMLPNet(nn.Module):
    def __init__(
        self,
        embedding: FeatureEmbedding,
        out_dim: int,
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
        linear_skip_idx: list[int] | None = None,
        linear_skip_lr_factor: float = 1.0,
    ) -> None:
        super().__init__()
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
        self.embedding = embedding
        self.dropout_schedule = dropout_schedule
        self.act_lr_factor = act_lr_factor
        self.plr_lr_factor = plr_lr_factor
        self.init_mode = init_mode
        # Estimator hook (sklearn.py): only a model that answers True gets a
        # batch of rows, so every other model's RNG stream is untouched.
        self.needs_data_init = init_mode != "ntp"
        self.zero_init_output = zero_init_output
        self.scale_position = scale_position
        self.scale_lr_factor = scale_lr_factor
        self.first_layer_lr_factor = first_layer_lr_factor
        self.bias_lr_factor = bias_lr_factor
        # RealMLP-TD-S scales the RAW numeric inputs; pytabkit's RealMLP-TD
        # puts its ``add_front_scale`` diagonal gain on the FIRST LAYER's
        # input instead — i.e. after the PBLD numeric embedding and beside
        # the one-hots and categorical embeddings
        # (``pytabkit/models/nn_models/models.py:279-280``).  With
        # ``scale_position="first_layer"`` the embedding's scaling layer is
        # relocated here; ``ScalingLayer`` holds no RNG draw, so moving it
        # leaves every other parameter's initialization byte-unchanged.
        self.front_scale: nn.Module | None = None
        if scale_position == "first_layer" and embedding.scaling is not None:
            embedding.scaling = None
        fn = _ACTIVATIONS[activation]
        act_cls = {"mish": nn.Mish, "selu": nn.SELU, "relu": nn.ReLU}[activation]
        layers: list[nn.Module] = []
        d_in = embedding.d_out
        for width in hidden_sizes:
            layers.append(NTPLinear(d_in, width))
            layers.append(ParametricActivation(width, fn) if use_parametric_act else act_cls())
            if dropout > 0:
                layers.append(
                    ScheduledDropout(dropout)
                    if dropout_schedule == "flat_cos"
                    else nn.Dropout(dropout)
                )
            d_in = width
        self.trunk = nn.Sequential(*layers)
        self.output_layer = NTPLinear(d_in, out_dim, zero_init=zero_init_output)
        if scale_position == "first_layer":
            self.front_scale = ScalingLayer(embedding.d_out)
        # Additive linear skip (0.9.1): a linear map from a subset of the
        # model's numeric inputs straight onto the logits. Zero-initialized,
        # so switching it on draws no random numbers and costs nothing at
        # init — every other parameter stays byte-identical, and the head's
        # bias initialization is unaffected. Selection goes through a fixed
        # int64 buffer (no data-dependent shapes; moves with ``.to()``).
        self.linear_skip_idx = _resolve_column_idx(
            linear_skip_idx, embedding.n_num, "linear_skip_idx"
        )
        self.linear_skip_lr_factor = linear_skip_lr_factor
        self.skip_weight: nn.Parameter | None = None
        self.skip_bias: nn.Parameter | None = None
        if self.linear_skip_idx is not None:
            self.register_buffer(
                "_skip_idx",
                torch.tensor(self.linear_skip_idx, dtype=torch.int64),
                persistent=False,
            )
            self.skip_weight = nn.Parameter(torch.zeros(len(self.linear_skip_idx), out_dim))
            self.skip_bias = nn.Parameter(torch.zeros(out_dim))

    def set_schedule_t(self, t: float) -> None:
        """Trainer hook: update scheduled dropout with training progress."""
        if self.dropout_schedule == "flat_cos":
            factor = flat_cos(t)
            for module in self.modules():
                if isinstance(module, ScheduledDropout):
                    module.set_factor(factor)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        h = self.embedding(x_num, x_cat)
        if self.front_scale is not None:
            h = self.front_scale(h)
        out = self.output_layer(self.trunk(h))
        if self.skip_weight is not None:
            # The skip reads the numeric input as the preprocessor produced
            # it — before the learnable scaling layer and before any numeric
            # embedding — so its weights are plain per-unit output coefficients.
            skip = x_num.index_select(1, self._skip_idx)
            out = out + skip @ self.skip_weight + self.skip_bias
        return out

    @torch.no_grad()
    def data_init(self, x_num: Tensor, x_cat: Tensor) -> None:
        """Estimator hook: run pytabkit's data-driven RealMLP-TD init.

        A no-op under the default ``init_mode="ntp"``.  Under ``"std+he5"``
        the trunk is walked layer by layer on the given batch of *already
        preprocessed* rows: each :class:`NTPLinear` is re-initialized so its
        pre-activations have unit std and its biases sit at random data
        points, using the activations the previous (already re-initialized)
        layers actually produce.  A zero-initialized output layer is left
        alone.
        """
        if self.init_mode != "std+he5":
            return
        was_training = self.training
        self.eval()                      # dropout must be identity during the walk
        h = self.embedding(x_num, x_cat)
        if self.front_scale is not None:
            h = self.front_scale(h)
        for module in self.trunk:
            h = module.data_init_(h) if isinstance(module, NTPLinear) else module(h)
        if not self.zero_init_output:
            self.output_layer.data_init_(h)
        if was_training:
            self.train()

    def param_groups(self) -> list[dict]:
        """RealMLP's per-group hyperparameters: the scaling layer at
        ``scale_lr_factor`` (6x by default), NTP weights 1x, NTP biases
        ``bias_lr_factor`` (0.1x) with no weight decay, parametric
        activations at ``act_lr_factor``, numeric embeddings at
        ``plr_lr_factor``.

        ``first_layer_lr_factor`` multiplies the first trunk layer's weight
        **and** bias factors — as pytabkit does
        (``pytabkit/models/nn_models/models.py:283-284``) — and never the
        scaling layer or the numeric embedding.

        The additive linear skip (``linear_skip_idx``) gets its own group at
        ``linear_skip_lr_factor`` with **zero weight decay**: it is a small,
        directly interpretable linear term whose scale should be set by the
        data, not shrunk by the trunk's regularizer.
        """
        ntp = [m for m in self.modules() if isinstance(m, NTPLinear)]
        first = next((m for m in self.trunk if isinstance(m, NTPLinear)), None)
        rest = [m for m in ntp if m is not first]
        weights = [m.weight for m in rest]
        biases = [m.bias for m in rest]
        scale = (
            list(self.embedding.scaling.parameters())
            if self.embedding.scaling is not None
            else []
        )
        if self.front_scale is not None:
            scale = scale + list(self.front_scale.parameters())
        act = [
            m.alpha for m in self.modules() if isinstance(m, ParametricActivation)
        ]
        plr = (
            list(self.embedding.num_embedding.parameters())
            if self.embedding.num_embedding is not None
            else []
        )
        skip = [self.skip_weight, self.skip_bias] if self.skip_weight is not None else []
        first_w = [first.weight] if first is not None else []
        first_b = [first.bias] if first is not None else []
        assigned = {
            id(p) for p in weights + biases + scale + act + plr + skip + first_w + first_b
        }
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in assigned]
        ff = self.first_layer_lr_factor
        groups = [
            {"params": scale, "lr_factor": self.scale_lr_factor},
            # The first layer keeps its own groups only when it is priced
            # differently; at ff == 1.0 it rejoins the shared groups so the
            # optimizer sees exactly the 0.8.0 group structure.
            *(
                [
                    {"params": first_w, "lr_factor": ff},
                    {"params": first_b, "lr_factor": self.bias_lr_factor * ff,
                     "wd_factor": 0.0},
                    {"params": weights, "lr_factor": 1.0},
                    {"params": biases, "lr_factor": self.bias_lr_factor, "wd_factor": 0.0},
                ]
                if ff != 1.0
                else [
                    {"params": first_w + weights, "lr_factor": 1.0},
                    {"params": first_b + biases, "lr_factor": self.bias_lr_factor,
                     "wd_factor": 0.0},
                ]
            ),
            {"params": act, "lr_factor": self.act_lr_factor},
            {"params": plr, "lr_factor": self.plr_lr_factor},
            {"params": skip, "lr_factor": self.linear_skip_lr_factor, "wd_factor": 0.0},
            {"params": other, "lr_factor": 1.0},
        ]
        return [g for g in groups if g["params"]]
