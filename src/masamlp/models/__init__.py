"""Model registry.

Third-party architectures plug into the estimators the same way built-ins
do: ``register_model("name")(MyModule)`` where ``MyModule.__init__`` accepts
``(embedding: FeatureEmbedding, out_dim: int, **model_params)`` and
``forward`` takes ``(x_num, x_cat)`` and returns raw ``(n, out_dim)``
outputs. An ``output_layer`` attribute (the final ``nn.Linear``) is optional
but enables head-bias initialization at the target's optimum. Token-based
models (attention over per-feature embeddings) instead set a class attribute
``embedding_kind = "tokens"`` and accept ``(embedding_config: dict, out_dim,
**model_params)``, constructing a :class:`~masamlp.models.base.TokenEmbedding`
from the config with their own token width.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from torch import nn

from masamlp.models.base import (
    FeatureEmbedding,
    LinearTokens,
    PeriodicEmbedding,
    PLREmbedding,
    TokenEmbedding,
)
from masamlp.models.danet import DANet
from masamlp.models.ft_transformer import FTTransformer
from masamlp.models.gandalf import GandalfNet, GatedFeatureLearningUnit
from masamlp.models.grn import GatedResidualBlock, GRNNet
from masamlp.models.layers import (
    GhostBatchNorm1d,
    ScalingLayer,
    entmax15,
    sparsemax,
    t_softmax,
)
from masamlp.models.lnn import CfCCell, TabularLNN
from masamlp.models.modernnca import ModernNCA
from masamlp.models.realmlp import NTPLinear, RealMLPNet
from masamlp.models.resnet import TabularResNet
from masamlp.models.tab_transformer import TabTransformer
from masamlp.models.tabm import EnsembleHead, TabM
from masamlp.models.tabr import TabR

_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}

# FeatureEmbedding options accepted inside model_params for every model.
_EMBEDDING_KEYS = ("d_num_embedding", "n_frequencies", "sigma", "cat_emb_dim", "num_scaling")

# Constructor parameters filled in by build_model, not user-facing knobs.
_NON_PARAM_KEYS = frozenset({"self", "embedding", "embedding_config", "out_dim", "n_label_classes"})


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    def decorator(builder: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model {name!r} is already registered")
        _MODEL_REGISTRY[name] = builder
        return builder

    return decorator


register_model("resnet")(TabularResNet)
register_model("danet")(DANet)
register_model("lnn")(TabularLNN)
register_model("realmlp")(RealMLPNet)
register_model("tabr")(TabR)
register_model("ft_transformer")(FTTransformer)
register_model("tab_transformer")(TabTransformer)
register_model("modernnca")(ModernNCA)
register_model("gandalf")(GandalfNet)
register_model("grn")(GRNNet)
register_model("tabm")(TabM)


def _check_model_params(name: str, builder: Callable[..., nn.Module], params: dict) -> None:
    """Reject unknown model_params up front with the valid keys spelled out,
    instead of the bare TypeError the constructor would raise."""
    signature = inspect.signature(builder).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.values()):
        return
    unknown = sorted(k for k in params if k not in signature)
    if unknown:
        valid = sorted(k for k in signature if k not in _NON_PARAM_KEYS)
        raise ValueError(
            f"Unknown model_params {unknown} for model {name!r}. Valid keys: {valid}, "
            f"plus the shared embedding keys {sorted(_EMBEDDING_KEYS)}. "
            "See docs/parameters.md."
        )


def build_model(
    name: str,
    model_params: dict[str, object] | None,
    n_num: int,
    cat_cardinalities: list[int],
    out_dim: int,
    num_embedding: str | None = None,
) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_MODEL_REGISTRY)}")
    builder = _MODEL_REGISTRY[name]
    params = dict(model_params or {})
    embed_kwargs = {k: params.pop(k) for k in _EMBEDDING_KEYS if k in params}
    _check_model_params(name, builder, params)
    if getattr(builder, "embedding_kind", "flat") == "tokens":
        config = {
            "n_num": n_num,
            "cat_cardinalities": cat_cardinalities,
            "num_embedding": num_embedding,
            **embed_kwargs,
        }
        return builder(embedding_config=config, out_dim=out_dim, **params)
    embedding = FeatureEmbedding(
        n_num, cat_cardinalities, num_embedding=num_embedding, **embed_kwargs
    )
    return builder(embedding=embedding, out_dim=out_dim, **params)


__all__ = [
    "FeatureEmbedding",
    "TokenEmbedding",
    "LinearTokens",
    "PeriodicEmbedding",
    "PLREmbedding",
    "GhostBatchNorm1d",
    "ScalingLayer",
    "sparsemax",
    "entmax15",
    "TabularResNet",
    "DANet",
    "TabularLNN",
    "CfCCell",
    "RealMLPNet",
    "NTPLinear",
    "TabR",
    "FTTransformer",
    "TabTransformer",
    "TabM",
    "EnsembleHead",
    "ModernNCA",
    "GandalfNet",
    "GatedFeatureLearningUnit",
    "GRNNet",
    "GatedResidualBlock",
    "t_softmax",
    "register_model",
    "build_model",
]
