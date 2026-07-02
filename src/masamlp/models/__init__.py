"""Model registry.

Third-party architectures plug into the estimators the same way built-ins
do: ``register_model("name")(MyModule)`` where ``MyModule.__init__`` accepts
``(embedding: FeatureEmbedding, out_dim: int, **model_params)``, ``forward``
takes ``(x_num, x_cat)`` and returns raw ``(n, out_dim)`` outputs, and an
``output_layer`` attribute exposes the final ``nn.Linear`` for bias
initialization.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from masamlp.models.base import FeatureEmbedding, PeriodicEmbedding, PLREmbedding
from masamlp.models.danet import DANet
from masamlp.models.layers import GhostBatchNorm1d, ScalingLayer, entmax15, sparsemax
from masamlp.models.lnn import CfCCell, TabularLNN
from masamlp.models.realmlp import NTPLinear, RealMLPNet
from masamlp.models.resnet import TabularResNet
from masamlp.models.tabr import TabR

_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}

# FeatureEmbedding options accepted inside model_params for every model.
_EMBEDDING_KEYS = ("d_num_embedding", "n_frequencies", "sigma", "cat_emb_dim", "num_scaling")


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
    params = dict(model_params or {})
    embed_kwargs = {k: params.pop(k) for k in _EMBEDDING_KEYS if k in params}
    embedding = FeatureEmbedding(
        n_num, cat_cardinalities, num_embedding=num_embedding, **embed_kwargs
    )
    model = _MODEL_REGISTRY[name](embedding=embedding, out_dim=out_dim, **params)
    if not hasattr(model, "output_layer"):
        raise TypeError(f"Model {name!r} must expose an `output_layer` (final nn.Linear)")
    return model


__all__ = [
    "FeatureEmbedding",
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
    "register_model",
    "build_model",
]
