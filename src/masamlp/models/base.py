"""Feature embedding shared by every model.

Numeric features pass through optionally as periodic or PLR-family
embeddings (Gorishniy et al. 2022, "On Embeddings for Numerical Features in
Tabular Deep Learning"; PBLD variant per Holzmüller et al. 2024 / pytabkit);
categorical features get per-column ``nn.Embedding`` with row 0 reserved for
unknown/missing. An optional RealMLP-style learnable per-feature scale is
applied to numeric inputs first. The output is a flat ``(n, d_out)`` tensor
the model trunks consume.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from masamlp.models.layers import ScalingLayer

#: num_embedding option -> (activation, cos_bias, densenet, lite)
_PLR_VARIANTS = {
    "pl": ("linear", False, False, False),
    "plr": ("relu", False, False, False),
    "plr-lite": ("relu", False, False, True),
    "pbld": ("linear", True, True, False),
}


def _auto_emb_dim(cardinality: int) -> int:
    return int(min(32, max(2, round(1.6 * cardinality**0.56))))


class PeriodicEmbedding(nn.Module):
    """Per-feature ``[cos(2*pi*c*x), sin(2*pi*c*x)]`` with learnable
    frequencies ``c ~ N(0, sigma^2)``; output ``(n, F, 2*n_frequencies)``."""

    def __init__(self, n_features: int, n_frequencies: int = 16, sigma: float = 0.1) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)

    def forward(self, x: Tensor) -> Tensor:
        angles = 2.0 * math.pi * x.unsqueeze(-1) * self.frequencies
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)


class PLREmbedding(nn.Module):
    """The PL / PLR / PBLD family: periodic features -> per-feature linear
    (-> activation), flattened to ``(n, F * d_embedding)``.

    - ``cos_bias=False``: first layer is ``[cos(2*pi*c*x), sin(2*pi*c*x)]``.
    - ``cos_bias=True`` (PBLD): ``cos(2*pi*c*x + b)`` with ``b ~ U[-pi, pi]``.
    - ``densenet=True`` (PBLD): the raw feature value is concatenated, taking
      one of the ``d_embedding`` output slots.
    - ``lite=True`` (rtdl's PLR "lite"): the linear layer is shared across
      features instead of per-feature (ModernNCA's default choice).
    - ``flatten=False`` keeps per-feature tokens ``(n, F, d_embedding)`` for
      token-based models (FT-Transformer).
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 16,
        n_frequencies: int = 16,
        sigma: float = 0.1,
        activation: str = "relu",
        cos_bias: bool = False,
        densenet: bool = False,
        lite: bool = False,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        if activation not in ("relu", "linear"):
            raise ValueError(f"Unknown PLR activation {activation!r}")
        if densenet and d_embedding < 2:
            raise ValueError("densenet needs d_embedding >= 2")
        self.activation = activation
        self.cos_bias = cos_bias
        self.densenet = densenet
        self.lite = lite
        self.flatten = flatten
        self.frequencies = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        # Smaller values than U[0, 2pi] behave better under weight decay.
        self.cos_bias_param = (
            nn.Parameter(math.pi * (2.0 * torch.rand(n_features, n_frequencies) - 1.0))
            if cos_bias
            else None
        )
        d_first = n_frequencies if cos_bias else 2 * n_frequencies
        d_linear = d_embedding - 1 if densenet else d_embedding
        bound = 1.0 / math.sqrt(d_first)
        w_shape = (d_first, d_linear) if lite else (n_features, d_first, d_linear)
        b_shape = (d_linear,) if lite else (n_features, d_linear)
        self.weight = nn.Parameter((2.0 * torch.rand(*w_shape) - 1.0) * bound)
        self.bias = nn.Parameter((2.0 * torch.rand(*b_shape) - 1.0) * bound)

    def forward(self, x: Tensor) -> Tensor:
        angles = 2.0 * math.pi * x.unsqueeze(-1) * self.frequencies
        if self.cos_bias_param is not None:
            periodic = torch.cos(angles + self.cos_bias_param)
        else:
            periodic = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if self.lite:
            out = periodic @ self.weight + self.bias
        else:
            out = torch.einsum("bfk,fkd->bfd", periodic, self.weight) + self.bias
        if self.activation == "relu":
            out = torch.relu(out)
        if self.densenet:
            out = torch.cat([out, x.unsqueeze(-1)], dim=-1)
        return out.flatten(1) if self.flatten else out


class FeatureEmbedding(nn.Module):
    """Numeric (raw / periodic / pl / plr / pbld) + categorical embeddings,
    concatenated. ``num_scaling=True`` prepends a learnable per-feature scale
    on the numeric inputs (RealMLP).

    ``cat_cardinalities`` already include the reserved unknown/missing slot
    (index 0), matching ``TabularPreprocessor.cat_cardinalities_``.
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        num_embedding: str | None = None,
        d_num_embedding: int = 16,
        n_frequencies: int = 16,
        sigma: float = 0.1,
        cat_emb_dim: int | None = None,
        num_scaling: bool = False,
    ) -> None:
        super().__init__()
        if n_num == 0 and not cat_cardinalities:
            raise ValueError("model needs at least one numeric or categorical feature")
        self.n_num = n_num
        self.scaling = ScalingLayer(n_num) if num_scaling and n_num > 0 else None
        self.num_embedding: nn.Module | None = None
        d_num = n_num
        if n_num > 0 and num_embedding is not None:
            if num_embedding == "periodic":
                self.num_embedding = PeriodicEmbedding(n_num, n_frequencies, sigma)
                d_num = n_num * 2 * n_frequencies
            elif num_embedding in _PLR_VARIANTS:
                act, cos_bias, densenet, lite = _PLR_VARIANTS[num_embedding]
                self.num_embedding = PLREmbedding(
                    n_num,
                    d_num_embedding,
                    n_frequencies,
                    sigma,
                    activation=act,
                    cos_bias=cos_bias,
                    densenet=densenet,
                    lite=lite,
                )
                d_num = n_num * d_num_embedding
            else:
                raise ValueError(
                    f"Unknown num_embedding {num_embedding!r}. Expected one of "
                    f"{('plr', 'plr-lite', 'pl', 'pbld', 'periodic')} or None"
                )
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(card, cat_emb_dim if cat_emb_dim is not None else _auto_emb_dim(card))
                for card in cat_cardinalities
            ]
        )
        self.d_out = d_num + sum(emb.embedding_dim for emb in self.cat_embeddings)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> Tensor:
        parts: list[Tensor] = []
        if self.n_num > 0:
            if self.scaling is not None:
                x_num = self.scaling(x_num)
            if self.num_embedding is not None:
                num = self.num_embedding(x_num)
                parts.append(num.flatten(1) if num.ndim == 3 else num)
            else:
                parts.append(x_num)
        for j, emb in enumerate(self.cat_embeddings):
            parts.append(emb(x_cat[:, j]))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)


class LinearTokens(nn.Module):
    """Per-feature linear tokens ``x_i * w_i + b_i`` (rtdl's
    LinearEmbeddings), output ``(n, F, d_token)``."""

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        bound = 1.0 / math.sqrt(d_token)
        self.weight = nn.Parameter((2.0 * torch.rand(n_features, d_token) - 1.0) * bound)
        self.bias = nn.Parameter((2.0 * torch.rand(n_features, d_token) - 1.0) * bound)

    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class TokenEmbedding(nn.Module):
    """Per-feature tokens for attention-based models.

    Categorical features always become ``d_token``-dim embedding tokens
    (index 0 reserved for unknown/missing). Numeric features either become
    tokens too (``tokenize_numeric=True``, FT-Transformer style — linear
    tokens by default, or a PLR-family embedding with
    ``d_embedding=d_token``) or stay a flat vector (TabTransformer style,
    optionally passed through a flat numeric embedding).

    ``forward`` returns ``(tokens, num_flat)`` with shapes
    ``(n, n_tokens, d_token)`` and ``(n, d_num_flat)``.
    """

    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        d_token: int,
        num_embedding: str | None = None,
        d_num_embedding: int = 16,
        n_frequencies: int = 16,
        sigma: float = 0.1,
        cat_emb_dim: int | None = None,  # ignored: tokens are d_token wide
        num_scaling: bool = False,
        tokenize_numeric: bool = True,
    ) -> None:
        super().__init__()
        if n_num == 0 and not cat_cardinalities:
            raise ValueError("model needs at least one numeric or categorical feature")
        self.n_num = n_num
        self.tokenize_numeric = tokenize_numeric
        self.scaling = ScalingLayer(n_num) if num_scaling and n_num > 0 else None
        self.num_tokens: nn.Module | None = None
        self.num_flat_embedding: FeatureEmbedding | None = None
        self.d_num_flat = 0
        if n_num > 0:
            if tokenize_numeric:
                if num_embedding is None:
                    self.num_tokens = LinearTokens(n_num, d_token)
                elif num_embedding in _PLR_VARIANTS:
                    act, cos_bias, densenet, lite = _PLR_VARIANTS[num_embedding]
                    self.num_tokens = PLREmbedding(
                        n_num,
                        d_embedding=d_token,
                        n_frequencies=n_frequencies,
                        sigma=sigma,
                        activation=act,
                        cos_bias=cos_bias,
                        densenet=densenet,
                        lite=lite,
                        flatten=False,
                    )
                else:
                    raise ValueError(
                        f"num_embedding {num_embedding!r} is not supported for "
                        "token-based models; use 'plr', 'plr-lite', 'pl', 'pbld', or None"
                    )
            else:
                # Flat numeric path (TabTransformer): reuse the flat embedding
                # without its categorical part or double scaling.
                self.num_flat_embedding = FeatureEmbedding(
                    n_num,
                    [],
                    num_embedding=num_embedding,
                    d_num_embedding=d_num_embedding,
                    n_frequencies=n_frequencies,
                    sigma=sigma,
                )
                self.d_num_flat = self.num_flat_embedding.d_out
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )
        bound = 1.0 / math.sqrt(d_token)
        for emb in self.cat_embeddings:
            nn.init.uniform_(emb.weight, -bound, bound)
        self.d_token = d_token
        self.n_tokens = (n_num if tokenize_numeric else 0) + len(cat_cardinalities)

    def forward(self, x_num: Tensor, x_cat: Tensor) -> tuple[Tensor, Tensor]:
        tokens: list[Tensor] = []
        num_flat = x_num.new_zeros(x_num.shape[0], 0)
        if self.n_num > 0:
            if self.scaling is not None:
                x_num = self.scaling(x_num)
            if self.num_tokens is not None:
                tokens.append(self.num_tokens(x_num))
            else:
                num_flat = self.num_flat_embedding(x_num, x_num.new_zeros(
                    x_num.shape[0], 0, dtype=torch.int64))
        if self.cat_embeddings:
            tokens.append(
                torch.stack([emb(x_cat[:, j]) for j, emb in enumerate(self.cat_embeddings)], dim=1)
            )
        if tokens:
            out = tokens[0] if len(tokens) == 1 else torch.cat(tokens, dim=1)
        else:
            out = x_num.new_zeros(x_num.shape[0], 0, self.d_token)
        return out, num_flat
