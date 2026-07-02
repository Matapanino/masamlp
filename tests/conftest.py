"""Shared fixtures: small seeded synthetic datasets and tiny model configs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Small architectures keep the suite fast; every model is exercised with them.
TINY_PARAMS = {
    "resnet": {"d": 32, "n_blocks": 1},
    "danet": {"n_layers": 2, "base_outdim": 16, "k": 2, "virtual_batch_size": 64},
    "lnn": {"d_hidden": 16, "n_steps": 2, "d_backbone": 32},
    "realmlp": {"hidden_sizes": [32, 32]},
    "tabr": {"d_main": 16, "context_size": 8},
    "ft_transformer": {"n_blocks": 1, "d_block": 64, "attention_dropout": 0.1, "ffn_dropout": 0.0},
    "tab_transformer": {"n_layers": 2, "d_token": 16},
    "modernnca": {"dim": 32, "d_block": 64},
}

# Attention models tokenize features; "periodic" has no fixed token width.
TOKEN_MODELS = ("ft_transformer", "tab_transformer")

# Estimator-level settings some models need to learn quickly in tests
# (RealMLP's NTP layers are built for its high-lr recipe).
TRAIN_KWARGS = {
    "realmlp": {
        "learning_rate": 0.05,
        "optimizer": "adam",
        "optimizer_betas": (0.9, 0.95),
        "lr_scheduler": "coslog4",
    },
    # On numeric-only data TabTransformer's head width scales with the tiny
    # raw input; PLR numeric embeddings (the intended extension) fix that.
    "tab_transformer": {"learning_rate": 3e-3, "num_embedding": "plr"},
}

ALL_MODELS = sorted(TINY_PARAMS)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def reg_data(rng):
    X = rng.normal(size=(400, 6))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + rng.normal(0, 0.1, 400)
    return X[:300], y[:300], X[300:], y[300:]


@pytest.fixture
def clf_data(rng):
    X = rng.normal(size=(400, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:300], y[:300], X[300:], y[300:]


@pytest.fixture
def multiclass_data(rng):
    X = rng.normal(size=(450, 6))
    y = np.digitize(X[:, 0] + 0.5 * X[:, 1], [-0.5, 0.5])
    return X[:350], y[:350], X[350:], y[350:]


@pytest.fixture
def mixed_df(rng):
    n = 300
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    y = np.where(df["cat"] == "a", 2.0, -1.0) + 0.5 * df["num1"].to_numpy()
    return df, y + rng.normal(0, 0.1, n)
