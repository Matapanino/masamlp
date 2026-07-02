"""Estimator-parameter presets for published training recipes.

Presets return plain kwargs dicts so they stay transparent and overridable:

    from masamlp import MasaClassifier
    from masamlp.presets import realmlp_params

    clf = MasaClassifier(**{**realmlp_params("classification"), "n_epochs": 128})
"""

from __future__ import annotations

from typing import Any


def realmlp_params(task: str = "regression") -> dict[str, Any]:
    """The RealMLP-TD-S training recipe (Holzmüller et al. 2024).

    Robust-scale-smooth-clip preprocessing over one-hot categories, the
    coslog4 per-step schedule, Adam with ``betas=(0.9, 0.95)``, base lr 0.04
    (classification, with label smoothing 0.1) / 0.07 (regression), batch
    size 256 for 256 epochs. Architecture defaults (scaling layer, SELU/Mish,
    zero-init output) come with ``model="realmlp"``. Add
    ``model_params={"num_embedding": "pbld"}`` — via ``num_embedding="pbld"``
    on the estimator — to move toward RealMLP-TD.
    """
    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")
    classification = task == "classification"
    params: dict[str, Any] = {
        "model": "realmlp",
        "numeric_scaler": "rssc",
        "cat_encoding": "onehot",
        "n_epochs": 256,
        "batch_size": 256,
        "optimizer": "adam",
        "optimizer_betas": (0.9, 0.95),
        "lr_scheduler": "coslog4",
        "learning_rate": 0.04 if classification else 0.07,
    }
    if classification:
        params["label_smoothing"] = 0.1
    return params


def realmlp_td_params(task: str = "regression") -> dict[str, Any]:
    """The full RealMLP-TD recipe (Holzmüller et al. 2024, per pytabkit's
    defaults), on top of TD-S: parametric activations (lr x0.1), scheduled
    dropout 0.15 (flat_cos), weight decay 2e-2 (flat_cos, none on biases),
    PBLD numeric embeddings (sigma 0.1, 16 frequencies, 4 dims, lr x0.1),
    hybrid categorical encoding (one-hot up to 9 categories, embeddings of
    size 8 above), lr 0.04 (classification) / 0.2 (regression), and output
    clipping for regression.

    Known deviations from pytabkit: decoupled AdamW-style weight decay
    (pytabkit couples it into Adam), and the reference N(0,1) init instead
    of pytabkit's data-driven ``he+5``/``std`` init modes.
    """
    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")
    classification = task == "classification"
    params: dict[str, Any] = {
        "model": "realmlp",
        "model_params": {
            "num_scaling": True,
            "activation": "selu" if classification else "mish",
            "use_parametric_act": True,
            "act_lr_factor": 0.1,
            "dropout": 0.15,
            "dropout_schedule": "flat_cos",
            "plr_lr_factor": 0.1,
            "d_num_embedding": 4,
            "n_frequencies": 16,
            "sigma": 0.1,
            "cat_emb_dim": 8,
        },
        "num_embedding": "pbld",
        "numeric_scaler": "rssc",
        "cat_encoding": "hybrid",
        "n_epochs": 256,
        "batch_size": 256,
        "optimizer": "adamw",
        "optimizer_betas": (0.9, 0.95),
        "lr_scheduler": "coslog4",
        "learning_rate": 0.04 if classification else 0.2,
        "weight_decay": 2e-2,
        "weight_decay_schedule": "flat_cos",
    }
    if classification:
        params["label_smoothing"] = 0.1
    else:
        params["clip_predictions"] = True
    return params
