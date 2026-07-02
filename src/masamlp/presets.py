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
