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


def realmlp_td_params(task: str = "regression", pytabkit_fidelity: bool = False) -> dict[str, Any]:
    """The full RealMLP-TD recipe (Holzmüller et al. 2024, per pytabkit's
    defaults), on top of TD-S: parametric activations (lr x0.1), scheduled
    dropout 0.15 (flat_cos), weight decay 2e-2 (flat_cos, none on biases),
    PBLD numeric embeddings (sigma 0.1, 16 frequencies, 4 dims, lr x0.1),
    hybrid categorical encoding (one-hot up to 9 categories, embeddings of
    size 8 above), lr 0.04 (classification) / 0.2 (regression), and output
    clipping for regression.

    Every value above matches pytabkit's ``DefaultParams.RealMLP_TD_CLASS``
    /``_REG``. Three *structural* choices did not, up to 0.8.0, and are what
    ``pytabkit_fidelity=True`` switches on (0.9.0):

    * ``init_mode="std+he5"`` — pytabkit's data-driven layerwise init
      (``weight_init_mode='std'``, ``bias_init_mode='he+5'``) instead of the
      standalone TD-S reference's ``N(0, 1)`` draw;
    * ``zero_init_output=False`` — TD initializes its output layer like any
      other layer (only TD-**S** zeroes it);
    * ``scale_position="first_layer"`` — TD's ``add_front_scale`` diagonal
      gain sits on the first hidden layer's *input*, i.e. after the PBLD
      embedding, not on the raw numerics.

    It also sets ``onehot_max_categories=8`` (pytabkit's
    ``max_one_hot_cat_size=9`` counts its reserved missing slot) and
    ``drop_last=True`` (its dataloader default). The weight decay is
    unchanged: pytabkit's is decoupled and lr-scaled, which is exactly what
    ``optimizer="adamw"`` already does.

    The default (``pytabkit_fidelity=False``) is unchanged from 0.8.0, so
    existing results reproduce.
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
    if pytabkit_fidelity:
        params["model_params"] = {
            **params["model_params"],
            "init_mode": "std+he5",
            "zero_init_output": False,
            "scale_position": "first_layer",
        }
        params["onehot_max_categories"] = 8
        params["drop_last"] = True
    return params


def realm_td_params(
    task: str = "regression", pytabkit_fidelity: bool = False, k: int = 8
) -> dict[str, Any]:
    """The RealMLP-TD recipe on the **RealM** architecture: the same backbone
    and the same training recipe as :func:`realmlp_td_params`, with the
    single model replaced by a ``k``-member full BatchEnsemble.

    Only the architecture changes — every training knob (coslog4 lr,
    flat_cos weight decay and dropout, PBLD embeddings, label smoothing,
    ``pytabkit_fidelity``) is inherited unchanged, because RealM's members
    train as ordinary predictors on the mean of their per-member losses.

    The three ensembling knobs are spelled out so they are easy to override:

    * ``k`` (default 8) — members. Parameter cost stays near one backbone but
      compute is ~k member-forwards, and one ``(n, k, d)`` activation grows
      linearly in ``k``; ``k=32`` is a scale-up confirmation, not a first try.
    * ``adapter_lr_factor`` (default 1.0) — learning-rate factor for every
      per-member parameter. The TabM-style tuning space is {1, 2, 4}.
    * ``member_init`` (default ``"auto"``) — the first adapter's init;
      ``"auto"`` picks ``"normal"`` here, because the recipe uses PBLD
      numeric embeddings.
    """
    params = realmlp_td_params(task, pytabkit_fidelity=pytabkit_fidelity)
    params["model"] = "realm"
    params["model_params"] = {
        **params["model_params"],
        "k": k,
        "adapter_lr_factor": 1.0,
        "member_init": "auto",
    }
    return params
