"""Directory-format save/load.

Layout: ``manifest.json`` (params + fitted metadata), ``preprocessor.json`` /
``preprocessor.npz`` (scaling and category state), and ``model_state.pt`` (a
plain tensor state_dict, loaded with ``weights_only=True`` — no pickle
execution on load). Custom objective/metric objects are intentionally not
serialized: prediction only needs the stored output transform; refitting a
loaded estimator requires re-setting them.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from masamlp.data.preprocessing import TabularPreprocessor
from masamlp.models import build_model

_MANIFEST = "manifest.json"
_PRE_JSON = "preprocessor.json"
_PRE_NPZ = "preprocessor.npz"
_STATE = "model_state.pt"


def _json_safe_params(params: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    safe: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in params.items():
        try:
            json.dumps(value)
        except TypeError:
            dropped.append(key)
        else:
            safe[key] = value
    return safe, dropped


def save_model_dir(est: Any, path: str) -> None:
    from masamlp import __version__

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)

    params, dropped = _json_safe_params(est.get_params())
    if "model_params" in dropped:
        raise ValueError(
            "model_params must be JSON-serializable (plain scalars) to save the model"
        )
    if dropped:
        warnings.warn(
            f"Parameters {dropped} are custom objects and were not serialized; "
            "the loaded model can predict, but refitting requires re-setting them",
            stacklevel=2,
        )

    x_num_width, _ = est.preprocessor_.transform_width()
    fitted: dict[str, Any] = {
        "n_features_in": est.n_features_in_,
        "feature_names_in": [str(n) for n in est.feature_names_in_],
        "out_dim": est.out_dim_,
        "transform_name": est.transform_name_,
        "n_num": x_num_width,
        "cat_cardinalities": est.preprocessor_.cat_cardinalities_,
        "resolved_model_params": est.resolved_model_params_,
        "best_iteration": est.best_iteration_,
        "best_score": est.best_score_,
        "evals_result": est.evals_result_,
    }
    if hasattr(est, "classes_"):
        fitted["classes"] = np.asarray(est.classes_).tolist()
    if getattr(est, "target_mean_", None) is not None:
        fitted["target_mean"] = est.target_mean_.tolist()
        fitted["target_std"] = est.target_std_.tolist()
    if getattr(est, "target_min_", None) is not None:
        fitted["target_min"] = est.target_min_.tolist()
        fitted["target_max"] = est.target_max_.tolist()
    if getattr(est.model_, "has_candidates", False):
        model = est.model_
        fitted["candidate_shapes"] = {
            "x_num": list(model.cand_x_num.shape),
            "x_cat": list(model.cand_x_cat.shape),
            "y": list(model.cand_y.shape),
            "y_dtype": str(model.cand_y.dtype).removeprefix("torch."),
        }
    members = getattr(est, "models_", None) or [est.model_]
    fitted["n_members"] = len(members)

    manifest = {
        "library": "masamlp",
        "version": __version__,
        "estimator": type(est).__name__,
        "params": params,
        "dropped_params": dropped,
        "fitted": fitted,
    }
    (out / _MANIFEST).write_text(json.dumps(manifest, indent=2))

    pre_meta, pre_arrays = est.preprocessor_.get_state()
    (out / _PRE_JSON).write_text(json.dumps(pre_meta, indent=2))
    np.savez(out / _PRE_NPZ, **pre_arrays)

    # Normalize to CPU tensors: XLA tensors must not be pickled raw, and
    # device-tagged archives (cuda:1, ...) are a portability hazard anyway.
    def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in module.state_dict().items()}

    if len(members) == 1:
        torch.save(_cpu_state(members[0]), out / _STATE)
    else:
        torch.save({"members": [_cpu_state(m) for m in members]}, out / _STATE)


def load_model_dir(path: str, cls: type) -> Any:
    src = Path(path)
    manifest = json.loads((src / _MANIFEST).read_text())
    if manifest.get("library") != "masamlp":
        raise ValueError(f"{path} is not a masamlp model directory")
    if manifest["estimator"] != cls.__name__:
        raise ValueError(
            f"Model was saved as {manifest['estimator']}, not {cls.__name__}; "
            f"load it with {manifest['estimator']}.load_model"
        )

    est = cls(**manifest["params"])
    fitted = manifest["fitted"]

    pre_meta = json.loads((src / _PRE_JSON).read_text())
    with np.load(src / _PRE_NPZ) as npz:
        pre_arrays = {k: npz[k] for k in npz.files}
    est.preprocessor_ = TabularPreprocessor.from_state(pre_meta, pre_arrays)

    resolved_params = fitted.get("resolved_model_params", est.model_params)
    est.resolved_model_params_ = resolved_params
    state = torch.load(src / _STATE, weights_only=True, map_location="cpu")
    states = state["members"] if fitted.get("n_members", 1) > 1 else [state]
    est.models_ = []
    for member_state in states:
        model = build_model(
            est.model,
            resolved_params,
            n_num=fitted["n_num"],
            cat_cardinalities=list(fitted["cat_cardinalities"]),
            out_dim=fitted["out_dim"],
            num_embedding=est.num_embedding,
        )
        if "candidate_shapes" in fitted:
            # Register placeholder buffers so load_state_dict can fill in the
            # retrieval corpus saved with the model.
            shapes = fitted["candidate_shapes"]
            model.set_candidates(
                torch.zeros(shapes["x_num"], dtype=torch.float32),
                torch.zeros(shapes["x_cat"], dtype=torch.int64),
                torch.zeros(shapes["y"], dtype=getattr(torch, shapes["y_dtype"])),
            )
        model.load_state_dict(member_state)
        model.eval()
        est.models_.append(model)
    est.model_ = est.models_[0]

    est.objective_ = None
    est.transform_name_ = fitted["transform_name"]
    est.out_dim_ = fitted["out_dim"]
    est.n_features_in_ = fitted["n_features_in"]
    est.feature_names_in_ = np.asarray(fitted["feature_names_in"], dtype=object)
    est.best_iteration_ = fitted["best_iteration"]
    est.best_score_ = fitted["best_score"]
    est.evals_result_ = fitted["evals_result"]
    if "classes" in fitted:
        est.classes_ = np.asarray(fitted["classes"])
    if hasattr(est, "target_standardize"):
        est.target_mean_ = est.target_std_ = None
        est.target_min_ = est.target_max_ = None
        if "target_mean" in fitted:
            est.target_mean_ = np.asarray(fitted["target_mean"], dtype=np.float64)
            est.target_std_ = np.asarray(fitted["target_std"], dtype=np.float64)
        if "target_min" in fitted:
            est.target_min_ = np.asarray(fitted["target_min"], dtype=np.float64)
            est.target_max_ = np.asarray(fitted["target_max"], dtype=np.float64)
    return est
