"""docs/parameters.md must stay in sync with the code: every constructor
parameter of the estimators and the built-in models has to appear backticked
in the reference, so adding a parameter without documenting it fails CI."""

import inspect
from pathlib import Path

from masamlp.classifier import MasaClassifier
from masamlp.models import _EMBEDDING_KEYS, _MODEL_REGISTRY, _NON_PARAM_KEYS
from masamlp.regressor import MasaRegressor

_DOC = (Path(__file__).resolve().parents[1] / "docs" / "parameters.md").read_text(
    encoding="utf-8"
)


def test_builtin_model_params_are_documented():
    missing = []
    for name, builder in sorted(_MODEL_REGISTRY.items()):
        if not builder.__module__.startswith("masamlp."):
            continue  # third-party registrations document themselves
        for param in inspect.signature(builder).parameters:
            if param in _NON_PARAM_KEYS:
                continue
            if f"`{param}`" not in _DOC:
                missing.append(f"{name}.{param}")
    assert not missing, f"model_params missing from docs/parameters.md: {missing}"


def test_estimator_params_are_documented():
    missing = []
    for cls in (MasaRegressor, MasaClassifier):
        for param in inspect.signature(cls.__init__).parameters:
            if param == "self":
                continue
            if f"`{param}`" not in _DOC:
                missing.append(f"{cls.__name__}.{param}")
    assert not missing, f"estimator params missing from docs/parameters.md: {missing}"


def test_shared_embedding_keys_are_documented():
    missing = [key for key in _EMBEDDING_KEYS if f"`{key}`" not in _DOC]
    assert not missing, f"embedding keys missing from docs/parameters.md: {missing}"
