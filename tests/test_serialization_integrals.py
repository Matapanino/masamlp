"""No-fit coverage of integral metadata and existing custom-object handling."""

import json

import numpy as np
import pytest
import torch

from masamlp.core.serialization import _json_default, _json_safe_params
from masamlp.models import build_model
from masamlp.regressor import MasaRegressor


class _NoFitPreprocessor:
    """Only the static serialization protocol; never estimates data statistics."""

    cat_cardinalities_ = []

    def transform_width(self):
        return 2, 0

    def get_state(self):
        return {"fixture": "no_fit"}, {"constant": np.array([1.0])}


@pytest.mark.parametrize("scalar_type", [np.int32, np.int64, np.uint64])
def test_no_fit_numpy_epoch_counts_survive_directory_roundtrip(tmp_path, monkeypatch, scalar_type):
    horizon, stop = scalar_type(6), scalar_type(3)
    params = {"hidden_sizes": [np.int64(4)], "activation": "silu"}
    est = MasaRegressor(model="realmlp", model_params=params, n_epochs=horizon, stop_epoch=stop)
    assert est.n_epochs is horizon and est.stop_epoch is stop  # sklearn constructor contract
    # Supply a constructor-created model plus static metadata. No estimator,
    # scaler, model or data-driven initialization is fitted in this test.
    est.preprocessor_ = _NoFitPreprocessor()
    est.model_ = build_model("realmlp", params, 2, [], 1)
    est.models_ = [est.model_]
    est.resolved_model_params_ = params
    est.n_features_in_, est.feature_names_in_ = 2, np.array(["a", "b"])
    est.out_dim_, est.transform_name_ = 1, "identity"
    est.best_iteration_, est.best_score_, est.evals_result_ = np.int64(2), None, {}

    def from_state(meta, arrays):
        assert meta == {"fixture": "no_fit"}
        np.testing.assert_array_equal(arrays["constant"], [1.0])
        return _NoFitPreprocessor()

    monkeypatch.setattr("masamlp.core.serialization.TabularPreprocessor.from_state", from_state)
    destination = tmp_path / "model"
    est.save_model(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["dropped_params"] == []
    assert manifest["params"]["n_epochs"] == 6
    assert manifest["params"]["stop_epoch"] == 3
    assert manifest["fitted"]["resolved_model_params"]["hidden_sizes"] == [4]
    restored = MasaRegressor.load_model(destination)
    assert restored.n_epochs == 6 and restored.stop_epoch == 3
    assert restored.best_iteration_ == 2
    assert restored.model_params == params
    for key, value in est.model_.state_dict().items():
        assert torch.equal(value, restored.model_.state_dict()[key])


def test_no_fit_nested_integrals_are_exact_and_custom_objects_still_drop():
    custom = object()
    values = {
        "nested": {"counts": [np.int16(3), (np.uint64(2**64 - 1),)]},
        "custom": custom, "nested_custom": {"custom": custom},
        "array": np.array([1, 2]),
    }
    safe, dropped = _json_safe_params(values)
    assert dropped == ["custom", "nested_custom", "array"]
    encoded = json.loads(json.dumps(safe, default=_json_default))
    assert encoded == {"nested": {"counts": [3, [2**64 - 1]]}}
    assert values["nested"]["counts"][0].dtype == np.dtype("int16")
