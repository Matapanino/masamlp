import numpy as np
import pytest
import torch

from masamlp.core.device import resolve_amp, resolve_device
from masamlp.regressor import MasaRegressor

_KW = dict(n_epochs=10, random_state=0, model_params={"d": 32, "n_blocks": 1})

cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()


def test_resolve_device_auto_and_validation():
    dev = resolve_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")
    if not cuda_available:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


def test_amp_gating():
    cpu = torch.device("cpu")
    assert resolve_amp("auto", cpu) == (False, None)
    enabled, dtype = resolve_amp(True, cpu)
    assert enabled and dtype == torch.bfloat16
    assert resolve_amp(False, cpu) == (False, None)
    with pytest.raises(ValueError, match="amp"):
        resolve_amp("banana", cpu)


def test_n_threads_runs(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(device="cpu", n_threads=1, **_KW).fit(X, y)
    assert np.isfinite(m.predict(X)).all()


def test_compile_flag_smoke(reg_data):
    X, y, _, _ = reg_data
    m = MasaRegressor(device="cpu", compile=True, n_epochs=3, random_state=0,
                      model_params={"d": 32, "n_blocks": 1})
    m.fit(X, y)
    assert np.isfinite(m.predict(X)).all()


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_cpu_cuda_parity(reg_data):
    X, y, X_test, _ = reg_data
    p_cpu = MasaRegressor(device="cpu", **_KW).fit(X, y).predict(X_test)
    p_gpu = MasaRegressor(device="cuda", amp=False, **_KW).fit(X, y).predict(X_test)
    np.testing.assert_allclose(p_cpu, p_gpu, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not mps_available, reason="MPS not available")
def test_mps_smoke(reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(device="mps", n_epochs=20, random_state=0,
                      model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    baseline = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    assert rmse < baseline
