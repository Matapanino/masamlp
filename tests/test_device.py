import numpy as np
import pytest
import torch

from masamlp.core.device import mps_functional, resolve_amp, resolve_device
from masamlp.regressor import MasaRegressor

_KW = dict(n_epochs=10, random_state=0, model_params={"d": 32, "n_blocks": 1})

cuda_available = torch.cuda.is_available()
# Functional probe, not is_available(): virtualized macOS CI runners report
# MPS as available but fail on the first allocation.
mps_available = mps_functional()


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
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(**{**_KW, "device": "cpu", "n_epochs": 30}).fit(X, y)
    p_cpu = m.predict(X_test)
    # Inference parity: the same fitted weights must predict (near-)identically
    # on CUDA. Training trajectories, by contrast, drift apart across devices
    # (float accumulation order compounds over steps — see docs/devices.md),
    # so cross-device *training* is compared on quality, not values.
    m.device = "cuda"
    np.testing.assert_allclose(p_cpu, m.predict(X_test), atol=1e-4, rtol=1e-4)

    m_gpu = MasaRegressor(
        **{**_KW, "device": "cuda", "amp": False, "n_epochs": 30}
    ).fit(X, y)
    baseline = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    rmse_cpu = float(np.sqrt(np.mean((p_cpu - y_test) ** 2)))
    rmse_gpu = float(np.sqrt(np.mean((m_gpu.predict(X_test) - y_test) ** 2)))
    assert rmse_cpu < 0.8 * baseline and rmse_gpu < 0.8 * baseline
    assert abs(rmse_cpu - rmse_gpu) < 0.1 * baseline


@pytest.mark.skipif(not mps_available, reason="MPS not available")
def test_mps_smoke(reg_data):
    X, y, X_test, y_test = reg_data
    m = MasaRegressor(device="mps", n_epochs=20, random_state=0,
                      model_params={"d": 32, "n_blocks": 1}).fit(X, y)
    rmse = float(np.sqrt(np.mean((m.predict(X_test) - y_test) ** 2)))
    baseline = float(np.sqrt(np.mean((y_test - y.mean()) ** 2)))
    assert rmse < baseline
