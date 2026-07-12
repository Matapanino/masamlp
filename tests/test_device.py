import importlib.util

import numpy as np
import pytest
import torch

from masamlp.core.device import (
    mps_functional,
    resolve_amp,
    resolve_device,
    resolve_predict_amp,
)
from masamlp.regressor import MasaRegressor

_KW = dict(n_epochs=10, random_state=0, model_params={"d": 32, "n_blocks": 1})

cuda_available = torch.cuda.is_available()
xla_available = importlib.util.find_spec("torch_xla") is not None
# Functional probe, not is_available(): virtualized macOS CI runners report
# MPS as available but fail on the first allocation.
mps_available = mps_functional()


def test_resolve_device_auto_and_validation():
    dev = resolve_device("auto")
    assert dev.type in ("cuda", "mps", "cpu")
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("gpu")
    if not xla_available:
        # "xla"/"tpu" are known vocabulary; without torch_xla they fail with
        # an install hint rather than an unknown-device error.
        with pytest.raises(RuntimeError, match="torch_xla"):
            resolve_device("xla")
        with pytest.raises(RuntimeError, match="torch_xla"):
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


def test_amp_auto_policies_on_xla_device_type():
    # Pure-policy checks on a torch.device("xla") handle; no torch_xla needed.
    xla = torch.device("xla")

    class DictPolicy:
        amp_auto = {"cuda": False}  # the retrieval models' policy

    class HardOff:
        amp_auto = False

    assert resolve_amp("auto", xla, DictPolicy()) == (True, torch.bfloat16)
    assert resolve_amp("auto", xla, HardOff()) == (False, None)
    assert resolve_amp("auto", xla) == (True, torch.bfloat16)
    assert resolve_amp(True, xla) == (True, torch.bfloat16)
    if cuda_available:
        cuda = torch.device("cuda")
        assert resolve_amp("auto", cuda, DictPolicy()) == (False, None)


def test_amp_auto_respects_model_policy(monkeypatch):
    import masamlp.core.device as device_mod

    class _NoAmp(torch.nn.Module):
        amp_auto = False

    class _Bf16Only(torch.nn.Module):
        amp_auto = "bf16"

    # The model gate fires before any CUDA API call, so a cuda device object
    # is safe to pass on CUDA-less machines.
    assert resolve_amp("auto", torch.device("cuda"), _NoAmp()) == (False, None)
    # Explicit amp=True overrides the model's auto policy.
    enabled, dtype = resolve_amp(True, torch.device("cpu"), _NoAmp())
    assert enabled and dtype == torch.bfloat16
    # Models without the attribute keep the plain auto behavior.
    assert resolve_amp("auto", torch.device("cpu"), torch.nn.Linear(2, 2)) == (False, None)
    # amp_auto="bf16": AMP under auto only when the device dtype is bf16.
    monkeypatch.setattr(device_mod, "_cuda_amp_dtype", lambda device: torch.float16)
    assert resolve_amp("auto", torch.device("cuda"), _Bf16Only()) == (False, None)
    monkeypatch.setattr(device_mod, "_cuda_amp_dtype", lambda device: torch.bfloat16)
    assert resolve_amp("auto", torch.device("cuda"), _Bf16Only()) == (True, torch.bfloat16)


def test_resolve_predict_amp():
    cpu = torch.device("cpu")
    xla = torch.device("xla")  # pure-policy check; no torch_xla needed
    assert resolve_predict_amp(False, cpu) is None
    assert resolve_predict_amp("off", xla) is None
    assert resolve_predict_amp(True, xla) is torch.bfloat16
    assert resolve_predict_amp("on", cpu) is torch.bfloat16
    with pytest.warns(UserWarning, match="amp_predict"):
        assert resolve_predict_amp(True, torch.device("mps")) is None
    with pytest.raises(ValueError, match="amp_predict"):
        resolve_predict_amp("banana", cpu)


def test_amp_predict_cpu_end_to_end(reg_data):
    # bf16 prediction on CPU: same fitted model, opt-in cast, close outputs.
    X, y, X_test, _ = reg_data
    m = MasaRegressor(device="cpu", **_KW).fit(X, y)
    p32 = m.predict(X_test)
    m.set_params(amp_predict=True)
    p16 = m.predict(X_test)
    assert np.all(np.isfinite(p16))
    np.testing.assert_allclose(p16, p32, atol=0.05, rtol=0.05)


def test_amp_auto_model_flags():
    from masamlp.models import FTTransformer, ModernNCA, TabR

    # KI-010 is CUDA-scoped since 0.4.0: bf16 measured exact-and-faster on TPU.
    assert TabR.amp_auto == {"cuda": False}
    assert ModernNCA.amp_auto == {"cuda": False}
    assert FTTransformer.amp_auto == "bf16"  # fp16 slower + less accurate on T4


def test_module_device_falls_back_to_buffers():
    from masamlp.core.device import module_device

    assert module_device(torch.nn.Linear(2, 2)) == torch.device("cpu")

    class _BufferOnly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("stats", torch.zeros(3))

    assert module_device(_BufferOnly()) == torch.device("cpu")
    assert module_device(torch.nn.Module()) == torch.device("cpu")


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
