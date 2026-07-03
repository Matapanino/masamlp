"""Device resolution and per-device feature gating (AMP, torch.compile)."""

from __future__ import annotations

import functools
import warnings

import torch

_KNOWN = ("auto", "cpu", "cuda", "mps")


@functools.lru_cache(maxsize=1)
def mps_functional() -> bool:
    """True when MPS is available *and* actually works. Virtualized macOS
    hosts (e.g. GitHub Actions runners) report MPS as available but fail on
    the first allocation, so probe with one."""
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.zeros(1, device="mps")
        return True
    except RuntimeError:
        return False


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve ``"auto"`` to cuda > mps > cpu; validate explicit choices."""
    if isinstance(device, torch.device):
        return device
    if device not in _KNOWN and not device.startswith("cuda:"):
        raise ValueError(f"Unknown device {device!r}. Expected one of {_KNOWN} or 'cuda:N'")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_functional():
            return torch.device("mps")
        return torch.device("cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if device == "mps" and not mps_functional():
        raise RuntimeError("device='mps' requested but MPS is not available or not functional")
    return torch.device(device)


def resolve_device_plan(
    device: str | torch.device, n_members: int
) -> list[torch.device] | None:
    """Per-member device assignment for ensemble-member sharding, or ``None``
    for the single-device path.

    Shards only when there are multiple members, CUDA reports multiple
    devices, and the request is ``"auto"`` or the index-less ``"cuda"`` — an
    explicit ``"cuda:0"``/``"cpu"``/``"mps"``/``torch.device`` opts out.
    Member ``i`` trains on ``cuda:(i % n_gpus)``.
    """
    if n_members <= 1 or not isinstance(device, str) or device not in ("auto", "cuda"):
        return None
    if not torch.cuda.is_available():
        return None
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return None
    return [torch.device("cuda", i % n_gpus) for i in range(n_members)]


def resolve_amp(
    amp: str | bool, device: torch.device, model: torch.nn.Module | None = None
) -> tuple[bool, torch.dtype | None]:
    """Return (enabled, autocast dtype) for mixed-precision training.

    ``"auto"`` enables bf16 on CUDA (fp16 on GPUs without bf16) and disables
    AMP on CPU/MPS, where it rarely pays off for tabular-sized models. Models
    may opt out of the auto policy with a class attribute ``amp_auto = False``
    (retrieval models: KI-010 — autocast around cdist/topk is slower and
    fp16 distances lose accuracy); an explicit ``amp=True`` still forces it.
    """
    if amp is False or amp == "off":
        return False, None
    if amp == "auto":
        if model is not None and not getattr(model, "amp_auto", True):
            return False, None
        if device.type != "cuda":
            return False, None
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return True, dtype
    if amp is True or amp == "on":
        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return True, dtype
        if device.type == "cpu":
            return True, torch.bfloat16
        warnings.warn("AMP is not supported on MPS; training in float32", stacklevel=2)
        return False, None
    raise ValueError(f"Unknown amp setting {amp!r}. Expected 'auto', True/'on', or False/'off'")


def maybe_compile(model: torch.nn.Module, enable: bool, device: torch.device) -> torch.nn.Module:
    """Apply ``torch.compile`` when requested, falling back with a warning."""
    if not enable:
        return model
    if device.type == "mps":
        warnings.warn("torch.compile is disabled on MPS; running eager", stacklevel=2)
        return model
    try:
        return torch.compile(model)
    except Exception as exc:  # pragma: no cover - depends on toolchain
        warnings.warn(f"torch.compile failed ({exc!r}); running eager", stacklevel=2)
        return model


def set_threads(n_threads: int | None) -> None:
    if n_threads is not None:
        torch.set_num_threads(int(n_threads))
