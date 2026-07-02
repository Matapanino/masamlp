"""Device resolution and per-device feature gating (AMP, torch.compile)."""

from __future__ import annotations

import warnings

import torch

_KNOWN = ("auto", "cpu", "cuda", "mps")


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve ``"auto"`` to cuda > mps > cpu; validate explicit choices."""
    if isinstance(device, torch.device):
        return device
    if device not in _KNOWN and not device.startswith("cuda:"):
        raise ValueError(f"Unknown device {device!r}. Expected one of {_KNOWN} or 'cuda:N'")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device='mps' requested but MPS is not available")
    return torch.device(device)


def resolve_amp(amp: str | bool, device: torch.device) -> tuple[bool, torch.dtype | None]:
    """Return (enabled, autocast dtype) for mixed-precision training.

    ``"auto"`` enables bf16 on CUDA (fp16 on GPUs without bf16) and disables
    AMP on CPU/MPS, where it rarely pays off for tabular-sized models.
    """
    if amp is False or amp == "off":
        return False, None
    if amp == "auto":
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
