"""Minimal repro hunt: torch.compile(backend="openxla") training inaccuracy.

masaMLP's TPU verification (v5e, torch/torch_xla 2.8) found that the same
training loop reaches rmse ~0.20 in lazy-tensor mode but ~3.18 with the
model wrapped in ``torch.compile(backend="openxla")`` (ft_transformer,
bf16 autocast). This script is masamlp-free so the winning row can be pasted
into a pytorch/xla issue verbatim: it sweeps {architecture} x {amp} x
{execution mode} on one synthetic regression task and flags every
combination where the dynamo backend's final loss diverges from lazy mode.

Run (TPU VM):  python benchmarks/openxla_repro.py [--rows 50000] [--epochs 5]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import Tensor, nn


def make_data(rows: int, n_features: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, n_features)).astype(np.float32)
    w = rng.normal(size=n_features)
    y = (X @ w + 0.5 * X[:, 0] * X[:, 1] + rng.normal(0, 0.1, rows)).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y[:, None])


class MLP(nn.Module):
    def __init__(self, d_in: int, width: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualLN(nn.Module):
    """MLP with LayerNorm residual blocks — between MLP and attention."""

    def __init__(self, d_in: int, width: int = 128, blocks: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.ReLU(),
                          nn.Linear(width, width))
            for _ in range(blocks)
        )
        self.head = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


class TinyAttention(nn.Module):
    """Feature tokens + one MHA block — the ft_transformer-shaped minimum."""

    def __init__(self, d_in: int, d_token: int = 32, n_heads: int = 4) -> None:
        super().__init__()
        self.token_w = nn.Parameter(torch.randn(d_in, d_token) * 0.1)
        self.token_b = nn.Parameter(torch.zeros(d_in, d_token))
        self.attn = nn.MultiheadAttention(d_token, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(nn.Linear(d_token, 2 * d_token), nn.ReLU(),
                                 nn.Linear(2 * d_token, d_token))
        self.head = nn.Linear(d_in * d_token, 1)

    def forward(self, x: Tensor) -> Tensor:
        tokens = x[:, :, None] * self.token_w + self.token_b  # (B, F, d)
        attn, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + attn)
        tokens = tokens + self.ffn(tokens)
        return self.head(tokens.flatten(1))


ARCHS: dict[str, type[nn.Module]] = {
    "mlp": MLP,
    "residual_ln": ResidualLN,
    "attention": TinyAttention,
}


def train_once(arch: str, mode: str, bf16: bool, device, X, Y, args) -> float:
    """One fresh training run; returns final full-data rmse (host float)."""
    import torch_xla

    sync = torch_xla.sync
    torch.manual_seed(0)
    if hasattr(torch_xla, "manual_seed"):
        torch_xla.manual_seed(0)
    model = ARCHS[arch](X.shape[1]).to(device)
    run_model = model
    if mode == "openxla":
        run_model = torch.compile(model, backend="openxla")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    Xd, Yd = X.to(device), Y.to(device)
    n = len(X)
    gen = torch.Generator().manual_seed(0)
    for _ in range(args.epochs):
        for idx_cpu in torch.randperm(n, generator=gen).split(args.batch):
            idx = idx_cpu.to(device)
            xb, yb = Xd[idx], Yd[idx]
            with torch.autocast("xla", dtype=torch.bfloat16, enabled=bf16):
                loss = ((run_model(xb) - yb) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sync()
    with torch.no_grad():
        preds = []
        for start in range(0, n, 8192):
            preds.append(model(Xd[start:start + 8192]).float())
            sync()
        rmse = float(torch.sqrt(((torch.cat(preds) - Yd) ** 2).mean()))
    return rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS),
                        choices=list(ARCHS))
    args = parser.parse_args()

    import torch_xla

    device = torch_xla.device() if hasattr(torch_xla, "device") else None
    if device is None:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    print(f"torch {torch.__version__} torch_xla {torch_xla.__version__} "
          f"device {device}", flush=True)

    X, Y = make_data(args.rows)
    flagged = []
    for arch in args.archs:
        for bf16 in (False, True):
            rmses = {}
            for mode in ("lazy", "openxla"):
                start = time.perf_counter()
                try:
                    rmses[mode] = train_once(arch, mode, bf16, device, X, Y, args)
                except Exception as exc:  # noqa: BLE001 - failures are findings
                    print(f"{arch:12s} bf16={bf16!s:5s} {mode:8s} FAILED: "
                          f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}",
                          flush=True)
                    continue
                print(f"{arch:12s} bf16={bf16!s:5s} {mode:8s} "
                      f"rmse {rmses[mode]:.4f}  ({time.perf_counter() - start:.1f}s)",
                      flush=True)
            if "lazy" in rmses and "openxla" in rmses:
                if rmses["openxla"] > 2.0 * rmses["lazy"] + 0.05:
                    flagged.append((arch, bf16, rmses["lazy"], rmses["openxla"]))

    print("\n== verdict ==", flush=True)
    if not flagged:
        print("no divergence reproduced at this scale/config", flush=True)
    for arch, bf16, lazy_rmse, dynamo_rmse in flagged:
        print(f"DIVERGES: arch={arch} bf16={bf16} lazy rmse {lazy_rmse:.4f} "
              f"vs openxla rmse {dynamo_rmse:.4f}", flush=True)


if __name__ == "__main__":
    main()
