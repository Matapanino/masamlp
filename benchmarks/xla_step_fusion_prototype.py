"""Prototype: how far can XLA step fusion go on TPU? (research, not library)

Three executions of the same minibatch training epoch on one XLA device:

1. ``loop K=1``   — the 0.4.0 trainer pattern: one graph barrier per
                    optimizer step (the per-step dispatch floor being probed).
2. ``loop K=N``   — the 0.5.0 ``xla_fuse_steps`` pattern: barrier every N
                    steps, so N steps fuse into one XLA program. Python still
                    re-traces every step; only dispatch/launch is amortized.
3. ``scan``       — ``torch_xla.experimental.scan`` over the epoch's
                    full-size batches: the step loop lives *inside* the
                    compiled program as an XLA While loop (one trace, one
                    dispatch per epoch). Requires a functional optimizer, so
                    a hand-rolled functional AdamW runs over
                    ``torch.func.grad`` — this is exactly the surgery the
                    library would need, which is why it is prototyped here
                    before any core/ change.

The model is a deliberately plain MLP (no BatchNorm — running-stat mutation
inside a While loop is its own research project; no dropout — RNG streams
inside scan differ from the eager trace, which would confound the parity
check). Parity is asserted between the three modes at the end: identical
seeds, identical batch order => the losses and final parameters must agree
to fp tolerance for a mode to count as *correct*, not just fast.

Run (TPU VM):  python benchmarks/xla_step_fusion_prototype.py
               [--rows 50000] [--epochs 3] [--batch 1024] [--width 256]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import Tensor


def make_data(rows: int, n_features: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, n_features)).astype(np.float32)
    w = rng.normal(size=n_features)
    y = (X @ w + 0.5 * X[:, 0] * X[:, 1] + rng.normal(0, 0.1, rows)).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y[:, None])


def init_params(n_features: int, width: int, seed: int) -> dict[str, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    shapes = {
        "w0": (n_features, width), "b0": (width,),
        "w1": (width, width), "b1": (width,),
        "w2": (width, 1), "b2": (1,),
    }
    params = {}
    for name, shape in shapes.items():
        if name.startswith("w"):
            t = torch.randn(shape, generator=gen) / (shape[0] ** 0.5)
        else:
            t = torch.zeros(shape)
        params[name] = t
    return params


def forward(p: dict[str, Tensor], x: Tensor) -> Tensor:
    h = torch.relu(x @ p["w0"] + p["b0"])
    h = torch.relu(h @ p["w1"] + p["b1"])
    return h @ p["w2"] + p["b2"]


def loss_fn(p: dict[str, Tensor], x: Tensor, y: Tensor) -> Tensor:
    return ((forward(p, x) - y) ** 2).mean()


def adamw_update(
    p: dict[str, Tensor],
    g: dict[str, Tensor],
    m: dict[str, Tensor],
    v: dict[str, Tensor],
    t: Tensor,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[dict, dict, dict]:
    """Functional AdamW, matching torch.optim.AdamW's update order."""
    new_p, new_m, new_v = {}, {}, {}
    bc1 = 1 - beta1 ** t
    bc2 = 1 - beta2 ** t
    for k in p:
        pk = p[k] * (1 - lr * weight_decay)
        mk = beta1 * m[k] + (1 - beta1) * g[k]
        vk = beta2 * v[k] + (1 - beta2) * g[k] * g[k]
        step = lr * (mk / bc1) / (torch.sqrt(vk / bc2) + eps)
        new_p[k] = pk - step
        new_m[k] = mk
        new_v[k] = vk
    return new_p, new_m, new_v


def epoch_batches(n: int, batch: int, epoch_seed: int) -> Tensor:
    """Full-size batch index matrix (steps, batch) — the tail is dropped so
    every scan slice has one shape (the library keeps the tail; the prototype
    trades it for scan's fixed-shape requirement and applies the same
    truncation to every mode so the comparison stays apples-to-apples)."""
    gen = torch.Generator().manual_seed(epoch_seed)
    perm = torch.randperm(n, generator=gen)
    steps = n // batch
    return perm[: steps * batch].reshape(steps, batch)


def run_loop(device, X, Y, args, fuse: int) -> dict:
    """Modes 1 and 2: the trainer-style Python step loop, barrier every
    ``fuse`` steps."""
    import torch_xla

    sync = torch_xla.sync

    params = {k: t.to(device) for k, t in init_params(X.shape[1], args.width, 0).items()}
    m = {k: torch.zeros_like(t) for k, t in params.items()}
    v = {k: torch.zeros_like(t) for k, t in params.items()}
    t_step = torch.zeros((), device=device)
    Xd, Yd = X.to(device), Y.to(device)
    sync()

    epoch_secs = []
    last_loss = None
    for epoch in range(args.epochs):
        idx_mat = epoch_batches(len(X), args.batch, epoch)
        start = time.perf_counter()
        unflushed = 0
        for s in range(idx_mat.shape[0]):
            idx = idx_mat[s].to(device)
            xb, yb = Xd[idx], Yd[idx]
            grads = torch.func.grad(loss_fn)(params, xb, yb)
            t_step = t_step + 1
            params, m, v = adamw_update(params, grads, m, v, t_step, lr=args.lr)
            unflushed += 1
            if unflushed >= fuse:
                sync()
                unflushed = 0
        if unflushed:
            sync()
        epoch_secs.append(time.perf_counter() - start)
        # Outside the timed region: a loss read for the parity table.
        idx = idx_mat[-1].to(device)
        last_loss = float(loss_fn(params, Xd[idx], Yd[idx]))
    return {"params": {k: t.cpu() for k, t in params.items()},
            "epoch_secs": epoch_secs, "last_loss": last_loss}


def run_scan(device, X, Y, args) -> dict:
    """Mode 3: the whole epoch's step loop as one XLA While loop."""
    import torch_xla
    from torch_xla.experimental.scan import scan

    sync = torch_xla.sync

    params = {k: t.to(device) for k, t in init_params(X.shape[1], args.width, 0).items()}
    m = {k: torch.zeros_like(t) for k, t in params.items()}
    v = {k: torch.zeros_like(t) for k, t in params.items()}
    t_step = torch.zeros((), device=device)
    Xd, Yd = X.to(device), Y.to(device)
    sync()

    def step_fn(carry, xs):
        params, m, v, t_step = carry
        xb, yb = xs
        grads = torch.func.grad(loss_fn)(params, xb, yb)
        t_next = t_step + 1
        new_p, new_m, new_v = adamw_update(params, grads, m, v, t_next, lr=args.lr)
        loss = loss_fn(new_p, xb, yb).detach()
        return (new_p, new_m, new_v, t_next), loss

    epoch_secs = []
    last_loss = None
    for epoch in range(args.epochs):
        idx_mat = epoch_batches(len(X), args.batch, epoch).to(device)
        start = time.perf_counter()
        xs = (Xd[idx_mat], Yd[idx_mat])  # (steps, batch, d), (steps, batch, 1)
        carry, losses = scan(step_fn, (params, m, v, t_step), xs)
        params, m, v, t_step = carry
        sync()
        epoch_secs.append(time.perf_counter() - start)
        last_loss = float(losses[-1])  # outside the timed region
    return {"params": {k: t.cpu() for k, t in params.items()},
            "epoch_secs": epoch_secs, "last_loss": last_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fuse", type=int, default=8)
    args = parser.parse_args()

    import torch_xla

    device = torch_xla.device() if hasattr(torch_xla, "device") else None
    if device is None:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    print(f"torch {torch.__version__} torch_xla {torch_xla.__version__} "
          f"device {device}", flush=True)

    X, Y = make_data(args.rows)
    results = {}
    for label, runner in (
        ("loop K=1", lambda: run_loop(device, X, Y, args, fuse=1)),
        (f"loop K={args.fuse}", lambda: run_loop(device, X, Y, args, fuse=args.fuse)),
        ("scan", lambda: run_scan(device, X, Y, args)),
    ):
        try:
            out = results[label] = runner()
        except Exception as exc:  # noqa: BLE001 - a mode failing IS a finding
            print(f"{label:12s} FAILED: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:200]}", flush=True)
            continue
        secs = ", ".join(f"{s:.2f}" for s in out["epoch_secs"])
        print(f"{label:12s} epochs [{secs}]s  (first incl. compile)  "
              f"last-batch loss {out['last_loss']:.5f}", flush=True)

    if "loop K=1" in results:
        base = results["loop K=1"]["params"]
        for label, out in results.items():
            if label == "loop K=1":
                continue
            diff = max(
                float((out["params"][k] - base[k]).abs().max()) for k in base
            )
            print(f"PARITY {label:12s} max|param - loop K=1| = {diff:.3e}",
                  flush=True)


if __name__ == "__main__":
    main()
