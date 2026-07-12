"""TPU (XLA) speed / correctness matrix, meant for a TPU VM (e.g. Kaggle v5e).

Numbers are comparable with docs/verdicts/2026-07-02-gpu-report.md (T4) and
2026-07-03-gpu-report.md (L4): same data generator and model configs
(``gpu_speed.CONFIGS``). Each XLA run also reports the compile count and any
aten fallbacks (torch_xla metrics) — the acceptance criterion is a small,
epoch-independent compile count and no fallbacks on the hot path — plus
batch-tuned rows for the TPU large-batch story (bigger batches change
convergence; the rmse column keeps that honest).

Cold-cache discipline: one measurement per process (the in-process XLA
compile cache flatters any repeat run — research/tpu-xla.md §9.1), which is
why the 0.5.0 modes are single-measurement flags a driver runs as separate
subprocesses.

Run:  python benchmarks/tpu_speed.py [--rows 50000] [--skip-cpu]
      python benchmarks/tpu_speed.py --retrieval-scale
      python benchmarks/tpu_speed.py --compile-probe
      # 0.5.0 (wave E/F):
      python benchmarks/tpu_speed.py --fuse-one resnet 8     # cold, 1 config
      python benchmarks/tpu_speed.py --fuse-parity           # K=1 vs K=8 numerics
      python benchmarks/tpu_speed.py --predict-amp-one resnet
      python benchmarks/tpu_speed.py --profile-tab-transformer
      python benchmarks/tpu_speed.py --retrieval-eval tabr 8 auto
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from gpu_speed import CONFIGS, make_data, run_one


def _xla_counters() -> tuple[int, list[str]]:
    import torch_xla.debug.metrics as met

    data = met.metric_data("CompileTime")
    compiles = 0 if data is None else data[0]
    aten = sorted(
        f"{n}={met.counter_value(n)}"
        for n in met.counter_names()
        if n.startswith("aten::")
    )
    return compiles, aten


def run_xla(name: str, kwargs: dict, amp, data, label: str | None = None,
            **overrides) -> dict:
    import torch_xla.debug.metrics as met

    met.clear_all()
    out = run_one(name, dict(kwargs, **overrides), "xla", amp, data, label=label)
    compiles, aten = _xla_counters()
    print(f"{'':28s} [xla compiles={compiles} "
          f"aten={';'.join(aten) if aten else 'none'}]", flush=True)
    return out


def matrix(args) -> None:
    numeric = make_data(args.rows, pred_rows=args.pred_rows)
    with_cats = make_data(args.rows, n_cats=8, pred_rows=args.pred_rows)
    for name, kwargs, needs_cats in CONFIGS:
        data = with_cats if needs_cats else numeric
        if not args.skip_cpu:
            run_one(name, dict(kwargs), "cpu", False, data)
        run_xla(name, kwargs, False, data)
        run_xla(name, kwargs, "auto", data)
        # The TPU large-batch story: same model, explicit batch_size.
        run_xla(name, kwargs, "auto", data, label=f"{name} [batch 8192]",
                batch_size=8192)


def retrieval_scale(args) -> None:
    """345k-row tabr/modernnca on the TPU: fit, chunked predict, and the
    bf16-vs-fp32 question (KI-010 was a T4 measurement) at real scale."""
    for rows in args.scale_rows:
        data = make_data(rows, pred_rows=50_000)
        for model in ("tabr", "modernnca"):
            for amp in (False, "auto"):
                run_xla(model, {"model": model, "n_epochs": 3}, amp, data,
                        label=f"{model} rows={rows}")


def compile_probe(args) -> None:
    """Lazy tracing vs torch.compile(backend='openxla') on one config —
    the ADR 0002 mode decision, measured."""
    data = make_data(args.rows, pred_rows=args.pred_rows)
    name, kwargs, _ = next(c for c in CONFIGS if c[0] == "ft_transformer")
    run_xla(name, kwargs, "auto", data, label="ft_transformer [lazy]")
    run_xla(name, kwargs, "auto", data, label="ft_transformer [openxla]",
            compile=True)


# --------------------------------------------------------------------- #
# 0.5.0 — step fusion (xla_fuse_steps)
# --------------------------------------------------------------------- #
def _config_for(model: str) -> tuple[str, dict, bool]:
    for name, kwargs, needs_cats in CONFIGS:
        if kwargs.get("model", name) == model or name == model:
            return name, kwargs, needs_cats
    raise SystemExit(f"no benchmark config for model {model!r}")


def fuse_one(args) -> None:
    """One cold (model, xla_fuse_steps) fit — the driver runs each in its
    own process so every row is a cold-compile-cache measurement."""
    model, k = args.fuse_one
    name, kwargs, needs_cats = _config_for(model)
    data = make_data(args.rows, n_cats=8 if needs_cats else 0,
                     pred_rows=args.pred_rows)
    run_xla(name, kwargs, "auto", data, label=f"{name} [fuse={k}]",
            xla_fuse_steps=int(k))


def fuse_parity(args) -> None:
    """Same seed, K=1 vs fused K: barrier placement must not change the
    result. One process is fine here — parity is about values, not time."""
    from masamlp import MasaRegressor

    for model in ("resnet", "realmlp"):
        name, kwargs, needs_cats = _config_for(model)
        data = make_data(args.rows, n_cats=8 if needs_cats else 0)
        X, y, X_test, y_test, _, cat_idx = data
        preds = {}
        for k in (1, 8):
            kw = dict(kwargs, xla_fuse_steps=k, device="xla", amp="auto",
                      random_state=0, verbose=0)
            if cat_idx:
                kw["categorical_features"] = cat_idx
            est = MasaRegressor(**kw).fit(X, y)
            preds[k] = est.predict(X_test)
            rmse = float(np.sqrt(np.mean((preds[k] - y_test) ** 2)))
            print(f"{name:28s} fuse={k}  rmse {rmse:.6f}", flush=True)
        diff = float(np.abs(preds[1] - preds[8]).max())
        print(f"{name:28s} PARITY max|pred(K=1) - pred(K=8)| = {diff:.3e}",
              flush=True)


# --------------------------------------------------------------------- #
# 0.5.0 — bf16 prediction (amp_predict)
# --------------------------------------------------------------------- #
def predict_amp_one(args) -> None:
    """fp32 vs bf16 prediction on one fitted model: steady-state timing
    (second predict of each dtype; the first pays the compile) + accuracy."""
    from masamlp import MasaRegressor

    name, kwargs, needs_cats = _config_for(args.predict_amp_one)
    data = make_data(args.rows, n_cats=8 if needs_cats else 0,
                     pred_rows=args.pred_rows)
    X, y, X_test, y_test, X_pred, cat_idx = data
    kw = dict(kwargs, device="xla", amp="auto", random_state=0, verbose=0)
    if cat_idx:
        kw["categorical_features"] = cat_idx
    est = MasaRegressor(**kw).fit(X, y)
    out = {}
    for setting, tag in ((False, "fp32"), (True, "bf16")):
        est.set_params(amp_predict=setting)
        est.predict(X_pred)  # compile + cache warmup for this dtype
        start = time.perf_counter()
        pred = est.predict(X_pred)
        sec = time.perf_counter() - start
        rmse = float(np.sqrt(np.mean((est.predict(X_test) - y_test) ** 2)))
        out[tag] = pred
        print(f"{name:28s} predict[{tag}]({len(X_pred)}) {sec:6.2f}s  "
              f"rmse {rmse:.4f}", flush=True)
    diff = np.abs(out["bf16"] - out["fp32"])
    denom = np.maximum(np.abs(out["fp32"]), 1e-6)
    print(f"{name:28s} bf16 vs fp32: max|diff| {diff.max():.3e}  "
          f"max rel {float((diff / denom).max()):.3e}", flush=True)


# --------------------------------------------------------------------- #
# 0.5.0 — retrieval eval at scale (TabR search fusion, OOM watch)
# --------------------------------------------------------------------- #
def retrieval_eval(args) -> None:
    """345k-scale eval-path measurement: fit once (3 epochs), then predict
    with the given eval-barrier interval and amp_predict setting. Run each
    combination in its own process (cold cache; OOM isolation)."""
    from masamlp import MasaRegressor

    model, sync_chunks, amp_predict = args.retrieval_eval
    sync_chunks = int(sync_chunks)
    amp_predict = amp_predict == "bf16"
    data = make_data(args.scale_rows[0], pred_rows=50_000)
    X, y, X_test, y_test, X_pred, _ = data
    est = MasaRegressor(model=model, n_epochs=3, device="xla", amp="auto",
                        amp_predict=amp_predict, random_state=0, verbose=0)
    start = time.perf_counter()
    est.fit(X, y)
    fit_sec = time.perf_counter() - start
    est.model_.xla_eval_sync_chunks = sync_chunks
    start = time.perf_counter()
    pred = est.predict(X_pred)
    pred_sec = time.perf_counter() - start
    rmse = float(np.sqrt(np.mean((est.predict(X_test) - y_test) ** 2)))
    tag = f"{model} sync={sync_chunks} predict={'bf16' if amp_predict else 'fp32'}"
    print(f"{tag:44s} fit {fit_sec:7.1f}s  predict({len(X_pred)}) "
          f"{pred_sec:6.2f}s  rmse {rmse:.4f}  "
          f"max|pred| {float(np.abs(pred).max()):.3g}", flush=True)
    import torch_xla.debug.metrics as met

    data_ct = met.metric_data("CompileTime")
    print(f"{tag:44s} [compiles={0 if data_ct is None else data_ct[0]}]",
          flush=True)


# --------------------------------------------------------------------- #
# 0.5.0 — tab_transformer profile (where do the 69s go?)
# --------------------------------------------------------------------- #
def profile_tab_transformer(args) -> None:
    """Per-section step timing for the verdict tab_transformer config at
    batch 1024: full train step vs embedding / transformer blocks / head
    forwards, each with a per-iteration barrier, plus fallback counters per
    section. Identifies whether the 5x-vs-L4 gap is attention internals,
    the categorical-embedding gathers, or the per-step dispatch floor."""
    import torch_xla
    import torch_xla.debug.metrics as met

    from masamlp.core.device import resolve_device, xla_sync_fn
    from masamlp.models import build_model

    device = resolve_device("xla")
    sync = xla_sync_fn()
    batch = 1024
    n_iter = args.profile_iters
    model = build_model(
        "tab_transformer", {"n_layers": 4, "d_token": 32},
        n_num=32, cat_cardinalities=[10] * 8, out_dim=1, num_embedding=None,
    ).to(device)
    x_num = torch.randn(batch, 32, device=device)
    x_cat = torch.randint(0, 10, (batch, 8), device=device)
    y = torch.randn(batch, 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def wait_device():
        # Block until the device really finished before reading the clock.
        if hasattr(torch_xla, "sync"):
            torch_xla.sync(wait=True)
        else:
            import torch_xla.core.xla_model as xm

            xm.wait_device_ops()

    def timed(label, fn, train_mode=True):
        model.train(train_mode)
        fn()  # compile pass
        sync()
        wait_device()
        met.clear_all()
        start = time.perf_counter()
        for _ in range(n_iter):
            fn()
            sync()
        wait_device()
        sec = time.perf_counter() - start
        compiles, aten = _xla_counters()
        print(f"{label:34s} {1e3 * sec / n_iter:8.2f} ms/iter  "
              f"[compiles={compiles} aten={';'.join(aten) if aten else 'none'}]",
              flush=True)

    def full_step():
        with torch.autocast("xla", dtype=torch.bfloat16):
            out = model(x_num, x_cat)
            loss = ((out - y) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def fwd_only():
        with torch.autocast("xla", dtype=torch.bfloat16), torch.no_grad():
            model(x_num, x_cat)

    def embedding_only():
        with torch.autocast("xla", dtype=torch.bfloat16), torch.no_grad():
            model.embedding(x_num, x_cat)

    with torch.no_grad():
        tokens_const, num_flat_const = model.embedding(x_num, x_cat)
    sync()

    def blocks_only():
        with torch.autocast("xla", dtype=torch.bfloat16), torch.no_grad():
            t = tokens_const
            for block in model.blocks:
                t = block(t)

    def head_only():
        with torch.autocast("xla", dtype=torch.bfloat16), torch.no_grad():
            flat = torch.cat(
                [tokens_const.flatten(1), model.num_norm(num_flat_const)], dim=1
            )
            model.output_layer(model.head(flat))

    timed("full train step (bf16)", full_step)
    timed("forward only (train mode)", fwd_only)
    timed("forward only (eval mode)", fwd_only, train_mode=False)
    timed("embedding only", embedding_only)
    timed("transformer blocks only", blocks_only)
    timed("head only", head_only)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--pred-rows", type=int, default=200_000)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--retrieval-scale", action="store_true")
    parser.add_argument("--scale-rows", type=int, nargs="+", default=[345_000])
    parser.add_argument("--compile-probe", action="store_true")
    parser.add_argument("--fuse-one", nargs=2, metavar=("MODEL", "K"),
                        help="one cold fit with xla_fuse_steps=K")
    parser.add_argument("--fuse-parity", action="store_true",
                        help="same-seed K=1 vs K=8 prediction parity")
    parser.add_argument("--predict-amp-one", metavar="MODEL",
                        help="fp32 vs bf16 prediction on one fitted model")
    parser.add_argument("--retrieval-eval", nargs=3,
                        metavar=("MODEL", "SYNC_CHUNKS", "PREDICT_DTYPE"),
                        help="345k retrieval eval: sync interval x fp32|bf16")
    parser.add_argument("--profile-tab-transformer", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=100)
    args = parser.parse_args()

    import torch_xla

    from masamlp.core.device import xla_backend_type

    print(f"torch {torch.__version__}  torch_xla {torch_xla.__version__}  "
          f"backend={xla_backend_type()}", flush=True)

    if args.retrieval_scale:
        retrieval_scale(args)
    elif args.compile_probe:
        compile_probe(args)
    elif args.fuse_one:
        fuse_one(args)
    elif args.fuse_parity:
        fuse_parity(args)
    elif args.predict_amp_one:
        predict_amp_one(args)
    elif args.retrieval_eval:
        retrieval_eval(args)
    elif args.profile_tab_transformer:
        profile_tab_transformer(args)
    else:
        matrix(args)


if __name__ == "__main__":
    main()
