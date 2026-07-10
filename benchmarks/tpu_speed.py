"""TPU (XLA) speed / correctness matrix, meant for a TPU VM (e.g. Kaggle v3-8).

Numbers are comparable with docs/verdicts/2026-07-02-gpu-report.md (T4) and
2026-07-03-gpu-report.md (L4): same data generator and model configs
(``gpu_speed.CONFIGS``). Each XLA run also reports the compile count and any
aten fallbacks (torch_xla metrics) — the acceptance criterion is a small,
epoch-independent compile count and no fallbacks on the hot path — plus
batch-tuned rows for the TPU large-batch story (bigger batches change
convergence; the rmse column keeps that honest).

Run:  python benchmarks/tpu_speed.py [--rows 50000] [--skip-cpu]
      python benchmarks/tpu_speed.py --retrieval-scale
      python benchmarks/tpu_speed.py --compile-probe
"""

from __future__ import annotations

import argparse

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--pred-rows", type=int, default=200_000)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--retrieval-scale", action="store_true")
    parser.add_argument("--scale-rows", type=int, nargs="+", default=[345_000])
    parser.add_argument("--compile-probe", action="store_true")
    args = parser.parse_args()

    import torch_xla

    from masamlp.core.device import xla_backend_type

    print(f"torch {torch.__version__}  torch_xla {torch_xla.__version__}  "
          f"backend={xla_backend_type()}", flush=True)

    if args.retrieval_scale:
        retrieval_scale(args)
    elif args.compile_probe:
        compile_probe(args)
    else:
        matrix(args)


if __name__ == "__main__":
    main()
