"""torch.profiler snapshot for one model: ~1 epoch of train steps + predict.

Prints the top ops by self time so the true bottleneck is visible before
any rewrite (the KI-009 discipline: profile first, then change math-free).

Run:  PYTHONPATH=src python3 benchmarks/profile_step.py ft_transformer [--rows 30000]
"""

from __future__ import annotations

import argparse

import torch
from gpu_speed import CONFIGS, make_data

from masamlp import MasaRegressor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=[kw.get("model") for _, kw, _ in CONFIGS
                                          if kw.get("model")])
    parser.add_argument("--rows", type=int, default=30_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="auto")
    parser.add_argument("--row-limit", type=int, default=25)
    args = parser.parse_args()

    name, kwargs, needs_cats = next(c for c in CONFIGS if c[1].get("model") == args.model)
    data = make_data(args.rows, n_cats=8 if needs_cats else 0, pred_rows=50_000)
    X, y, _, _, X_pred, cat_idx = data
    # batch_size default 1024 -> ~0.8*rows/1024 steps in the single epoch.
    kwargs = dict(kwargs, n_epochs=1)
    if cat_idx:
        kwargs["categorical_features"] = cat_idx
    est = MasaRegressor(random_state=0, device=args.device, amp=args.amp, **kwargs)

    activities = [torch.profiler.ProfilerActivity.CPU]
    sort_by = "self_cpu_time_total"
    if torch.cuda.is_available() and args.device in ("auto", "cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_by = "self_cuda_time_total"

    print(f"== {name}: fit (1 epoch) ==", flush=True)
    with torch.profiler.profile(activities=activities) as prof:
        est.fit(X, y)
    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit))

    print(f"\n== {name}: predict ({len(X_pred)} rows) ==", flush=True)
    with torch.profiler.profile(activities=activities) as prof:
        est.predict(X_pred)
    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit))


if __name__ == "__main__":
    main()
