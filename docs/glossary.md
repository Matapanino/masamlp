# Glossary

Terms used across masaMLP's docs, ADRs, and verdict reports. Started with
the TPU/XLA design session (ADR 0002/0003); grows as designs do.

## Project terms

- **Differentiator gates** — the three test files that protect what makes
  masaMLP different from pytabkit/pytorch_tabular: `test_sample_weight.py`,
  `test_custom_objective.py`, `test_custom_metric.py`. Any new device or
  execution mode must pass them unchanged.
- **Member sharding** — the 0.3.0 multi-device strategy: ensemble members
  are distributed round-robin across devices and trained concurrently, one
  worker thread per device (`core/parallel.py`). Member parallelism, not
  batch splitting — these models are too small for scatter/gather.
- **Retrieval corpus / candidates** — the training rows tabr/modernnca keep
  in `nn` buffers and search per batch (cdist + topk). Static during fit;
  excluded from early-stopping snapshots via `static_state_keys`.
- **Verdict** — a dated measurement report under `docs/verdicts/` produced
  by a real-hardware verification run; the only place performance claims
  may come from.
- **Zoo run** — training every registered model once on a small seeded
  dataset (`benchmarks/model_zoo.py` or the acceptance scripts) to prove
  breadth, as opposed to depth on one model.

## TPU / XLA terms (ADR 0002, 0003)

- **XLA** — Google's Accelerated Linear Algebra compiler. Traces tensor
  programs into graphs, compiles one executable per tensor-shape signature,
  and replays them. TPUs are programmed through it; `device="xla"` is
  masaMLP's canonical name for any XLA-backed device.
- **torch_xla** — the PyTorch/XLA bridge package; makes XLA devices look
  like torch devices. Strictly version-coupled to torch (minor-for-minor),
  which is why masaMLP declares no `[tpu]` extra and lazily imports it.
- **PJRT** — the runtime layer torch_xla uses to own devices
  (`PJRT_DEVICE=TPU|CPU|...`). Its process/thread topology on multi-core
  TPUs decides whether member sharding can stay thread-based (ADR 0002 §1).
- **XLA:CPU** — the XLA compiler targeting host CPU (`PJRT_DEVICE=CPU`).
  Slow, but runs the identical code path as TPU — masaMLP's CI vehicle for
  XLA regressions without TPU hardware.
- **Lazy tensor mode** — torch_xla's default execution: ops queue into a
  pending graph that only runs at a *sync point*. The alternative is
  dispatching through `torch.compile(backend="openxla")`.
- **Sync point / `mark_step`** — the boundary that cuts the pending lazy
  graph, compiles (first time) and executes it. masaMLP places one per
  optimizer step; the per-epoch loss read stays the only host round-trip.
- **Recompilation** — XLA compiling a fresh graph because a shape changed
  or a Python scalar baked into the graph changed. The enemy of TPU
  throughput; countered by static-shape rewrites and tensor-valued
  schedules (ADR 0003 §2–3).
- **Aten fallback** — an op XLA cannot lower, executed on host CPU with
  device↔host transfers around it. Visible in `met.metrics_report()`;
  "zero fallbacks on the hot path" is an acceptance criterion.
- **`met.metrics_report()`** — torch_xla's counter dump (compiles,
  fallbacks, transfers). Collected by every TPU verdict run.
- **MXU** — a TPU core's matrix unit; bf16-native. The reason `amp="auto"`
  means bf16 autocast on XLA, and fp32-only TPU support would be pointless.
- **bf16 (bfloat16)** — half-storage float with fp32's exponent range; no
  GradScaler needed, unlike fp16.
- **TPU v3-8** — Kaggle's TPU VM offering: 4 chips / 8 cores, 8 XLA
  devices, ~T4-class matmul throughput per core, 16 GB HBM each. The
  hardware masaMLP's TPU verdicts run on (weekly quota, separate from the
  GPU quota).
- **v5e-1 / v6e-1** — single-chip TPUs in Colab's CLI pool; ruled out as
  the experiment vehicle (billable, and single-device so they cannot
  exercise sharding).
- **Zero-weight padding** — the sanctioned (unused) trick for shape-stable
  batching: pad a batch with rows whose sample weight is 0; the trainer's
  weighted reduction `(loss*w).sum()/w.sum()` makes them exactly inert.
