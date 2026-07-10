# Devices: CPU, CUDA (single and multi-GPU), MPS, TPU/XLA (experimental)

`device="auto"` resolves **tpu > cuda > mps > cpu** (TPU only when
torch_xla is installed *and* the environment carries TPU markers, so
non-TPU machines pay nothing). Everything below is handled by
`core/trainer.py` + `core/device.py` + `core/parallel.py`; models never
contain device logic.

## Shared fast path

- All tensors move to the device **once**; minibatches are index slices of
  device-resident tensors. There is no DataLoader and no worker processes —
  for small/medium tabular data that overhead dominates otherwise.
- `batch_size="auto"`: full-batch when the training set has at most 4096
  rows, else 1024. `batch_size=None` forces full-batch, an int forces that
  size.
- One host synchronization per epoch (the loss finiteness check), keeping
  the accelerator pipeline full.

## CUDA

- `amp="auto"` enables bf16 autocast (fp16 + GradScaler on GPUs without
  bf16). Disable with `amp=False`. **The auto policy is per-model**: a model
  class may set `amp_auto = False` to opt out entirely — the retrieval
  models (`tabr`/`modernnca`) do, because autocast around their cdist/topk
  search is slower (KI-010) and fp16 distances lose accuracy — or
  `amp_auto = "bf16"` to accept bf16 but not fp16, as `ft_transformer` does
  (fp16 measured slower and less accurate on T4). An explicit `amp=True`
  still forces AMP on.
- `compile=True` applies `torch.compile`, falling back to eager with a
  warning if the backend fails (lazily, at the first step).
- Shuffling permutations are drawn on CPU from the seeded generator, so a
  run with the same seed visits the same batches on any device.

## Multi-GPU (ensemble-member sharding)

With `device="auto"` (or the index-less `"cuda"`), more than one CUDA
device, and `n_ens > 1`, ensemble members are distributed round-robin
across all GPUs and trained **concurrently** — one worker thread per device,
each training its members sequentially (`core/parallel.py`). This is member
parallelism, not batch splitting: these models are far too small for
DataParallel-style scatter/gather to pay off.

- Each device gets **one** copy of the training split, the eval sets, and
  (for `tabr`/`modernnca`) the retrieval corpus, shared by all of its
  members.
- Prediction after a sharded fit also runs on the members' resident
  devices; results are averaged on CPU. Saved models load onto CPU as
  usual.
- **Opt out with an explicit device**, e.g. `device="cuda:0"`.
- `ens_mode="vectorized"` stays single-device (vmap); loop-mode ensembles
  are the ones that shard.
- `torch.compile` is ignored (warn) in sharded fits; objectives that carry
  `nn.Module`s fall back to the sequential loop (their modules are shared
  across members). Custom objectives/metrics must be thread-safe to train
  sharded — the built-ins are.

## TPU / XLA (experimental)

Requires `torch_xla` (not a masamlp extra — its version is strictly coupled
to torch's, minor for minor; install the pair your platform documents, e.g.
`pip install torch~=2.8.0 torch_xla~=2.8.0` on a Cloud/Kaggle TPU VM).
Design record: ADR 0002/0003; survey: [research/tpu-xla.md](research/tpu-xla.md).

- `device="tpu"` asserts the XLA backend really is a TPU (fail-fast instead
  of silently training on CPU); `device="xla"` accepts any PJRT backend —
  `PJRT_DEVICE=CPU` runs the identical code path hardware-free and is what
  CI uses (`xla-smoke` job).
- Training runs in torch_xla's default lazy-tensor mode with one graph
  barrier per optimizer step. The index-slice batching produces a fixed
  two-shape set per fit (`batch_size`, `n % batch_size`), so everything
  compiles once and replays. `compile=True` switches to
  `torch.compile(backend="openxla")` instead (experimental).
- `amp="auto"` means **bf16 autocast** (TPUs are bf16-native; no GradScaler
  exists on this path). Per-model `amp_auto` policies apply as on CUDA.
  fp16 is never used on TPU.
- `batch_size="auto"` resolves exactly as on every other device — masaMLP
  never changes convergence behavior per device. TPUs like large batches:
  for throughput on big data, set `batch_size` (and re-tune
  `learning_rate`) explicitly.
- All ten models train and predict on XLA. Speed targets and measurements
  cover the matmul-heavy six (resnet, realmlp, ft_transformer,
  tab_transformer, tabr, modernnca) — see `docs/verdicts/`. The
  entmax/sort-based models (danet, gandalf, grn) and lnn work but are not
  tuned for TPU.
- `ens_mode="vectorized"` raises on XLA (torch.func vmap over lazy tensors
  is unvalidated); loop-mode `n_ens` works normally on the single device.
- Multi-core TPUs (e.g. a v3-8's 8 cores): PJRT limits one process to one
  chip, so in-library member sharding is deferred (roadmap). To use the
  whole board today, run one process per chip yourself:
  `TPU_VISIBLE_CHIPS=0 python fit_a.py & TPU_VISIBLE_CHIPS=1 python fit_b.py & ...`
- Reproducibility: same seed + same device ⇒ same result holds on XLA (the
  XLA device RNG is seeded from `random_state`); XLA vs CUDA/CPU results
  are close, not bitwise — the existing cross-device rule.

## MPS (Apple Silicon)

- Supported for training and inference in float32. AMP and compile are
  gated off (warn + fallback). Treated as a development/smoke platform.
- Availability is probed with a real allocation (`mps_functional()`), not
  just `is_available()`: virtualized macOS hosts (GitHub Actions runners)
  advertise MPS but cannot allocate, so `device="auto"` falls back to CPU
  there and the CI smoke test skips itself.

## CPU

- `n_threads` sets `torch.set_num_threads`. `amp=True` opts into bf16
  autocast on CPUs that benefit; default is float32.

## Reproducibility

Same seed + same device + same thread settings => same result. Results
across devices are close but not bitwise equal.

Sharded (multi-GPU) fits are reproducible for a fixed GPU topology: member
weight init is unchanged (seeded sequentially on the main thread), and each
worker seeds only its own device's CUDA generator (`seed_scope="device"`) —
one worker per device means RNG streams never interleave. Sharded vs
single-device results are expected to match on identical GPU models but are
not guaranteed to (see KI-004); the benchmark's `--multi-gpu` mode reports
the observed max-abs prediction difference instead of promising zero.
