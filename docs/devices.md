# Devices: CPU, CUDA (single and multi-GPU), MPS

`device="auto"` resolves **cuda > mps > cpu**. Everything below is handled
by `core/trainer.py` + `core/device.py` + `core/parallel.py`; models never
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
  class may set `amp_auto = False` to opt out — the retrieval models
  (`tabr`/`modernnca`) do, because autocast around their cdist/topk search
  is slower (KI-010) and fp16 distances lose accuracy; `ft_transformer`
  does too (fp16 measured slower and less accurate on T4). An explicit
  `amp=True` still forces AMP on.
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
