# Devices: CPU, CUDA, MPS

`device="auto"` resolves **cuda > mps > cpu**. Everything below is handled
by `core/trainer.py` + `core/device.py`; models never contain device logic.

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
  bf16). Disable with `amp=False`.
- `compile=True` applies `torch.compile`, falling back to eager with a
  warning if the backend fails (lazily, at the first step).
- Shuffling permutations are drawn on CPU from the seeded generator, so a
  run with the same seed visits the same batches on any device.

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
