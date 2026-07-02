# Benchmarks

Not shipped with the package; the library (`src/`) never imports anything
here.

- `parity_realmlp.py` — the honesty check: masamlp's RealMLP-TD-S recipe vs
  the author's standalone reference implementation (vendored under
  `vendor/`, MIT) on california housing and adult, with sklearn's
  HistGradientBoosting as an anchor. Expected outcome: comparable metrics
  (same recipe, different shuffling details), not bitwise equality.
- `model_zoo.py` — every registered model on the same two datasets with its
  recommended knobs. Single seed, capped epochs, subsampled rows, no HPO:
  a smoke-level leaderboard, not a paper-grade ranking.

Run from the repo root:

```bash
PYTHONPATH=src python3 benchmarks/parity_realmlp.py
PYTHONPATH=src python3 benchmarks/model_zoo.py
```

If OpenML downloads fail with SSL errors (framework Python on macOS):
`export SSL_CERT_FILE=$(python -m certifi)`.
