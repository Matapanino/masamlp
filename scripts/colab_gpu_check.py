#!/usr/bin/env python3
"""On-VM entrypoint for masaMLP CUDA verification (Colab GPU).

Invoked by ``scripts/colab_gpu_check.sh`` via ``colab exec``. Extracts the
uploaded working tree, installs masamlp against Colab's preinstalled
CUDA torch, then:

  1. PYTEST — the CUDA-relevant test files (``test_device.py`` un-skips the
     cpu/cuda parity test here) plus the ensemble suite.
  2. SMOKE — every registered model fits and predicts on ``device="cuda"``;
     one AMP run; one save-on-cuda -> load -> predict roundtrip.
  3. SPEED — ``benchmarks/gpu_speed.py`` (cpu vs cuda vs cuda+AMP; loop vs
     vectorized n_ens=8).

Writes ``/content/gpu_report.md``. Exit code is nonzero if pytest failed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/content/masamlp_repo")
REPORT = Path("/content/gpu_report.md")
_lines: list[str] = []


def log(text: str) -> None:
    print(text, flush=True)
    _lines.append(text)


def sh(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, **kwargs)


def setup() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    sh(["tar", "xzf", "/content/masamlp.tar.gz", "-C", str(WORK)])
    sh([sys.executable, "-m", "pip", "install", "-q", "pytest"])
    # --no-deps: Colab ships a CUDA torch + numpy/pandas/sklearn already.
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(WORK)])


def run_pytest() -> int:
    result = sh(
        [sys.executable, "-m", "pytest", "tests/test_device.py", "tests/test_ensemble.py",
         "tests/test_realmlp.py", "-q"],
        cwd=WORK, capture_output=True, text=True,
    )
    log("## pytest (device / ensemble / realmlp)\n")
    log("```\n" + (result.stdout or "")[-4000:] + "\n```\n")
    return result.returncode


def cuda_smoke() -> None:
    import numpy as np
    import torch

    from masamlp import MasaClassifier, MasaRegressor

    log(f"## CUDA smoke — torch {torch.__version__}, "
        f"{torch.cuda.get_device_name(0)}\n")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 10)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    log("| model | cuda acc | fit s |\n|---|---|---|")
    for name in ["resnet", "realmlp", "ft_transformer", "tab_transformer",
                 "danet", "tabr", "modernnca", "gandalf", "grn", "lnn"]:
        start = time.perf_counter()
        clf = MasaClassifier(model=name, n_epochs=15, device="cuda", random_state=0)
        clf.fit(X[:5000], y[:5000])
        acc = float((clf.predict(X[5000:]) == y[5000:]).mean())
        log(f"| {name} | {acc:.3f} | {time.perf_counter() - start:.1f} |")

    yr = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.1, len(X)).astype(np.float32)
    reg = MasaRegressor(n_epochs=15, device="cuda", amp="auto", random_state=0)
    reg.fit(X[:5000], yr[:5000])
    rmse = float(np.sqrt(np.mean((reg.predict(X[5000:]) - yr[5000:]) ** 2)))
    log(f"\nAMP auto (resnet reg): rmse={rmse:.4f}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        reg.save_model(tmp)
        loaded = MasaRegressor.load_model(tmp)
        ok = np.allclose(reg.predict(X[5000:]), loaded.predict(X[5000:]))
    log(f"save(cuda) -> load -> predict parity: {ok}\n")


def speed_benchmark() -> None:
    result = sh(
        [sys.executable, "benchmarks/gpu_speed.py", "--rows", "50000"],
        cwd=WORK, capture_output=True, text=True,
    )
    log("## gpu_speed.py --rows 50000\n")
    log("```\n" + (result.stdout or "")[-4000:] + "\n```\n")
    if result.returncode != 0:
        log("```\n" + (result.stderr or "")[-2000:] + "\n```\n")


def main() -> int:
    import torch

    log(f"# masaMLP GPU verification — torch {torch.__version__}, "
        f"cuda={torch.cuda.is_available()}\n")
    rc = run_pytest()
    cuda_smoke()
    speed_benchmark()
    log(f"pytest exit code: {rc}")
    REPORT.write_text("\n".join(_lines))
    print(f"report written to {REPORT}", flush=True)
    return rc


if __name__ == "__main__":
    setup()
    sys.exit(main())
