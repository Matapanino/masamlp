#!/usr/bin/env python3
"""On-VM entrypoint for masaMLP CUDA verification (Colab GPU).

Invoked via ``colab exec -f`` (see scripts/colab_gpu_check.sh). Extracts the
uploaded working tree, then runs three phases, streaming all subprocess
output (``colab exec``'s timeout is an *idle* timeout) and rewriting
``/content/gpu_report.md`` after every phase so a dead exec still leaves a
partial report to download:

  1. PYTEST — the CUDA-relevant test files (``test_device.py`` un-skips the
     cpu/cuda parity test here) plus the ensemble/realmlp suites.
  2. SMOKE — every registered model fits and predicts on ``device="cuda"``;
     one AMP run; one save-on-cuda -> load -> predict roundtrip.
  3. SPEED — ``benchmarks/gpu_speed.py --skip-cpu`` (cuda vs cuda+AMP; loop
     vs vectorized n_ens=8; the 2-vCPU host makes CPU baselines pointless).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/content/masamlp_repo")
REPORT = Path("/content/gpu_report.md")
_lines: list[str] = []


def log(text: str) -> None:
    print(text, flush=True)
    _lines.append(text)


def flush_report() -> None:
    REPORT.write_text("\n".join(_lines))


def sh_stream(cmd: list[str], cwd=None) -> tuple[int, str]:
    """Run a subprocess, echoing each line (feeds the exec idle timeout) and
    returning the captured output."""
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=os.environ.copy(),
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    proc.wait()
    return proc.returncode, "".join(captured)


def setup() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    sh_stream(["tar", "xzf", "/content/masamlp.tar.gz", "-C", str(WORK)])
    # A previous run's editable install can shadow the fresh tree with a
    # broken finder; imports go through PYTHONPATH/src only.
    sh_stream([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "masamlp"])
    sh_stream([sys.executable, "-m", "pip", "install", "-q", "pytest"])
    sys.path.insert(0, str(WORK / "src"))
    os.environ["PYTHONPATH"] = str(WORK / "src")


def run_pytest() -> int:
    rc, out = sh_stream(
        [sys.executable, "-m", "pytest", "tests/test_device.py", "tests/test_ensemble.py",
         "tests/test_realmlp.py", "tests/test_retrieval_cache.py", "tests/test_parallel.py",
         "-q"],
        cwd=WORK,
    )
    log("## pytest (device / ensemble / realmlp / retrieval_cache / parallel)\n")
    log("```\n" + out[-4000:] + "\n```\n")
    return rc


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
    rc, out = sh_stream(
        [sys.executable, "benchmarks/gpu_speed.py", "--rows", "30000", "--skip-cpu"],
        cwd=WORK,
    )
    log("## gpu_speed.py --rows 30000 --skip-cpu\n")
    log("```\n" + out[-4000:] + "\n```\n")


def main() -> int:
    import torch

    log(f"# masaMLP GPU verification — torch {torch.__version__}, "
        f"cuda={torch.cuda.is_available()}\n")
    flush_report()
    rc = 1
    try:
        rc = run_pytest()
    except Exception:
        log("pytest phase crashed:\n```\n" + traceback.format_exc() + "\n```")
    flush_report()
    for name, phase in (("cuda_smoke", cuda_smoke), ("speed", speed_benchmark)):
        try:
            phase()
        except Exception:
            log(f"{name} phase crashed:\n```\n" + traceback.format_exc() + "\n```")
            rc = rc or 1
        flush_report()
    log(f"pytest exit code: {rc}")
    flush_report()
    print(f"report written to {REPORT}", flush=True)
    return rc


if __name__ == "__main__":
    setup()
    sys.exit(main())
