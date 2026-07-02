#!/usr/bin/env python3
"""Tiny on-VM launcher (run via ``colab exec``): extract the uploaded tree
and start scripts/colab_gpu_check.py *detached*, logging to
/content/gpu_run.log. The exec returns in seconds, so a dropped kernel
websocket cannot kill the job; the local loop polls the incrementally
written /content/gpu_report.md instead.
"""

import subprocess
import sys
from pathlib import Path

WORK = Path("/content/masamlp_repo")
WORK.mkdir(parents=True, exist_ok=True)
subprocess.run(["tar", "xzf", "/content/masamlp.tar.gz", "-C", str(WORK)], check=True)

log = open("/content/gpu_run.log", "w")
proc = subprocess.Popen(
    [sys.executable, str(WORK / "scripts" / "colab_gpu_check.py")],
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
)
print(f"launched detached pid={proc.pid}", flush=True)
