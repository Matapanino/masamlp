#!/usr/bin/env bash
# Run masaMLP's CUDA verification on a Google Colab GPU VM and pull back the
# report. There is no local GPU, so this is the way to exercise device="cuda".
#
# Mirrors the catstat/repleafgbm Colab dev loop, hardened against kernel
# websocket drops: the exec only *launches* the job detached on the VM
# (scripts/colab_gpu_launch.py); this loop then polls the incrementally
# written report via `colab download` (fresh connection each time) until the
# completion marker appears.
#
# Usage:
#   bash scripts/colab_gpu_check.sh [--gpu T4|L4|A100] [--session NAME]
#                                   [--keep] [--reuse] [--deadline SEC]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="masamlp-gpu"
KEEP=0
REUSE=0
DEADLINE="${DEADLINE:-2400}"
POLL=45
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        --reuse) REUSE=1; shift ;;
        --deadline) DEADLINE="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

command -v colab >/dev/null 2>&1 || { echo "error: colab CLI not installed" >&2; exit 1; }
if ! git diff --quiet HEAD; then
    echo "error: tracked files differ from HEAD -- commit/stash first." >&2
    git status --short >&2
    exit 1
fi

DATE="$(date +%F)"
REPORT_OUT="docs/verdicts/${DATE}-gpu-report.md"
TARBALL="$(mktemp -t masamlp-XXXXXX).tar.gz"
git archive --format=tar.gz -o "$TARBALL" HEAD

if [[ "$REUSE" -eq 0 ]]; then
    echo ">> provisioning $GPU VM (session: $SESSION)"
    colab new -s "$SESSION" --gpu "$GPU"
fi
stop_vm() { rm -f "$TARBALL"; [[ "$KEEP" -eq 0 ]] && colab stop -s "$SESSION" || true; }
trap stop_vm EXIT

echo ">> uploading working tree"
colab upload -s "$SESSION" "$TARBALL" /content/masamlp.tar.gz

echo ">> launching detached job"
colab exec -s "$SESSION" --timeout 180 -f scripts/colab_gpu_launch.py

echo ">> polling for completion (deadline: ${DEADLINE}s, every ${POLL}s)"
mkdir -p docs/verdicts
START=$(date +%s)
while true; do
    sleep "$POLL"
    ELAPSED=$(( $(date +%s) - START ))
    colab download -s "$SESSION" /content/gpu_report.md "$REPORT_OUT" >/dev/null 2>&1 || true
    if [[ -f "$REPORT_OUT" ]] && grep -q "pytest exit code:" "$REPORT_OUT"; then
        echo ">> job finished after ${ELAPSED}s"
        break
    fi
    if [[ "$ELAPSED" -ge "$DEADLINE" ]]; then
        echo ">> deadline reached (${ELAPSED}s); pulling logs and giving up" >&2
        colab download -s "$SESSION" /content/gpu_run.log "docs/verdicts/${DATE}-gpu-run.log" \
            >/dev/null 2>&1 || true
        exit 1
    fi
    echo "   ... ${ELAPSED}s elapsed"
done

colab download -s "$SESSION" /content/gpu_run.log "docs/verdicts/${DATE}-gpu-run.log" \
    >/dev/null 2>&1 || true
echo ">> report at $REPORT_OUT"
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
