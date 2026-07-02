#!/usr/bin/env bash
# Run masaMLP's CUDA verification on a Google Colab GPU VM and pull back the
# report. There is no local GPU, so this is the way to exercise device="cuda".
#
# Mirrors the catstat/repleafgbm Colab dev loop. Requires the Colab CLI:
#   uv tool install google-colab-cli   # or: pip install google-colab-cli
#
# Usage:
#   bash scripts/colab_gpu_check.sh [--gpu T4|L4|A100] [--session NAME] [--keep]
#
# The upload tarball is `git archive HEAD` (committed code only); a dirty
# tracked tree is refused so secrets/artifacts cannot ride along.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="T4"
SESSION="masamlp-gpu"
KEEP=0
EXEC_TIMEOUT="${EXEC_TIMEOUT:-2400}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --session) SESSION="$2"; shift 2 ;;
        --keep) KEEP=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if ! command -v colab >/dev/null 2>&1; then
    echo "error: the 'colab' CLI is not installed." >&2
    echo "  uv tool install google-colab-cli   # or: pip install google-colab-cli" >&2
    exit 1
fi

if ! git diff --quiet HEAD; then
    echo "error: tracked files differ from HEAD -- commit/stash first." >&2
    git status --short >&2
    exit 1
fi

DATE="$(date +%F)"
REPORT_OUT="docs/verdicts/${DATE}-gpu-report.md"
TARBALL="$(mktemp -t masamlp-XXXXXX).tar.gz"
echo ">> archiving committed tree (git archive HEAD) -> $TARBALL"
git archive --format=tar.gz -o "$TARBALL" HEAD

echo ">> provisioning $GPU VM (session: $SESSION)"
colab new -s "$SESSION" --gpu "$GPU"
stop_vm() { rm -f "$TARBALL"; [[ "$KEEP" -eq 0 ]] && colab stop -s "$SESSION" || true; }
trap stop_vm EXIT

echo ">> uploading working tree"
colab upload -s "$SESSION" "$TARBALL" /content/masamlp.tar.gz

echo ">> running GPU verification (watchdog: ${EXEC_TIMEOUT}s)"
colab exec -s "$SESSION" --timeout $((EXEC_TIMEOUT - 100)) -f scripts/colab_gpu_check.py &
exec_pid=$!
( sleep "$EXEC_TIMEOUT"; kill -KILL "$exec_pid" 2>/dev/null ) &
wd_pid=$!
exec_rc=0
wait "$exec_pid" || exec_rc=$?
kill "$wd_pid" 2>/dev/null || true
wait "$wd_pid" 2>/dev/null || true

echo ">> downloading report -> $REPORT_OUT (exec rc=$exec_rc)"
mkdir -p docs/verdicts
colab download -s "$SESSION" /content/gpu_report.md "$REPORT_OUT" \
    || echo "   (no report produced)"

echo ">> done. report at $REPORT_OUT"
if [[ "$KEEP" -eq 1 ]]; then
    echo ">> VM left running (session: $SESSION); 'colab stop -s $SESSION' when done."
fi
exit "$exec_rc"
