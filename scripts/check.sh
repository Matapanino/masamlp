#!/usr/bin/env bash
# One-command verification: lint + tests + end-to-end example.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m ruff check src tests examples
python -m pytest tests/ -q
python examples/quickstart.py
echo "check.sh: all green"
