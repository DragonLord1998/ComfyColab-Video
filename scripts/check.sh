#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

cd "$ROOT"
"$PYTHON_BIN" -m compileall -q custom_nodes runtime scripts tests
"$PYTHON_BIN" -m unittest discover -s tests -v
bash -n scripts/check.sh
