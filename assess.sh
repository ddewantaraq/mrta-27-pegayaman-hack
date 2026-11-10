#!/usr/bin/env bash
# Dual-track assess: Track A (structure) + Track B (Metrica H/A).
# Usage:
#   bash assess.sh evaluation/results/sm_v1_7_compliant.csv
#   bash assess.sh evaluation/results/sm_v1_7_compliant.csv --skip-metrica
#   bash assess.sh evaluation/results/sm_v1_7_compliant.csv --public-log
set -euo pipefail
cd "$(dirname "$0")"

if command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "FAIL  python not found"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: bash assess.sh <cand.csv> [--skip-metrica] [--fetch] [--public-log] [--ref path]"
  exit 1
fi

CAND=$1
shift

"$PY" starter/assess.py --cand "$CAND" "$@"
