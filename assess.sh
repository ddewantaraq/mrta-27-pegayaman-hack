#!/usr/bin/env bash
# Three-track assess: Track A (structure) + B (Metrica) + C (Sportec).
# Default ref = evaluation/results/LATEST (approved best).
# Usage:
#   bash assess.sh evaluation/results/sm_v2_3_compliant.csv
#   bash assess.sh evaluation/results/sm_v2_3_compliant.csv --ref evaluation/results/sm_v2_2_compliant.csv
#   bash assess.sh evaluation/results/sm_v2_3_compliant.csv --skip-sportec
#   bash assess.sh evaluation/results/sm_v2_3_compliant.csv --fetch-sportec
#   bash assess.sh evaluation/results/sm_v2_3_compliant.csv --public-log
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
  echo "Usage: bash assess.sh <cand.csv> [--ref path] [--skip-metrica] [--skip-sportec] [--fetch] [--fetch-sportec] [--public-log]"
  exit 1
fi

CAND=$1
shift

"$PY" starter/assess.py --cand "$CAND" "$@"
