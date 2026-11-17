#!/usr/bin/env bash
# Four-track assess: A (structure) + B (Metrica) + C (Sportec) + D (SkillCorner).
# Default ref = evaluation/results/LATEST (approved best = sm_v2_7).
# Usage:
#   bash assess.sh evaluation/results/sm_v2_7_compliant.csv
#   bash assess.sh evaluation/results/sm_v2_7_compliant.csv --ref evaluation/results/sm_v2_6_compliant.csv --skillcorner
#   bash assess.sh evaluation/results/sm_v2_7_compliant.csv --skip-sportec
#   bash assess.sh evaluation/results/sm_v2_7_compliant.csv --fetch-sportec --skillcorner
#   bash assess.sh evaluation/results/sm_v2_7_compliant.csv --public-log
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
  echo "Usage: bash assess.sh <cand.csv> [--ref path] [--skip-metrica] [--skip-sportec] [--skillcorner] [--fetch] [--fetch-sportec] [--public-log]"
  exit 1
fi

CAND=$1
shift

"$PY" starter/assess.py --cand "$CAND" "$@"
