#!/usr/bin/env bash
# Predict → validate → dual-track assess (structure + Metrica).
# Auto out = next version after evaluation/results/LATEST (approved best);
# bare sm_v*.csv preferred; _compliant only if that bare name already exists.
# Optional: PUBLIC_LOG=1 bash predict_v1.sh   # also run public 5-min log
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

echo "=== predict ==="
log=$(mktemp)
set +e
"$PY" starter/predict_v1.py 2>&1 | tee "$log"
py_status=${PIPESTATUS[0]}
set -e
if [[ "$py_status" -ne 0 ]]; then
  echo "FAIL  predict"
  rm -f "$log"
  exit 1
fi

SUBMISSION=$(sed -n 's/^SUBMISSION //p' "$log" | tail -n 1)
rm -f "$log"
if [[ -z "$SUBMISSION" ]]; then
  echo "FAIL  predict did not report a submission path"
  exit 1
fi

echo "=== validate ==="
"$PY" evaluation/validate_submission.py "$SUBMISSION"

echo "=== assess (Track A + Track B) ==="
ASSESS_ARGS=(--cand "$SUBMISSION")
if [[ "${PUBLIC_LOG:-0}" == "1" ]] || [[ "${1:-}" == "--public-log" ]]; then
  ASSESS_ARGS+=(--public-log)
fi
"$PY" starter/assess.py "${ASSESS_ARGS[@]}"

echo "OK  predicted, validated, and assessed $SUBMISSION"
