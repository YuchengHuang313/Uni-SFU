#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEG_MIN="${SEG_MIN:-1}"
SEG_MAX="${SEG_MAX:-16}"
DEGREE_MAX="${DEGREE_MAX:-3}"
PHASE3_ALGO="${PHASE3_ALGO:-bestfirst}"
REFINE_PLAUSIBLE="${REFINE_PLAUSIBLE:-0}"
BINARY_SEARCH="${BINARY_SEARCH:-0}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 MAX_ERROR [MAX_ERROR ...]"
  echo "Example: $0 1e-10 5e-10 1e-9 5e-9"
  echo
  echo "Optional env vars:"
  echo "  PYTHON_BIN=python3"
  echo "  SEG_MIN=1"
  echo "  SEG_MAX=16"
  echo "  DEGREE_MAX=3"
  echo "  PHASE3_ALGO=bestfirst"
  echo "  REFINE_PLAUSIBLE=1"
  echo "  BINARY_SEARCH=1"
  exit 1
fi

extra_args=()
if [[ "${REFINE_PLAUSIBLE}" == "1" ]]; then
  extra_args+=(--refine-plausible)
fi
if [[ "${BINARY_SEARCH}" == "1" ]]; then
  extra_args+=(--binary-search)
fi

for max_error in "$@"; do
  echo
  echo "============================================================"
  echo "Running optimize_piecewise_activations.py with max_error=${max_error}"
  echo "============================================================"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/optimize_piecewise_activations.py" \
    --max-error "${max_error}" \
    --seg-min "${SEG_MIN}" \
    --seg-max "${SEG_MAX}" \
    --degree-max "${DEGREE_MAX}" \
    --phase3-algo "${PHASE3_ALGO}" \
    "${extra_args[@]}"
done
