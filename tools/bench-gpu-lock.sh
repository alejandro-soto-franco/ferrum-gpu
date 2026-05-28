#!/usr/bin/env bash
# Locks the primary GPU's graphics clock for bench stability.
# Run before `make bench` or `make perf-gate`; reset on exit.
#
# Deviations from plan:
#   1. `nvidia-smi -lgc/-rgc` requires root. The script wraps both calls
#      in `${SUDO_CMD}` (defaults to `sudo`); set SUDO_CMD= to opt out.
#   2. The plan queries `clocks.base.graphics`, which is not a valid field
#      on the Blackwell driver shipped with the 5060 Laptop (and likely
#      other recent drivers). We query `clocks.max.graphics` instead, and
#      accept a `BASE_CLOCK` env var (MHz) as an explicit override for the
#      lock target (recommended when thermal throttling at max boost
#      would itself confound bench timing on long runs).
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SUDO_CMD="${SUDO_CMD-sudo}"

cleanup() {
    ${SUDO_CMD} nvidia-smi -i "$GPU_ID" -rgc >/dev/null 2>&1 || true
    echo "GPU $GPU_ID clocks reset."
}
trap cleanup EXIT

if [[ -n "${BASE_CLOCK:-}" ]]; then
    BASE="$BASE_CLOCK"
else
    BASE=$(nvidia-smi -i "$GPU_ID" --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | tr -d ' ')
fi
echo "Locking GPU $GPU_ID to graphics clock ${BASE} MHz."
${SUDO_CMD} nvidia-smi -i "$GPU_ID" -lgc "$BASE","$BASE"

if [ "$#" -eq 0 ]; then
    echo "No command given. Press Ctrl-C to release the lock."
    sleep infinity
else
    "$@"
fi
