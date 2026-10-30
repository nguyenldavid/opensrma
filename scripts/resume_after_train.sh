#!/bin/bash
# Waits for the given PID (a running `osb train`) to exit, then resumes
# scripts/local_run.sh (finished backbones are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
BASELINE_STEPS="${BASELINE_STEPS:-2800}" OPEN_STEPS="${OPEN_STEPS:-4000}" \
  exec bash scripts/local_run.sh
