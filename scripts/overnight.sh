#!/bin/bash
# Waits for a running `osb data --stages enrich`, then runs the offline data
# stages (build, split + zero-leakage audit, freeze) and scripts/local_run.sh.
# Detach it so it survives the terminal:
#   setsid nohup bash scripts/overnight.sh > logs/overnight.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

while pgrep -f "osb data .*--stages enrich" >/dev/null; do sleep 30; done
if grep -q "Traceback" logs/data_enrich.log; then
  echo "enrich failed; see logs/data_enrich.log" >&2
  exit 1
fi

echo "=== $(date +%T) offline data stages"
osb data --config configs/local/benchmark_3080ti.yaml \
  --datasets-config configs/local/datasets_3080ti.yaml \
  --stages build,split,freeze --device cuda

BASELINE_STEPS="${BASELINE_STEPS:-2800}" OPEN_STEPS="${OPEN_STEPS:-4000}" \
  bash scripts/local_run.sh
