#!/bin/bash
# Full Open-SR-Bench + TITAN-Open pipeline on ONE workstation GPU
# (default profile: RTX 3080 Ti, 12 GB). Resumable: re-run after any
# interruption and finished steps are skipped.
#
#   bash scripts/run_local.sh                          # everything
#   STEPS="train calibrate dev" bash scripts/run_local.sh
#   HARDWARE=gpu_24gb ENSEMBLES="titan_open titan_sr_baseline" bash scripts/run_local.sh
#
# Steps: data (network + offline build), rct, train, calibrate, dev.
# Held-out benchmarks are never scored here: see the instructions printed
# at the end (pre-register, commit, then `osb evaluate --unblind`).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/benchmark_local.yaml}"
export OSB_HARDWARE="${HARDWARE:-rtx3080ti}"
ENSEMBLES="${ENSEMBLES:-titan_open_lite titan_sr_baseline}"
STEPS="${STEPS:-data rct train calibrate dev}"
DEVICE="${DEVICE:-cuda}"
mkdir -p logs
LOG="logs/run_local_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "config=$CONFIG hardware=$OSB_HARDWARE ensembles=[$ENSEMBLES] steps=[$STEPS]"

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

if has_step data; then
  if [[ -z "${NCBI_EMAIL:-}" ]]; then
    echo "export NCBI_EMAIL=you@org (and ideally NCBI_API_KEY) first" >&2
    exit 1
  fi
  osb data --config "$CONFIG" --stages online
  osb data --config "$CONFIG" --stages offline --device "$DEVICE"
  osb audit --config "$CONFIG"
fi

if has_step rct; then
  osb rct train --config "$CONFIG"
  osb rct score --config "$CONFIG"
  osb rct systems --config "$CONFIG"
fi

if has_step train; then
  for ENSEMBLE in $ENSEMBLES; do
    while read -r BACKBONE FOLD; do
      RUN_DIR="${OSB_RUNS_ROOT:-runs}/$BACKBONE/fold$FOLD"
      if [[ ! -f "$RUN_DIR/summary.json" ]]; then
        echo "== train $BACKBONE fold $FOLD"
        osb train --config "$CONFIG" --backbone "$BACKBONE" --fold "$FOLD"
      fi
      if [[ ! -f "$RUN_DIR/preds.parquet" ]]; then
        echo "== predict $BACKBONE fold $FOLD"
        osb predict --config "$CONFIG" --backbone "$BACKBONE" --fold "$FOLD"
      fi
    done < <(osb plan-runs --ensemble "$ENSEMBLE")
  done
fi

if has_step calibrate; then
  for ENSEMBLE in $ENSEMBLES; do
    osb calibrate --config "$CONFIG" --ensemble "$ENSEMBLE" --rct off
    if [[ "$ENSEMBLE" != titan_sr_baseline ]]; then
      osb calibrate --config "$CONFIG" --ensemble "$ENSEMBLE" --rct auto
    fi
  done
fi

if has_step dev; then
  osb evaluate --config "$CONFIG" --benchmark val \
    --reference "${REFERENCE:-titan_sr_baseline@tau99}"
fi

cat <<'MSG'

Done. Development results: runs/reports/val_T20/leaderboard.md (dev set:
thresholds were locked on it, so it is NOT a held-out estimate).

To report held-out results (each unblinding is logged):
  osb preregister
  git add datasets/manifests && git commit -m "Pre-register thresholds"
  osb evaluate --benchmark internal_test --unblind
  osb evaluate --benchmark open_sr_bench --unblind
  osb evaluate --benchmark titan_test --unblind
MSG
