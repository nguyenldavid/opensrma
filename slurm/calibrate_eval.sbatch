#!/bin/bash
#SBATCH --job-name=osb-calibrate-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Lock thresholds on validation for each ensemble, then evaluate every
# system on every frozen benchmark (5,000 family-bootstrap replicates).
set -euo pipefail
source "${OSB_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
for ENSEMBLE in ${ENSEMBLES:-titan_open titan_sr_baseline}; do
  osb calibrate --ensemble "$ENSEMBLE" --rct off
  if [[ "$ENSEMBLE" != titan_sr_baseline ]]; then
    osb calibrate --ensemble "$ENSEMBLE" --rct auto   # "+rct" variant
  fi
done
if ls "$OSB_RUNS_ROOT"/systems/gpt-oss-20b__condition_b.parquet >/dev/null 2>&1; then
  osb cascade --ensemble "${CASCADE_ENSEMBLE:-titan_open}" \
    --benchmark open_sr_bench
fi
# Development leaderboard (validation = calibration split; not held out).
osb evaluate --benchmark val --reference "${REFERENCE:-titan_sr_baseline@tau99}"
# Held-out benchmarks only with UNBLIND=1, after `osb preregister` + commit.
if [[ "${UNBLIND:-0}" == "1" ]]; then
  for BENCH in internal_test open_sr_bench titan_test prospective; do
    if [[ -f "$OSB_DATA_ROOT/frozen/$BENCH/tiers.parquet" ]]; then
      osb evaluate --benchmark "$BENCH" --unblind \
        --reference "${REFERENCE:-titan_sr_baseline@tau99}" || true
    fi
  done
else
  echo "Held-out benchmarks NOT evaluated. When thresholds are final:"
  echo "  osb preregister && git add datasets/manifests && git commit"
  echo "  UNBLIND=1 sbatch slurm/calibrate_eval.sbatch"
fi
echo "Reports: $OSB_RUNS_ROOT/reports/"
