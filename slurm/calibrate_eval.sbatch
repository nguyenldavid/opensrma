#!/bin/bash
#SBATCH --job-name=osb-calibrate-eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# Lock thresholds on validation for each ensemble, pre-register them (local
# git commit of datasets/manifests/, protocol §7), then evaluate every
# system on every frozen benchmark (5,000 family-bootstrap replicates).
# `osb evaluate` refuses to run on thresholds that are not committed, and
# any failing step fails the job. Push the pre-registration commit from the
# login node afterwards.
#
# Resubmitting after a failure is safe: an ensemble whose locked file is
# already committed byte-identical (RUN_TAG, default "killarney") is NOT
# re-calibrated, so its registered thresholds and decision files stay as
# evaluated. To re-lock deliberately, see ALLOW_RELOCK in preregister.sh.
#
# The pre-specified secondary thresholds (A19, runs/systems_secondary/) are
# evaluated separately into reports/<benchmark>_T<k>_secondary/ against
# SECONDARY_REFERENCE (default titan_sr_baseline@pooled_conformal.tau99).
set -euo pipefail
source "${OSB_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
ENSEMBLES="${ENSEMBLES:-titan_open titan_sr_baseline}"
for ENSEMBLE in $ENSEMBLES; do
  if bash slurm/preregister.sh --registered "$ENSEMBLE"; then
    echo "[calibrate] $ENSEMBLE already pre-registered; not re-calibrating."
    continue
  fi
  osb calibrate --ensemble "$ENSEMBLE"
done
# shellcheck disable=SC2086
bash slurm/preregister.sh $ENSEMBLES
if ls "$OSB_RUNS_ROOT"/systems/gpt-oss-20b__condition_b.parquet >/dev/null 2>&1; then
  osb cascade --ensemble "${CASCADE_ENSEMBLE:-titan_open}" \
    --benchmark open_sr_bench
fi
for BENCH in open_sr_bench titan_test prospective; do
  if [[ -f "$OSB_DATA_ROOT/frozen/$BENCH/tiers.parquet" ]]; then
    osb evaluate --benchmark "$BENCH" \
      --reference "${REFERENCE:-titan_sr_baseline@tau99}"
    if compgen -G "$OSB_RUNS_ROOT/systems_secondary/*.parquet" >/dev/null
    then
      osb evaluate --benchmark "$BENCH" --tag secondary \
        --systems "$OSB_RUNS_ROOT/systems_secondary/*.parquet" \
        --reference \
        "${SECONDARY_REFERENCE:-titan_sr_baseline@pooled_conformal.tau99}"
    fi
  fi
done
echo "Reports: $OSB_RUNS_ROOT/reports/"
echo "REMINDER: push the pre-registration commit from the login node."
