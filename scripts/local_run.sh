#!/bin/bash
# Overnight single-GPU pilot: TITAN-Open (local) vs the TITAN-SR baseline on
# TITAN-SR's own held-out test split. See docs/LOCAL_RUN.md.
#
#   bash scripts/local_run.sh            # after the data stages have run
#
# Every step is resumable: a backbone whose summary.json exists is not
# retrained, and predictions are only recomputed when missing. Step budgets
# (optimizer steps, after grad accumulation) are env vars so the run fits
# the night; 0 means "use the epoch count from the YAML".
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

CFG=(--config configs/local/benchmark_3080ti.yaml
     --ensemble-config configs/local/titan_ensemble_3080ti.yaml)
SETS="${PREDICT_SETS:-val,titan_test}"
BASELINE_STEPS="${BASELINE_STEPS:-0}"
OPEN_STEPS="${OPEN_STEPS:-0}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1

run_backbone() {
  local backbone="$1" steps="$2"
  local run_dir="runs/$backbone/fold0"
  echo "=== $(date +%T) $backbone (max_steps=$steps)"
  if [[ ! -f "$run_dir/summary.json" ]]; then
    osb train "${CFG[@]}" --backbone "$backbone" --fold 0 --max-steps "$steps"
  fi
  if [[ ! -f "$run_dir/preds.parquet" ]]; then
    osb predict "${CFG[@]}" --backbone "$backbone" --fold 0 --sets "$SETS" \
      --batch-size "${PREDICT_BATCH:-64}"
  fi
}

for b in titan_sr_pubmedbert titan_sr_biolinkbert; do
  run_backbone "$b" "$BASELINE_STEPS"
done
for b in biolinkbert_large modernbert_large; do
  run_backbone "$b" "$OPEN_STEPS"
done

echo "=== $(date +%T) calibrate (thresholds locked on val only)"
osb calibrate "${CFG[@]}" --ensemble titan_sr_baseline
osb calibrate "${CFG[@]}" --ensemble titan_open_local

# Pre-registration: commit the locked operating points (and the frozen
# titan_test manifest) before the held-out split is unblinded. Local commit
# only; the ".local_3080ti" suffix keeps them apart from protocol runs.
for e in titan_sr_baseline titan_open_local; do
  cp "runs/calibration/$e/locked_operating_points.json" \
    "datasets/manifests/locked_operating_points.$e.local_3080ti.json"
done
cp datasets/frozen/titan_test/manifest.json \
  datasets/manifests/titan_test.local_3080ti.manifest.json
git add datasets/manifests/*.local_3080ti*.json
git commit -m "Pre-register locked operating points (local 3080 Ti pilot)" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TipqYFMay9xC8nLChjQXmu" \
  -- datasets/manifests/ || echo "nothing new to pre-register"

echo "=== $(date +%T) evaluate on titan_test"
osb evaluate "${CFG[@]}" --benchmark titan_test
echo "=== $(date +%T) done"
