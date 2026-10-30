#!/bin/bash
#SBATCH --job-name=osb-pilot-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# L40S pilot: one array task = one backbone (fold 0) on ONE GPU
# (world_size 1, no torchrun): train with the protocol schedule, then score
# val + titan_test with the best checkpoint. Submitted by
# slurm/pilot_l40s.sh, which exports PILOT_CONFIG_DIR, PILOT_BACKBONES
# (array index -> backbone) and the pilot OSB_DATA_ROOT / OSB_RUNS_ROOT.
#
# Resumable like scripts/local_run.sh: training is skipped when
# summary.json exists, prediction when preds.parquet exists. There is no
# mid-run checkpoint resume: a task killed by the walltime retrains from
# scratch, so cap long members with MAX_STEPS_<backbone>.
#
# Per-backbone knobs (exported by the caller; 0 / empty = protocol):
#   MAX_STEPS_<backbone>=N        optimizer-step cap (osb train --max-steps)
#   TRAIN_ARGS_<backbone>="..."   extra osb train flags, e.g.
#                                 "--override gradient_checkpointing=true"
#   TRAIN_ARGS="..."              same, for every backbone without its own
#   PREDICT_BATCH=128             osb predict --batch-size
set -euo pipefail
source "${OSB_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
: "${PILOT_CONFIG_DIR:?submit this job with slurm/pilot_l40s.sh}"
: "${PILOT_BACKBONES:?submit this job with slurm/pilot_l40s.sh}"
read -ra BACKBONES <<< "$PILOT_BACKBONES"
BACKBONE="${BACKBONES[${SLURM_ARRAY_TASK_ID:-0}]}"
CFG=(--config "$PILOT_CONFIG_DIR/benchmark.yaml"
     --ensemble-config "$PILOT_CONFIG_DIR/titan_ensemble.yaml")
STEPS_VAR="MAX_STEPS_${BACKBONE}"
ARGS_VAR="TRAIN_ARGS_${BACKBONE}"
STEPS="${!STEPS_VAR:-0}"
EXTRA="${!ARGS_VAR:-${TRAIN_ARGS:-}}"
RUN_DIR="$OSB_RUNS_ROOT/$BACKBONE/fold0"
echo "backbone=$BACKBONE max_steps=$STEPS extra='$EXTRA' run_dir=$RUN_DIR"
nvidia-smi -L
if [[ ! -f "$RUN_DIR/summary.json" ]]; then
  # shellcheck disable=SC2086
  osb train "${CFG[@]}" --backbone "$BACKBONE" --fold 0 \
    --max-steps "$STEPS" $EXTRA
fi
if [[ ! -f "$RUN_DIR/preds.parquet" ]]; then
  osb predict "${CFG[@]}" --backbone "$BACKBONE" --fold 0 \
    --sets val,titan_test --batch-size "${PREDICT_BATCH:-128}"
fi
echo "done: $BACKBONE"
