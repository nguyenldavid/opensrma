#!/bin/bash
#SBATCH --job-name=osb-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# One array task = one (backbone, fold): train on all GPUs of the node with
# DDP, then score val + every benchmark with the best checkpoint.
#
#   ENSEMBLE=titan_open sbatch --array=0-14 slurm/train_array.sbatch
#   # H100 node instead of L40S:
#   ENSEMBLE=titan_open_xl sbatch --array=0-19 --gpus-per-node=h100:8 \
#       --cpus-per-task=48 --mem=1000G slurm/train_array.sbatch
# `osb plan-runs --ensemble $ENSEMBLE` lists the index -> run mapping.
set -euo pipefail
source "${OSB_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
ENSEMBLE="${ENSEMBLE:-titan_open}"
read -r BACKBONE FOLD < <(osb plan-runs --ensemble "$ENSEMBLE" \
                                        --index "${SLURM_ARRAY_TASK_ID:-0}")
NGPU=$(nvidia-smi -L | wc -l)
echo "ensemble=$ENSEMBLE backbone=$BACKBONE fold=$FOLD gpus=$NGPU"
RUN_DIR="$OSB_RUNS_ROOT/$BACKBONE/fold$FOLD"
if [[ ! -f "$RUN_DIR/summary.json" ]]; then
  torchrun --standalone --nproc_per_node="$NGPU" -m opensrbench.cli train \
    --backbone "$BACKBONE" --fold "$FOLD" --hardware "${HARDWARE:-auto}" \
    ${TRAIN_ARGS:-}
fi
torchrun --standalone --nproc_per_node="$NGPU" -m opensrbench.cli predict \
  --backbone "$BACKBONE" --fold "$FOLD" --hardware "${HARDWARE:-auto}" \
  --sets "${PREDICT_SETS:-val,internal_test,open_sr_bench,titan_test,prospective}"
