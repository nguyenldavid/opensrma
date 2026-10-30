#!/bin/bash
# One-command Killarney submission of the whole offline pipeline:
#   data_offline -> train/predict arrays (TITAN-Open + TITAN-SR baseline)
#   -> [vLLM verifier sweep] -> calibrate + cascade + evaluate.
# Run on the login node AFTER slurm/run_data_online.sh has finished:
#   bash slurm/submit_all.sh                 # L40S defaults
#   GPU=h100 bash slurm/submit_all.sh        # use H100 nodes for training
#   SKIP_DATA=1 bash slurm/submit_all.sh     # data already built
set -euo pipefail
source "$(dirname "$0")/env.sh"
GPU="${GPU:-l40s}"
if [[ "$GPU" == "h100" ]]; then
  TRAIN_RES=(--gpus-per-node=h100:8 --cpus-per-task=48 --mem=1000G)
else
  TRAIN_RES=(--gpus-per-node=l40s:4 --cpus-per-task=32 --mem=384G)
fi
ENSEMBLES="${ENSEMBLES:-titan_open titan_sr_baseline}"
DEP=""
if [[ -z "${SKIP_DATA:-}" ]]; then
  DATA_JOB=$(sbatch --parsable slurm/data_offline.sbatch)
  echo "data_offline: $DATA_JOB"
  DEP="--dependency=afterok:$DATA_JOB"
fi
TRAIN_JOBS=()
for ENSEMBLE in $ENSEMBLES; do
  N=$(osb plan-runs --ensemble "$ENSEMBLE" | wc -l)
  JOB=$(ENSEMBLE="$ENSEMBLE" sbatch --parsable $DEP --array=0-$((N - 1)) \
        --export=ALL,ENSEMBLE="$ENSEMBLE" "${TRAIN_RES[@]}" \
        slurm/train_array.sbatch)
  echo "train $ENSEMBLE ($N runs): $JOB"
  TRAIN_JOBS+=("$JOB")
done
AFTER=$(IFS=:; echo "${TRAIN_JOBS[*]}")
if [[ -n "${WITH_LLM:-}" ]]; then
  LLM_JOB=$(sbatch --parsable $DEP slurm/llm_verifier.sbatch)
  echo "llm_verifier: $LLM_JOB"
  AFTER="$AFTER:$LLM_JOB"
fi
EVAL_JOB=$(sbatch --parsable --dependency=afterok:"$AFTER" \
           --export=ALL,ENSEMBLES="$ENSEMBLES" slurm/calibrate_eval.sbatch)
echo "calibrate_eval: $EVAL_JOB"
echo "Monitor with: squeue -u $USER   |   logs in $OSB_REPO/logs/"
