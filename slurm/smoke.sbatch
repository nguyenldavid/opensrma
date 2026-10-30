#!/bin/bash
#SBATCH --job-name=osb-smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# Sanity check of a fresh account/venv: environment report + synthetic
# end-to-end pipeline + unit tests on a GPU node.
set -euo pipefail
source "${OSB_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
osb doctor || true
osb smoke --workdir "$OSB_STORAGE/smoke_run"
python -m pytest -q
