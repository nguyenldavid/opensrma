#!/bin/bash
# Shared environment for every Killarney job and login-node script.
# Edit the values in the "site settings" block once; everything else reads
# them. Source it:  source slurm/env.sh
#
# Killarney facts used here (verify with `module spider`, `sacctmgr`):
#   * Login node: killarney.alliancecan.ca. Accounts are AIP RAPs: aip-<pi>.
#   * GPUs: l40s (4 x 48 GB per standard node, 64 cores, 512 GB RAM) and
#     h100 (8 x 80 GB SXM per performance node, 48 cores, 2 TB RAM).
#   * Storage: /home (small), /scratch/$USER (large, purged when inactive),
#     /project/<group> (persistent). Put data on scratch or project.
#   * Treat compute nodes as offline: network stages run on the login node
#     (or a data-transfer node) and models are pre-downloaded into HF_HOME.

# ----------------------------- site settings ------------------------------
export OSB_ACCOUNT="${OSB_ACCOUNT:-aip-CHANGEME}"          # Your AIP RAP.
export OSB_STORAGE="${OSB_STORAGE:-/scratch/$USER/opensrbench}"
export OSB_VENV="${OSB_VENV:-$HOME/venvs/opensrbench}"
export OSB_VLLM_VENV="${OSB_VLLM_VENV:-$HOME/venvs/opensrbench-vllm}"
export OSB_MODULES="${OSB_MODULES:-StdEnv/2023 python/3.11 cuda/12.6 arrow}"
export NCBI_EMAIL="${NCBI_EMAIL:-}"                         # Required by NCBI.
export NCBI_API_KEY="${NCBI_API_KEY:-}"                     # 10 req/s.
# --------------------------------------------------------------------------

export OSB_REPO="${OSB_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export OSB_DATA_ROOT="${OSB_DATA_ROOT:-$OSB_STORAGE/datasets}"
export OSB_RUNS_ROOT="${OSB_RUNS_ROOT:-$OSB_STORAGE/runs}"
export HF_HOME="${HF_HOME:-$OSB_STORAGE/hf_cache}"
export SBATCH_ACCOUNT="$OSB_ACCOUNT"     # sbatch reads this automatically.
export SALLOC_ACCOUNT="$OSB_ACCOUNT"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$OSB_DATA_ROOT" "$OSB_RUNS_ROOT" "$HF_HOME" "$OSB_REPO/logs"

if command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load $OSB_MODULES 2>/dev/null || \
    echo "[env.sh] WARNING: could not load all of: $OSB_MODULES" \
         "(check names with 'module spider')." >&2
fi
if [[ -f "$OSB_VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$OSB_VENV/bin/activate"
fi
# Inside Slurm jobs never touch the network for models.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
fi
cd "$OSB_REPO" || exit 1
