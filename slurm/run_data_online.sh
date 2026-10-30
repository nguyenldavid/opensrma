#!/bin/bash
# Network stages (download, ingest, enrich, pools). Run on the LOGIN node
# inside tmux/screen; it is resumable, so just re-run after interruptions:
#   tmux new -s osb-data 'bash slurm/run_data_online.sh'
# Expect ~1-2 days for the full harvest + pool reconstruction with an NCBI
# API key (most time is polite rate-limited E-utilities traffic).
set -euo pipefail
source "$(dirname "$0")/env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
if [[ -z "$NCBI_EMAIL" ]]; then
  echo "Set NCBI_EMAIL (and ideally NCBI_API_KEY) in slurm/env.sh" >&2
  exit 1
fi
osb data --stages online "$@" 2>&1 | tee -a "logs/data_online_$(date +%F).log"
