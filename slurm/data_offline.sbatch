#!/bin/bash
#SBATCH --job-name=osb-data-offline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Offline stages: build (BM25 + MedCPT pool assembly on 1 GPU), family
# graph + split + zero-leakage audit, frozen tiers. Fails loudly on leakage.
set -euo pipefail
source "${OSB_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
osb data --stages offline --device cuda
python - <<'PY'
import json, os
audit = json.load(open(os.path.join(os.environ["OSB_DATA_ROOT"], "splits",
                                    "audit.json")))
leaks = audit["decontamination"]["final_train_vs_protected"]
assert sum(leaks.values()) == 0, f"LEAKAGE: {leaks}"
print("Zero-leakage audit passed:", leaks)
print("Benchmark:", audit["benchmark"])
PY
