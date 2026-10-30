#!/bin/bash
#SBATCH --job-name=osb-pilot-offline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# L40S pilot, offline data stages: build (MedCPT on 1 GPU), family graph +
# split + zero-leakage audit, frozen tiers. Fails loudly on leakage.
# Submitted by slurm/pilot_l40s.sh, which exports PILOT_CONFIG_DIR and the
# pilot OSB_DATA_ROOT / OSB_RUNS_ROOT (the local pilot took ~2 min here).
set -euo pipefail
source "${OSB_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
: "${PILOT_CONFIG_DIR:?submit this job with slurm/pilot_l40s.sh}"
osb data --config "$PILOT_CONFIG_DIR/benchmark.yaml" \
  --datasets-config "$PILOT_CONFIG_DIR/datasets.yaml" \
  --stages offline --device cuda
python - <<'PY'
import json, os
root = os.environ["OSB_DATA_ROOT"]
audit = json.load(open(os.path.join(root, "splits", "audit.json")))
for key in ("final_train_vs_protected", "final_train_vs_val"):
    leaks = audit["decontamination"][key]
    assert sum(leaks.values()) == 0, f"LEAKAGE ({key}): {leaks}"
    print(f"Zero-leakage audit passed ({key}):", leaks)
manifest = os.path.join(root, "frozen", "titan_test", "manifest.json")
assert os.path.exists(manifest), f"missing {manifest}"
print("Frozen titan_test:", manifest)
PY
