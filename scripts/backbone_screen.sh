#!/bin/bash
# Backbone screen (docs/BACKBONE_SCREEN.md): 4 encoders x 2 seeds, 1,500
# steps each, train-fold holdout only. Resumable: finished arms are skipped.
#   setsid nohup bash scripts/backbone_screen.sh > logs/screen.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
# Long DeBERTa batches fragment the 12 GB card; seed 1234 hit OOM without.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/screen

CANDIDATES="${CANDIDATES:-deberta_v3_large biomedbert_large biolinkbert_large
  modernbert_large bioclinical_modernbert_large_512
  bioclinical_modernbert_large}"
for seed in 1234 1235; do
  for b in $CANDIDATES; do
    if grep -qs "\"backbone\": \"$b\", \"seed\": $seed," \
        runs/screen/results.jsonl; then
      continue
    fi
    echo "=== $(date +%T) $b seed=$seed"
    extra=()
    if [[ "$b" == bioclinical_modernbert_large ]]; then
      # 1024 tokens: same global batch 32 as 8 x 4 to fit 12 GB.
      extra=(--override optim.batch_size_per_gpu=8
             --override optim.grad_accum=4)
    fi
    python scripts/screen_arm.py --backbone "$b" --seed "$seed" \
      --max-steps "${SCREEN_STEPS:-1500}" "${extra[@]}" \
      || echo "!!! $b seed=$seed failed"
  done
done
echo "=== $(date +%T) done"
cat runs/screen/results.jsonl
