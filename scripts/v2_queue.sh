#!/bin/bash
# GPU queue after the backbone screen: trains on the rebuilt pilot_v2 data
# (TITAN-SR + Chan with the Track B fixes) at the protocol epoch budgets,
# predicts validation only and calibrates on validation. No titan_test
# prediction or evaluation (the Killarney pilot is the titan_test run).
#   setsid nohup bash scripts/v2_queue.sh > logs/v2_queue.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
FNX="${FNX_ROOT:-/mnt/FNX/dav_Projects_2026/opensrma}"
export OSB_DATA_ROOT="$FNX/data_pilot_v2" OSB_RUNS_ROOT="$FNX/runs_v2"
CFG=(--config configs/killarney/benchmark_pilot.yaml
     --ensemble-config configs/local/titan_ensemble_3080ti.yaml)
# Large encoders: same memory setup as the screen (global batch 32).
LARGE=(--override optim.batch_size_per_gpu=16 --override optim.grad_accum=2
       --override gradient_checkpointing=true)

echo "=== $(date +%T) waiting for the screen and the pilot_v2 data"
while pgrep -f "backbone_screen.sh" > /dev/null; do sleep 60; done
until [[ -f "$OSB_DATA_ROOT/frozen/titan_test/manifest.json" ]]; do
  sleep 60
done

# Committed rule (docs/BACKBONE_SCREEN.md) applied to runs/screen.
mkdir -p "$OSB_RUNS_ROOT"
python - > "$OSB_RUNS_ROOT/screen_decision.txt" <<'EOF'
import collections, json
rows = [json.loads(l) for l in open("runs/screen/results.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    by[r["backbone"]].append(r["auc_per_review"])
means = {b: sum(v) / len(v) for b, v in by.items()}
best = max(means.values())
ranked = sorted(means, key=means.get, reverse=True)
passing = [b for b in ranked if means[b] >= best - 0.05]
print("DECISION screen means:", {b: round(means[b], 4) for b in ranked},
      "seeds:", {b: len(by[b]) for b in ranked})
print("DECISION titan_open:", " ".join(passing[:3]))
EOF
cat "$OSB_RUNS_ROOT/screen_decision.txt"
OPEN=$(sed -n 's/^DECISION titan_open: //p' "$OSB_RUNS_ROOT/screen_decision.txt")

run() {  # $1 = backbone, rest = extra train args
  local b="$1"; shift
  local dir="$OSB_RUNS_ROOT/$b/fold0"
  echo "=== $(date +%T) train $b"
  if [[ ! -f "$dir/summary.json" ]]; then
    osb train "${CFG[@]}" --backbone "$b" --fold 0 "$@" \
      || { echo "!!! $b training failed"; return 0; }
  fi
  if [[ ! -f "$dir/preds.parquet" ]]; then
    osb predict "${CFG[@]}" --backbone "$b" --fold 0 --sets val \
      --batch-size 64 || { echo "!!! $b prediction failed"; return 0; }
  fi
  echo "DONE $(date +%T) $b $(tr -d '\n' < "$dir/summary.json")"
}

for b in titan_sr_pubmedbert titan_sr_biolinkbert; do run "$b"; done
for b in $OPEN; do run "$b" "${LARGE[@]}"; done
echo "DONE $(date +%T) v2 queue finished; members: titan_sr x2 + $OPEN"
