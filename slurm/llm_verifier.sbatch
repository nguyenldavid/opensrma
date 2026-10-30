#!/bin/bash
#SBATCH --job-name=osb-llm-verifier
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Serves an open-weight verifier with vLLM on this GPU node and runs the
# Condition A/B sweep against it (no API keys, no egress). gpt-oss-20b
# (MXFP4, ~16 GB) also fits one L40S: --gpus-per-node=l40s:1.
set -euo pipefail
source "${OSB_REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
MODEL="${VLLM_MODEL:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-8000}"
export VLLM_BASE_URL="http://127.0.0.1:$PORT/v1"
"$OSB_VLLM_VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
  --max-model-len "${VLLM_MAX_LEN:-16384}" --gpu-memory-utilization 0.90 \
  > "logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "vLLM did not start; see logs/vllm_${SLURM_JOB_ID}.log" >&2; exit 1; }
osb llm-sweep --models "${LLM_MODELS:-gpt-oss-20b}" \
  --conditions "${LLM_CONDITIONS:-condition_a,condition_b}"
