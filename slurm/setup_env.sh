#!/bin/bash
# One-time environment setup on the Killarney LOGIN node.
#   bash slurm/setup_env.sh            # core + training deps
#   bash slurm/setup_env.sh --vllm     # also a separate vLLM venv (verifier)
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ ! -d "$OSB_VENV" ]]; then
  python -m venv "$OSB_VENV" 2>/dev/null || virtualenv --no-download "$OSB_VENV"
fi
source "$OSB_VENV/bin/activate"
pip install --upgrade pip
# Prefer the Alliance wheelhouse (cluster-optimised builds), else PyPI.
for pkg in torch numpy pandas pyarrow scikit-learn; do
  pip install --no-index "$pkg" 2>/dev/null || pip install "$pkg"
done
pip install -e "$OSB_REPO[train,lora,llm,xlsx,dev]"
# FlashAttention-2 speeds up ModernBERT / Qwen3 on H100 + L40S (optional).
pip install --no-index flash-attn 2>/dev/null || \
  pip install flash-attn --no-build-isolation 2>/dev/null || \
  echo "flash-attn not installed; models fall back to SDPA attention."

if [[ "${1:-}" == "--vllm" ]]; then
  deactivate
  python -m venv "$OSB_VLLM_VENV" 2>/dev/null || \
    virtualenv --no-download "$OSB_VLLM_VENV"
  source "$OSB_VLLM_VENV/bin/activate"
  pip install --upgrade pip
  pip install vllm
fi
echo "Environment ready. Next: bash slurm/prefetch_models.sh"
osb doctor --config "$OSB_REPO/configs/benchmark_v1.yaml" || true
