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
# torch >= 2.6: transformers 5.x silently disables older torch (and peft
# then fails to import); pyproject.toml pins the same floor.
for pkg in "torch>=2.6" numpy pandas pyarrow scikit-learn; do
  pip install --no-index "$pkg" 2>/dev/null || pip install "$pkg"
done
pip install -e "$OSB_REPO[train,lora,llm,xlsx,dev]"
# Fail loudly (not just warn) on the torch / transformers / peft trap.
python - <<'PY' || { echo "[setup_env] FAILED: see above." >&2; exit 1; }
import re
import sys

import torch
import transformers

version = tuple(int(x) for x in re.findall(r"\d+", torch.__version__)[:2])
if version < (2, 6):
    sys.exit(f"torch {torch.__version__} < 2.6: transformers "
             f"{transformers.__version__} disables it. Install torch>=2.6 "
             "(pip install --no-index 'torch>=2.6' or from PyPI).")
if not transformers.utils.is_torch_available():
    sys.exit(f"transformers {transformers.__version__} cannot use torch "
             f"{torch.__version__}.")
try:
    import peft
except Exception as err:  # pylint: disable=broad-except
    sys.exit(f"import peft failed: {err!r}")
print(f"[setup_env] torch {torch.__version__}, transformers "
      f"{transformers.__version__}, peft {peft.__version__}: OK")
PY
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
