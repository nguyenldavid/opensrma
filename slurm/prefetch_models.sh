#!/bin/bash
# Downloads every Hugging Face model used by an ensemble (+ MedCPT and the
# vLLM verifier) into $HF_HOME on the LOGIN node, so GPU jobs run offline.
#   bash slurm/prefetch_models.sh [ensemble ...]   (default: all)
set -euo pipefail
source "$(dirname "$0")/env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
ENSEMBLES="${*:-titan_open titan_open_xl titan_sr_baseline}"
python - "$ENSEMBLES" <<'PY'
import sys
import huggingface_hub
from opensrbench import config
from opensrbench.modeling import spec

cfg = config.Config.load("configs/titan_ensemble.yaml")
names = {"ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder",
         "openai/gpt-oss-20b"}
for ensemble in sys.argv[1].split():
    for backbone in spec.ensemble_members(cfg, ensemble):
        names.add(spec.backbone_spec(cfg, backbone)["model_name"])
names.discard("__tiny_bert__")
for name in sorted(names):
    print("downloading", name, flush=True)
    huggingface_hub.snapshot_download(
        name, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                              "*.py", "*.tiktoken", "tokenizer*", "vocab*",
                              "merges*", "*.bin"] if "gpt-oss" not in name
        else None)
PY
echo "Models cached in $HF_HOME"
