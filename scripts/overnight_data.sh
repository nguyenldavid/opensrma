#!/bin/bash
# Overnight data jobs on /mnt/FNX (no GPU; NCBI stages strictly one at a
# time so the combined rate stays <= 3 req/s):
#   1. pilot_v2: TITAN-SR + Chan rebuilt with the Track B / empty-text fixes
#      (ingest, enrich, build, split + audit, freeze; CPU).
#   2. protocol: every dataset + Europe PMC harvest (download, ingest,
#      enrich). Pools are left to Killarney.
# Needs NCBI_EMAIL in the environment. Resumable: re-run the same command.
#   setsid nohup bash scripts/overnight_data.sh > logs/overnight_data.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
: "${NCBI_EMAIL:?set NCBI_EMAIL}"
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
FNX="${FNX_ROOT:-/mnt/FNX/dav_Projects_2026/opensrma}"
mkdir -p "$FNX/shared_cache"

# Seed the PubMed cache once from the pilot (keeps 359k fetched records).
if [[ ! -f "$FNX/shared_cache/pubmed.sqlite" ]]; then
  cp datasets/cache/pubmed.sqlite "$FNX/shared_cache/pubmed.sqlite"
fi
link_cache() {  # $1 = data root
  mkdir -p "$1"
  [[ -e "$1/cache" ]] || ln -s "$FNX/shared_cache" "$1/cache"
}

echo "=== $(date +%T) pilot_v2"
V2="$FNX/data_pilot_v2"
link_cache "$V2"
[[ -e "$V2/raw" ]] || cp -a datasets/raw "$V2/raw"
P=(--config configs/killarney/benchmark_pilot.yaml
   --datasets-config configs/killarney/datasets_pilot.yaml)
OSB_DATA_ROOT="$V2" osb data "${P[@]}" --stages ingest,enrich
OSB_DATA_ROOT="$V2" osb data "${P[@]}" --stages build,split,freeze \
  --device cpu

echo "=== $(date +%T) protocol (download, ingest, enrich)"
PR="$FNX/data_protocol"
link_cache "$PR"
OSB_DATA_ROOT="$PR" osb data --config configs/local/benchmark_fnx_protocol.yaml \
  --stages download,ingest,enrich
echo "=== $(date +%T) done"
