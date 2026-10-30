#!/bin/bash
# Pre-registration (protocol §7): commit every locked operating point and
# every frozen benchmark manifest to datasets/manifests/ BEFORE any held-out
# benchmark is evaluated. `osb evaluate` refuses to run otherwise.
#
#   bash slurm/preregister.sh titan_open titan_sr_baseline
#   RUN_TAG=killarney_v2 bash slurm/preregister.sh titan_open
#
# Copies, for each ENSEMBLE and RUN_TAG (default "killarney"):
#   $OSB_RUNS_ROOT/calibration/<e>/locked_operating_points.json
#       -> datasets/manifests/locked_operating_points.<e>.<RUN_TAG>.json
#   $OSB_DATA_ROOT/frozen/<b>/manifest.json
#       -> datasets/manifests/<b>.<RUN_TAG>.manifest.json
# and makes a LOCAL commit of exactly those files (compute nodes are
# offline; push from the login node). Re-running with identical files is a
# no-op. Overwriting an already committed file with different content is a
# re-lock and is refused unless ALLOW_RELOCK=1 (the old version stays in git
# history; say why in the protocol before doing it).
#
#   bash slurm/preregister.sh --registered ENSEMBLE
# exits 0 iff the ensemble's locked file is already committed, byte-identical,
# under RUN_TAG. Jobs use it to skip `osb calibrate` on a resubmission, so
# registered thresholds and their decision files are never overwritten.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [[ $# -lt 1 || ( "$1" == --registered && $# -ne 2 ) ]]; then
  echo "usage: bash slurm/preregister.sh ENSEMBLE [ENSEMBLE...]" >&2
  echo "       bash slurm/preregister.sh --registered ENSEMBLE" >&2
  exit 2
fi
RUN_TAG="${RUN_TAG:-killarney}"
DEST=datasets/manifests

# committed SRC DST: true iff DST is in HEAD with exactly SRC's content.
committed() {
  git cat-file -e "HEAD:$2" 2>/dev/null \
    && [[ "$(git rev-parse "HEAD:$2")" == \
          "$(git hash-object --path="$2" "$1")" ]]
}

if [[ "$1" == --registered ]]; then
  src="$OSB_RUNS_ROOT/calibration/$2/locked_operating_points.json"
  [[ -f "$src" ]] && \
    committed "$src" "$DEST/locked_operating_points.$2.$RUN_TAG.json"
  exit
fi
mkdir -p "$DEST"

if ! git config user.email >/dev/null || ! git config user.name >/dev/null
then
  echo "[preregister] git user.name/user.email are not set; run" \
       "'git config --global user.name ...' and '... user.email ...'" >&2
  exit 1
fi

COPIED=()
# copy SRC DST: refuses to replace a committed file with different content.
copy() {
  local src="$1" dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "[preregister] missing $src" >&2
    exit 1
  fi
  if git cat-file -e "HEAD:$dst" 2>/dev/null && ! committed "$src" "$dst" \
     && [[ -z "${ALLOW_RELOCK:-}" ]]; then
    echo "[preregister] $dst is already committed with different content." \
         "Refusing to re-lock; set ALLOW_RELOCK=1 only if no held-out" \
         "benchmark has been evaluated with the committed version." >&2
    exit 1
  fi
  cp "$src" "$dst"
  git add -- "$dst"
  COPIED+=("$dst")
  echo "[preregister] $src -> $dst"
}

for e in "$@"; do
  copy "$OSB_RUNS_ROOT/calibration/$e/locked_operating_points.json" \
    "$DEST/locked_operating_points.$e.$RUN_TAG.json"
done
shopt -s nullglob
manifests=("$OSB_DATA_ROOT"/frozen/*/manifest.json)
if [[ ${#manifests[@]} -eq 0 ]]; then
  echo "[preregister] no frozen manifests under $OSB_DATA_ROOT/frozen" >&2
  exit 1
fi
for m in "${manifests[@]}"; do
  b="$(basename "$(dirname "$m")")"
  copy "$m" "$DEST/$b.$RUN_TAG.manifest.json"
done

if git diff --cached --quiet HEAD -- "${COPIED[@]}"; then
  echo "[preregister] nothing new to pre-register (already in HEAD)."
else
  git commit -m "Pre-register locked operating points ($RUN_TAG)" \
    -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TipqYFMay9xC8nLChjQXmu" \
    -- "${COPIED[@]}"
fi
echo "[preregister] HEAD: $(git log -1 --format='%h %s')"
echo "[preregister] REMINDER: this commit is local only. Push it from the" \
     "login node (git push) so the pre-registration is public."
