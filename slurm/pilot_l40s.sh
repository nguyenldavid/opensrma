#!/bin/bash
# One-command TITAN-Open vs TITAN-SR pilot on Killarney L40S GPUs: TITAN-SR
# + Chan 2024 data only, compared on TITAN-SR's own held-out test split
# (the local 3080 Ti pilot's scope, docs/LOCAL_RUN.md, with protocol model
# settings). Run on the LOGIN node from the repo root, inside tmux (the
# network stages need internet and take up to an hour):
#   tmux new -s osb-pilot \
#     'bash slurm/pilot_l40s.sh 2>&1 | tee -a logs/pilot_l40s.log'
#   bash slurm/pilot_l40s.sh --dry-run      # print the plan, submit nothing
#
# Steps; each is skipped when already done, so re-running is safe once no
# pilot job is queued (the script refuses otherwise; it also validates every
# job with `sbatch --test-only` before queuing any):
#   1. Resolve configs/killarney/*_pilot.yaml onto the protocol YAMLs into
#      $OSB_PILOT_ROOT/configs/ (refuses if they changed since the last run)
#      and cache any missing Hugging Face model (login node).
#   2. download, ingest, enrich (login node, network).
#   3. sbatch build, split, freeze on 1 L40S (slurm/pilot_offline.sbatch).
#   4. sbatch one 1-GPU array task per backbone: train, then predict val +
#      titan_test (slurm/pilot_train.sbatch), afterok on 3.
#   5. sbatch this script with --finalize, afterok on 4: osb calibrate for
#      both ensembles, slurm/preregister.sh (local git commit), osb evaluate
#      --benchmark titan_test.
#
# Environment knobs (defaults in brackets):
#   OSB_PILOT_ROOT     data, runs and configs [$OSB_STORAGE/killarney_pilot]
#   OPEN_MEMBERS       replace titan_open's members, e.g.
#                      "biomedbert_large biolinkbert_large deberta_v3_large"
#   SR_MEMBERS         replace titan_sr_baseline's members
#   MAX_STEPS_<bb>     optimizer-step cap for backbone <bb> [0 = epochs]
#   TRAIN_ARGS_<bb>    extra `osb train` flags for <bb> (see pilot_train)
#   TRAIN_TIME [24:00:00]  OFFLINE_TIME [04:00:00]  FINAL_TIME [06:00:00]
#   FINAL_SBATCH_ARGS  extra sbatch flags for the final (CPU) job
#   SKIP_ONLINE=1 / SKIP_OFFLINE=1 / SKIP_TRAIN=1  skip a finished step
#   PILOT_FORCE_CONFIG=1  overwrite resolved configs that differ
#   PILOT_ALLOW_QUEUED=1  submit although pilot jobs are already queued
#   RUN_TAG [killarney_pilot]  pre-registration file suffix
set -euo pipefail

MODE=submit
case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry ;;
  --finalize) MODE=finalize ;;
  -h | --help) sed -n '2,/^set -euo/{/^set/d;s/^# \{0,1\}//;p;}' "$0"; exit 0 ;;
  *) echo "usage: bash slurm/pilot_l40s.sh [--dry-run]" >&2; exit 2 ;;
esac

# OSB_REPO is exported into the --finalize job (Slurm runs a spool copy).
REPO_DIR="${OSB_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$REPO_DIR/slurm/env.sh"
export OSB_PILOT_ROOT="${OSB_PILOT_ROOT:-$OSB_STORAGE/killarney_pilot}"
export OSB_DATA_ROOT="$OSB_PILOT_ROOT/datasets"
export OSB_RUNS_ROOT="$OSB_PILOT_ROOT/runs"
export PILOT_CONFIG_DIR="${PILOT_CONFIG_DIR:-$OSB_PILOT_ROOT/configs}"
export RUN_TAG="${RUN_TAG:-killarney_pilot}"
ENSEMBLES=(titan_open titan_sr_baseline)
CFG=(--config "$PILOT_CONFIG_DIR/benchmark.yaml"
     --ensemble-config "$PILOT_CONFIG_DIR/titan_ensemble.yaml")
DATA_CFG=(--config "$PILOT_CONFIG_DIR/benchmark.yaml"
          --datasets-config "$PILOT_CONFIG_DIR/datasets.yaml")

die() { echo "[pilot] ERROR: $*" >&2; exit 1; }

# ------------------------------- final job --------------------------------
if [[ "$MODE" == finalize ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] || die "--finalize runs inside Slurm only."
  for e in "${ENSEMBLES[@]}"; do
    # A resubmitted final job must not overwrite registered thresholds.
    if bash slurm/preregister.sh --registered "$e"; then
      echo "=== $e already pre-registered ($RUN_TAG); not re-calibrating"
      continue
    fi
    echo "=== $(date +%T) calibrate $e (thresholds locked on val only)"
    osb calibrate "${CFG[@]}" --ensemble "$e"
  done
  echo "=== $(date +%T) pre-register (local commit; push from login node)"
  bash slurm/preregister.sh "${ENSEMBLES[@]}"
  echo "=== $(date +%T) evaluate titan_test"
  osb evaluate "${CFG[@]}" --benchmark titan_test
  echo "=== $(date +%T) evaluate titan_test, secondary thresholds (A19)"
  osb evaluate "${CFG[@]}" --benchmark titan_test --tag secondary \
    --systems "$OSB_RUNS_ROOT/systems_secondary/*.parquet" \
    --reference titan_sr_baseline@pooled_conformal.tau99
  echo "Reports: $OSB_RUNS_ROOT/reports/"
  echo "REMINDER: push the pre-registration commit from the login node."
  exit 0
fi

# ------------------------------ login node --------------------------------
[[ -z "${SLURM_JOB_ID:-}" ]] || die "run this on the login node, not in a job."
[[ -n "$OSB_ACCOUNT" && "$OSB_ACCOUNT" != *CHANGEME* ]] || \
  die "set OSB_ACCOUNT (aip-<pi>) in slurm/env.sh."
[[ -n "$NCBI_EMAIL" ]] || die "set NCBI_EMAIL in slurm/env.sh."
command -v osb >/dev/null || \
  die "osb not found; run bash slurm/setup_env.sh first."
if [[ "$MODE" == submit ]]; then
  command -v sbatch >/dev/null || die "sbatch not found (not on Killarney?)."
  # The final job commits the pre-registration; fail now, not after training.
  git config user.name >/dev/null && git config user.email >/dev/null || \
    die "set git config --global user.name / user.email (pre-registration" \
        "commit)."
fi
if [[ -z "$NCBI_API_KEY" ]]; then
  echo "[pilot] WARNING: NCBI_API_KEY unset; enrich runs at 3 req/s." >&2
fi

# 1. Resolve the overlays into reviewable copies (records member
# deviations; detects protocol edits mid-pilot). The checks live in
# src/opensrbench/pilot.py (tested in tests/config_test.py).
RESOLVE_DIR="$PILOT_CONFIG_DIR"
if [[ "$MODE" == dry ]]; then
  RESOLVE_DIR="$(mktemp -d)"
  trap 'rm -rf "$RESOLVE_DIR"' EXIT
fi
mkdir -p "$RESOLVE_DIR"
[[ "$MODE" == dry ]] || mkdir -p "$OSB_DATA_ROOT" "$OSB_RUNS_ROOT"
echo "=== $(date +%T) resolve configs/killarney/*_pilot.yaml -> $RESOLVE_DIR"
PILOT_BACKBONES=$(python - "$OSB_REPO/configs/killarney" "$RESOLVE_DIR" \
                  "$MODE" <<'PY'
"""Deep-merges each pilot overlay onto its protocol file and checks it."""
import difflib
import os
import sys

import yaml

from opensrbench import config
from opensrbench import pilot
from opensrbench.modeling import spec

overlay_dir, out_dir, mode = sys.argv[1:4]
OVERLAYS = {
    "benchmark.yaml": "benchmark_pilot.yaml",
    "datasets.yaml": "datasets_pilot.yaml",
    "titan_ensemble.yaml": "titan_ensemble_pilot.yaml",
}
MEMBER_VARS = {"titan_open": "OPEN_MEMBERS", "titan_sr_baseline": "SR_MEMBERS"}


def log(msg):
    print(f"[pilot] {msg}", file=sys.stderr, flush=True)


def load_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


stale = []
for out_name, overlay_name in OVERLAYS.items():
    overlay_path = os.path.join(overlay_dir, overlay_name)
    base_path = os.path.normpath(
        os.path.join(overlay_dir, load_yaml(overlay_path)["extends"]))
    base = config.load_yaml(base_path)
    data = config.load_yaml(overlay_path)
    notes = []
    if out_name == "benchmark.yaml":
        # Calibration and evaluation must be the protocol's, unchanged.
        pilot.check_benchmark(data, base)
    if out_name == "datasets.yaml":
        pilot.check_datasets(data)
    if out_name == "titan_ensemble.yaml":
        for ensemble, var in MEMBER_VARS.items():
            members = os.environ.get(var, "").replace(",", " ").split()
            if members and members != data["ensembles"][ensemble]:
                note = (f"{ensemble} = {members} via {var} (protocol: "
                        f"{data['ensembles'][ensemble]})")
                log(f"WARNING: deviation from protocol: {note}")
                notes.append(note)
                data["ensembles"][ensemble] = members
    repo = os.path.dirname(os.path.dirname(os.path.abspath(overlay_dir)))
    header = (f"# Resolved by slurm/pilot_l40s.sh from "
              f"{os.path.relpath(base_path, repo)} + "
              f"configs/killarney/{overlay_name}. Do not edit.\n")
    header += "".join(f"# DEVIATION: {n}\n" for n in notes)
    text = header + yaml.safe_dump(data, sort_keys=False, width=79)
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            old = fh.read()
        if old != text and not os.environ.get("PILOT_FORCE_CONFIG"):
            sys.stderr.writelines(difflib.unified_diff(
                old.splitlines(True), text.splitlines(True), out_path,
                "new resolution"))
            stale.append(out_path)
            continue
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
if stale:
    log(f"ERROR: resolved configs changed since the last run: {stale}. "
        "Runs already trained used the old ones. Set PILOT_FORCE_CONFIG=1 "
        "to overwrite (and delete the affected runs to retrain).")
    sys.exit(3)

# Load the resolved files through the repo's loader; every key must resolve.
bench = config.Config.load(os.path.join(out_dir, "benchmark.yaml"))
ens = config.Config.load(os.path.join(out_dir, "titan_ensemble.yaml"))
config.Config.load(os.path.join(out_dir, "datasets.yaml"))
log(f"calibration.threshold_method = "
    f"{bench.get('calibration.threshold_method')} (secondary: "
    f"{bench.get('calibration.secondary_threshold_methods', [])})")
prereg = bench.get("evaluation.require_preregistration", False)
log(f"evaluation.require_preregistration = {prereg}")
if not prereg:
    log("WARNING: the protocol config does not gate evaluation on "
        "pre-registration.")
backbones, models = pilot.check_ensembles(ens, list(MEMBER_VARS)), set()
for backbone in backbones:
    bspec = spec.backbone_spec(ens, backbone)
    models.add(bspec["model_name"])
    log(f"{backbone} max_seq_len={bspec['max_seq_len']} "
        f"batch={bspec['optim']['batch_size_per_gpu']}x"
        f"{bspec['optim'].get('grad_accum', 1)} lr={bspec['optim']['lr']} "
        f"epochs={bspec['optim']['epochs']} "
        f"ckpt={bspec.get('gradient_checkpointing', False)}")
models |= {bench.get("cohort.dense.query_encoder"),
           bench.get("cohort.dense.article_encoder"),
           bench.get("splits.embedder")}
models.discard("tfidf")

# Jobs are offline: every model must be in HF_HOME before submission.
import huggingface_hub  # pylint: disable=g-import-not-at-top

# Same file patterns as slurm/prefetch_models.sh, so its cache passes.
PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.py",
            "*.tiktoken", "tokenizer*", "vocab*", "merges*", "*.bin"]
for name in sorted(models):
    try:
        huggingface_hub.snapshot_download(
            name, local_files_only=True, allow_patterns=PATTERNS)
        continue
    except Exception:  # pylint: disable=broad-except
        pass
    if mode == "dry":
        log(f"model not cached (would download): {name}")
        continue
    log(f"downloading {name}")
    huggingface_hub.snapshot_download(name, allow_patterns=PATTERNS)
print(" ".join(backbones))
PY
)
export PILOT_BACKBONES
read -ra BACKBONES <<< "$PILOT_BACKBONES"
echo "[pilot] backbones (array index order): $PILOT_BACKBONES"

# run CMD...: executes, or only prints under --dry-run.
run() {
  if [[ "$MODE" == dry ]]; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}
# submit NAME SBATCH_ARGS...: prints the job id (DRY_<NAME> under --dry-run).
submit() {
  local name="$1"; shift
  if [[ "$MODE" == dry ]]; then
    echo "[dry-run] sbatch $*" >&2
    echo "DRY_$name"
  else
    sbatch --parsable "$@" | cut -d';' -f1
  fi
}

# Plan the jobs from what is already on disk (step 2 never changes this).
NEED_OFFLINE=1
if [[ -n "${SKIP_OFFLINE:-}" ]] || [[ -f "$OSB_DATA_ROOT/splits/audit.json" \
      && -f "$OSB_DATA_ROOT/frozen/titan_test/manifest.json" ]]; then
  NEED_OFFLINE=""
fi
TODO=()
for i in "${!BACKBONES[@]}"; do
  [[ -f "$OSB_RUNS_ROOT/${BACKBONES[$i]}/fold0/preds.parquet" ]] || TODO+=("$i")
done
[[ -z "${SKIP_TRAIN:-}" ]] || TODO=()
ARRAY=$(IFS=,; echo "${TODO[*]}")
# sbatch arguments without dependencies (added at submission).
OFFLINE_ARGS=(--export=ALL --time="${OFFLINE_TIME:-04:00:00}"
              slurm/pilot_offline.sbatch)
TRAIN_ARGS_SBATCH=(--export=ALL --array="$ARRAY"
                   --time="${TRAIN_TIME:-24:00:00}" slurm/pilot_train.sbatch)
# shellcheck disable=SC2206
FINAL_ARGS=(--export=ALL --job-name=osb-pilot-final --nodes=1
            --cpus-per-task=16 --mem=128G --time="${FINAL_TIME:-06:00:00}"
            --output=logs/%x_%j.out ${FINAL_SBATCH_ARGS:-}
            slurm/pilot_l40s.sh --finalize)

if [[ "$MODE" == submit ]]; then
  # A second set of jobs would rebuild the same data root and train into
  # the same runs/<backbone>/fold0 concurrently.
  QUEUED=$(squeue -h -u "$USER" \
           -n osb-pilot-offline,osb-pilot-train,osb-pilot-final \
           -o "%i %j %T" || die "squeue failed.")
  if [[ -n "$QUEUED" && -z "${PILOT_ALLOW_QUEUED:-}" ]]; then
    echo "$QUEUED" >&2
    die "pilot jobs are already queued or running (above). Wait for them" \
        "or scancel them, then re-run (PILOT_ALLOW_QUEUED=1 overrides)."
  fi
  # Let Slurm validate every job before anything is queued (e.g. a refused
  # CPU-only final job; see FINAL_SBATCH_ARGS).
  preflight() {
    local out
    out=$(sbatch --test-only "$@" 2>&1) || {
      echo "$out" >&2
      die "sbatch --test-only refused: sbatch $*"
    }
  }
  [[ -z "$NEED_OFFLINE" ]] || preflight "${OFFLINE_ARGS[@]}"
  [[ ${#TODO[@]} -eq 0 ]] || preflight "${TRAIN_ARGS_SBATCH[@]}"
  preflight "${FINAL_ARGS[@]}"
  echo "[pilot] sbatch --test-only accepted every job."
fi

# 2. Network stages on the login node (cached and resumable).
ONLINE_MARK="$OSB_DATA_ROOT/.pilot_online_done"
if [[ -n "${SKIP_ONLINE:-}" || -f "$ONLINE_MARK" ]]; then
  echo "=== skip download,ingest,enrich (SKIP_ONLINE or $ONLINE_MARK)"
else
  echo "=== $(date +%T) download,ingest,enrich (login node, network)"
  if [[ "$MODE" == dry ]]; then
    run osb data "${DATA_CFG[@]}" --stages download,ingest,enrich
  else
    (
      unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
      osb data "${DATA_CFG[@]}" --stages download,ingest,enrich
    ) 2>&1 | tee -a "logs/pilot_online_$(date +%F).log"
    touch "$ONLINE_MARK"
  fi
fi

# 3. Offline stages on one L40S. Dependants are cancelled if a job fails
# (--kill-on-invalid-dep), so a failed pilot leaves nothing pending.
DEP=()
if [[ -z "$NEED_OFFLINE" ]]; then
  echo "=== skip build,split,freeze (frozen titan_test exists)"
  OFFLINE_JOB=""
else
  OFFLINE_JOB=$(submit offline "${OFFLINE_ARGS[@]}")
  DEP=(--dependency=afterok:"$OFFLINE_JOB" --kill-on-invalid-dep=yes)
fi

# 4. One 1-GPU array task per backbone that still lacks predictions.
TRAIN_JOB=""
if [[ ${#TODO[@]} -eq 0 ]]; then
  echo "=== skip training (SKIP_TRAIN or every backbone has preds.parquet)"
else
  for i in "${TODO[@]}"; do
    b="${BACKBONES[$i]}"; steps="MAX_STEPS_$b"
    echo "[pilot] array task $i = $b (max_steps=${!steps:-0})"
  done
  TRAIN_JOB=$(submit train "${DEP[@]}" "${TRAIN_ARGS_SBATCH[@]}")
  DEP=(--dependency=afterok:"$TRAIN_JOB" --kill-on-invalid-dep=yes)
fi

# 5. Calibrate, pre-register, evaluate (CPU job).
FINAL_JOB=$(submit final "${DEP[@]}" "${FINAL_ARGS[@]}")

echo
if [[ "$MODE" == dry ]]; then echo "(dry run: nothing was submitted)"; fi
echo "Jobs:    offline=${OFFLINE_JOB:-skipped} train=${TRAIN_JOB:-skipped}" \
     "final=$FINAL_JOB"
JOBS=$(echo "$OFFLINE_JOB $TRAIN_JOB $FINAL_JOB" | xargs | tr ' ' ',')
echo "Watch:   squeue -u \$USER"
echo "         sacct -j $JOBS -o JobID%20,JobName%20,State,Elapsed,ExitCode"
echo "         tail -f logs/osb-pilot-train_${TRAIN_JOB:-<id>}_*.out"
echo "Outputs: $OSB_RUNS_ROOT/reports/titan_test_T20/ (after final=$FINAL_JOB)"
echo "Then:    git push   (the pre-registration commit is made in the job)"
echo "If a job fails, its dependants are cancelled: check sacct, fix the" \
     "cause, re-run this script (only unfinished steps are resubmitted)."
