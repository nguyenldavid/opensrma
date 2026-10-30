# Running Open-SR-Bench + TITAN-Open on Killarney

Killarney is the Vector Institute / SciNet PAICE cluster
(`killarney.alliancecan.ca`). Access needs a CCDB account with an SSH key
and membership of an **AIP RAP** (Slurm account `aip-<pi>`).

| Tier | Nodes | GPUs / node | CPU / RAM per node | Slurm request |
|---|---:|---|---|---|
| Standard | 168 | 4 × NVIDIA L40S 48 GB | 64 cores / 512 GB | `--gpus-per-node=l40s:4` |
| Performance | 10 | 8 × NVIDIA H100 SXM 80 GB | 48 cores / 2 TB | `--gpus-per-node=h100:8` |

Storage: `/home` (small), `/scratch/$USER` (large, purged when inactive),
`/project/<group>` (persistent, backed up). Assume compute nodes have **no
internet**: network work runs on the login node, models are cached in
`HF_HOME` beforehand and jobs run with `HF_HUB_OFFLINE=1`.

> Verify site details that change over time (module names, walltime caps,
> account string) with `module spider`, `sacctmgr show assoc user=$USER`
> and the Alliance wiki page for Killarney.

---

## Quick start: L40S pilot (TITAN-SR + Chan)

TITAN-Open (`titan_open`: ModernBERT-large, BioLinkBERT-large,
DeBERTa-v3-large; stacking) vs the TITAN-SR re-implementation
(`titan_sr_baseline`; mean of sigmoids) on **TITAN-SR's own held-out test
split**, with the local pilot's data scope ([LOCAL_RUN.md](LOCAL_RUN.md):
only `titan_sr` + `chan2024`, no pool augmentation, no harvest) but the
**protocol model settings** (ModernBERT at 1024 tokens, protocol batch
sizes, LR, epochs and early stopping, no gradient checkpointing) and the
protocol's calibration (`threshold_method` and secondary methods) and
pre-registration gate, inherited from `configs/benchmark_v1.yaml`. It is a
pilot, not the protocol run.

```bash
ssh killarney.alliancecan.ca
cd /project/<group>   # or $HOME
git clone https://github.com/nguyenldavid/opensrma && cd opensrma
git checkout claude/practical-lovelace-7oe49s
```

Edit the site settings at the top of `slurm/env.sh` (`OSB_ACCOUNT=aip-<pi>`,
`NCBI_EMAIL`, ideally `NCBI_API_KEY`; table in §0), and set a git identity
(the final job makes the pre-registration commit):

```bash
git config --global user.name "Your Name" && git config --global user.email you@example.org
```

```bash
bash slurm/setup_env.sh
```

```bash
bash slurm/prefetch_models.sh titan_open titan_sr_baseline
```

```bash
sbatch slurm/smoke.sbatch        # wait for it to pass: logs/osb-smoke_<id>.out
```

```bash
tmux new -s osb-pilot 'bash slurm/pilot_l40s.sh 2>&1 | tee -a logs/pilot_l40s.log'
```

Put any knob from the table below **inside** the tmux command string, e.g.

```bash
tmux new -s osb-pilot 'OPEN_MEMBERS="biolinkbert_large deberta_v3_large" bash slurm/pilot_l40s.sh 2>&1 | tee -a logs/pilot_l40s.log'
```

Variables `export`ed in your shell are **not** seen by the script once a
tmux server is already running (a new session inherits the server's
environment, not your shell's), so the protocol members would train
silently. The script logs every deviation it applies (`WARNING: deviation
from protocol`); check for it at the top of the log.

`bash slurm/pilot_l40s.sh --dry-run` prints the plan without submitting.
`setup_env.sh` now fails if torch < 2.6 or `import peft` fails
(transformers 5.x silently disables older torch). `prefetch_models.sh` also
fetches `openai/gpt-oss-20b` (~13 GB, not used by the pilot); the pilot
script downloads any other missing model itself.

What `pilot_l40s.sh` does (login node; every step is skipped when already
done, so re-running it is safe **once no pilot job is queued or running**;
it refuses otherwise, `PILOT_ALLOW_QUEUED=1` overrides). Before queuing
anything it validates all jobs with `sbatch --test-only`, so a refused job
(e.g. a CPU-only final job) fails before anything is submitted.

1. Resolves the overlays `configs/killarney/{benchmark,datasets,titan_ensemble}_pilot.yaml`
   (`extends:` the protocol YAMLs) into `$OSB_PILOT_ROOT/configs/` and
   checks them (`src/opensrbench/pilot.py`: calibration and evaluation
   unchanged, only `titan_sr` + `chan2024` enabled, one fold per member).
   Every changed key is marked `KILLARNEY PILOT:` with the protocol value.
   If a resolved file would change on a re-run (protocol edited
   mid-pilot), it stops and shows the diff; `PILOT_FORCE_CONFIG=1`
   overwrites.
2. Runs `download,ingest,enrich` on the login node (needs internet; ~30–60
   min with an API key, mostly PubMed EFetch for ~360k PMIDs; the pilot
   uses the local pilot's EFetch throughput settings, 500 PMIDs × 3
   concurrent requests; with the protocol's 200 × 1 expect ~2–3 h).
3. Submits `slurm/pilot_offline.sbatch` (1 × L40S): `build,split,freeze`
   with MedCPT and the zero-leakage audit (2 min locally).
4. Submits `slurm/pilot_train.sbatch` as an array, **one L40S per
   backbone** (world size 1, no DDP), `afterok` on 3: train, then predict
   `val,titan_test`. Skips training if `summary.json` exists and prediction
   if `preds.parquet` exists; only unfinished backbones get a task.
5. Submits itself with `--finalize` (CPU job), `afterok` on 4: `osb
   calibrate` for both ensembles, `RUN_TAG=killarney_pilot bash
   slurm/preregister.sh titan_open titan_sr_baseline` (local commit of the
   locked thresholds + frozen manifests into `datasets/manifests/`), then
   `osb evaluate --benchmark titan_test` (primary, `equal_review_lcb`) and
   the pre-specified secondary `pooled_conformal` set (A19) into
   `reports/titan_test_T20_secondary/`. A resubmitted final job does not
   re-calibrate an ensemble that is already pre-registered.

It prints the job ids and `squeue` / `sacct` / `tail -f` commands to watch
them. Logs: `logs/osb-pilot-{offline,train,final}_*.out`.

**If a job fails**, its dependants are cancelled automatically
(`--kill-on-invalid-dep=yes`). Check `sacct -j <ids>`, fix the cause, and
re-run `pilot_l40s.sh`: it resubmits only the unfinished steps (backbones
without `preds.parquet`).

| Variable | Default | Effect |
|---|---|---|
| `OPEN_MEMBERS` / `SR_MEMBERS` | protocol members | replace an ensemble's backbones, e.g. `OPEN_MEMBERS="biomedbert_large biolinkbert_large deberta_v3_large"` (recorded as `DEVIATION` in the resolved config) |
| `MAX_STEPS_<backbone>` | `0` (protocol epochs) | optimizer-step cap, e.g. `MAX_STEPS_deberta_v3_large=20000` |
| `TRAIN_ARGS_<backbone>` / `TRAIN_ARGS` | empty | extra `osb train` flags, e.g. `TRAIN_ARGS_modernbert_large="--override gradient_checkpointing=true"` on CUDA OOM |
| `TRAIN_TIME` / `OFFLINE_TIME` / `FINAL_TIME` | `24:00:00` / `04:00:00` / `06:00:00` | walltimes |
| `FINAL_SBATCH_ARGS` | empty | e.g. `--gpus-per-node=l40s:1` if CPU-only jobs are refused |
| `SKIP_ONLINE` / `SKIP_OFFLINE` / `SKIP_TRAIN` | unset | `=1` skips a finished step |
| `OSB_PILOT_ROOT` | `$OSB_STORAGE/killarney_pilot` | data, runs, resolved configs |

> **Walltime warning.** There is no mid-run checkpoint resume: a training
> task killed at its walltime restarts from scratch on re-submission. If a
> member may not finish in 24 h, cap it with `MAX_STEPS_<backbone>` (report
> the cap) or raise `TRAIN_TIME` within the partition's limit.

**Expected runtime (rough estimate, not a measurement).** Scaled from the
local 3080 Ti pilot ([LOCAL_RUN.md](LOCAL_RUN.md): BioLinkBERT-large ~34
samples/s, ModernBERT-large ~12 samples/s at 1024 tokens, base models ~127
samples/s, all with gradient checkpointing) assuming an L40S without
checkpointing is ~2–3× faster: TITAN-SR baseline members ~1–2 h each (≤ 10
epochs, early stopping), BioLinkBERT-large ~3–4 h, DeBERTa-v3-large (eager
attention, no measurement yet) ~4–7 h, ModernBERT-large at 1024 tokens ~6–10
h, each including prediction of ~360k val + titan_test records. DeBERTa-v3-
large and ModernBERT-large are the slowest members. The array tasks run in
parallel, so after queue waits expect **roughly 8–12 h end to end**
(~15–25 L40S GPU-hours). Measure with `MAX_STEPS_<backbone>=50` first if the
allocation is tight.

Outputs, under `$OSB_PILOT_ROOT` (default
`/scratch/$USER/opensrbench/killarney_pilot`):

```text
configs/{benchmark,datasets,titan_ensemble}.yaml   # resolved configs used by every job
datasets/splits/audit.json                         # zero-leakage audit
runs/<backbone>/fold0/{summary.json,preds.parquet,best/}
runs/calibration/<ensemble>/locked_operating_points.json
runs/reports/titan_test_T20/leaderboard.md         # the comparison
runs/reports/titan_test_T20_secondary/             # pooled_conformal (A19 secondary)
```

and the pre-registration commit in the clone (`datasets/manifests/*.killarney_pilot*.json`).
**Push the pre-registration commit from the login node** (`git push`):
compute nodes are offline, so the job only commits locally.

> **Full protocol (`submit_all.sh`) is not ready yet.** Multi-GPU (DDP)
> training has a fix pending verification on GPUs, and the full-scale data
> stages need several days of network traffic while `data_offline.sbatch`'s
> 24 h build limit is not yet addressed. Use the pilot above until then.
>
> With DDP, the global batch is `batch_size_per_gpu × grad_accum` on any
> GPU count (`osb train` logs `global batch ... = W GPU(s) x micro-batch m
> x grad_accum a`). A recipe that cannot be split over the GPUs without
> changing the number of review groups per step stops at startup. With the
> current `configs/titan_ensemble.yaml`, `train_array.sbatch`'s 4 × L40S
> fails for every `titan_open` member: modernbert_large (24 × 1) needs one
> GPU per task; biolinkbert_large, biomedbert_large (32 × 1) and
> deberta_v3_large (16 × 2) work on 2 GPUs. Choose GPUs per task or raise
> `grad_accum` (a recipe change) before submitting.

---

## 0. One-time setup (login node, ~20 min)

```bash
ssh killarney.alliancecan.ca
cd /project/<group>   # or $HOME
git clone https://github.com/nguyenldavid/opensrma && cd opensrma
```

Edit the **site settings** block at the top of `slurm/env.sh`:

| Variable | Example | Purpose |
|---|---|---|
| `OSB_ACCOUNT` | `aip-smithlab` | Slurm account (exported as `SBATCH_ACCOUNT`) |
| `OSB_STORAGE` | `/scratch/$USER/opensrbench` | data root, runs, HF cache |
| `OSB_MODULES` | `StdEnv/2023 python/3.11 cuda/12.6 arrow` | modules loaded by every job |
| `NCBI_EMAIL` | `you@lab.org` | required by NCBI E-utilities |
| `NCBI_API_KEY` | (from your NCBI account) | 10 instead of 3 requests/s |

Then:

```bash
bash slurm/setup_env.sh --vllm
```

```bash
bash slurm/prefetch_models.sh
```

```bash
sbatch slurm/smoke.sbatch
```

`setup_env.sh` builds `$OSB_VENV` (prefers the Alliance wheelhouse for
torch/numpy/pandas/pyarrow, then PyPI), installs the package with all extras
and FlashAttention-2 if available, and with `--vllm` a separate vLLM venv
for the verifier. `prefetch_models.sh` downloads every backbone of
`titan_open`, `titan_open_xl` and `titan_sr_baseline`, the MedCPT encoders and
`openai/gpt-oss-20b`. `smoke.sbatch` runs `osb doctor`, the synthetic
end-to-end pipeline and the test suite on an L40S — if it passes, the
account and environment are good.

## 1. Network data stages (login node, resumable)

```bash
tmux new -s osb-data 'bash slurm/run_data_online.sh'
```

This runs `download → ingest → enrich → pools`:

* **download**: git-clones / downloads every source in
  `configs/datasets.yaml` (CLEF TAR, Chan 2024 [466 MB, checksum-verified],
  SYNERGY, Cohen, SWIFT, Wallace, SIGIR 2017, TITAN-SR) and harvests
  Europe PMC (search → JATS full text → reference resolution).
* **ingest**: parses everything into canonical parquet (offline, minutes).
* **enrich**: resolves review PMIDs and fetches every citation's
  title/abstract into `cache/pubmed.sqlite` (~1–2 M PMIDs). TITAN-SR Track
  B reviews now carry the PMID/DOI encoded in their PDF filename: if
  `ingest` + `enrich` ran before that change, re-run both (about 2,000
  extra review PMIDs, a small EFetch) and then the offline stages, or the
  ~2,100 Track B reviews keep empty titles/objectives (they then get no
  cosine edges in the family graph, protocol A21).
* **pools**: Tier-1/Tier-2 date-bounded retrieval for harvested reviews and
  pool augmentation for TITAN-SR/SIGIR reviews.

Runtime is dominated by polite NCBI traffic: roughly a day with an API key.
Everything is cached; if the session dies just run the same command again.
Detach with `Ctrl-b d`, re-attach with `tmux attach -t osb-data`. To run a
subset: `bash slurm/run_data_online.sh --only clef_tar,chan2024`.

> If your Killarney compute nodes *do* have outbound internet (check with
> `srun --pty -t 10 curl -sI https://eutils.ncbi.nlm.nih.gov`), you may run
> the same command inside a CPU job instead.

## 2. Everything else: one command

```bash
WITH_LLM=1 bash slurm/submit_all.sh
```

| Job | Script | Resources (default) | What it does |
|---|---|---|---|
| data_offline | `slurm/data_offline.sbatch` | 1 × L40S, 16 CPU, 192 GB, 24 h | build (BM25 + MedCPT pools), split + **zero-leakage audit** (job fails on any leak), freeze tiers |
| train (array) | `slurm/train_array.sbatch` | 4 × L40S per task | one task per (backbone, fold): DDP training then whole-cohort prediction |
| llm_verifier | `slurm/llm_verifier.sbatch` | 1 × H100 | starts vLLM (gpt-oss-20b) on the node and runs Conditions A/B on T20 |
| calibrate_eval | `slurm/calibrate_eval.sbatch` | CPU, 128 GB | stacking + threshold locking per ensemble, cascade, leaderboards |

Variants:

```bash
GPU=h100 bash slurm/submit_all.sh                     # train on H100 nodes
```

```bash
SKIP_DATA=1 ENSEMBLES="titan_open_xl titan_sr_baseline" GPU=h100 bash slurm/submit_all.sh
```

Array sizes come from `osb plan-runs`: `titan_open` = 3 backbones × 5 folds
= 15 tasks, `titan_open_xl` = 20, `titan_sr_baseline` = 2. Re-submitting is
safe: a task whose `summary.json` exists skips training and only predicts.

### Manual control

```bash
osb plan-runs --ensemble titan_open
```

```bash
ENSEMBLE=titan_open sbatch --array=0-14 slurm/train_array.sbatch
```

```bash
ENSEMBLE=titan_open sbatch --array=3 --gpus-per-node=h100:8 --cpus-per-task=48 --mem=1000G slurm/train_array.sbatch
```

```bash
TRAIN_ARGS="--override optim.lr=2e-5 --override loss.rank_type=pairwise" ENSEMBLE=titan_open sbatch --array=0 slurm/train_array.sbatch
```

Interactive debugging on a GPU node:

```bash
srun --gpus-per-node=l40s:1 --cpus-per-task=8 --mem=64G -t 1:00:00 --pty bash
source slurm/env.sh
torchrun --standalone --nproc_per_node=1 -m opensrbench.cli train --backbone biolinkbert_large --fold 0 --max-steps 50 --limit-reviews 50
```

## 3. Outputs

```text
$OSB_DATA_ROOT/splits/audit.json                     # leakage audit + split sizes
$OSB_DATA_ROOT/frozen/<benchmark>/manifest.json      # tier hashes (pre-registration)
$OSB_RUNS_ROOT/<backbone>/fold<k>/best/              # checkpoint (safetensors)
$OSB_RUNS_ROOT/<backbone>/fold<k>/{train_log.jsonl,summary.json,preds.parquet}
$OSB_RUNS_ROOT/calibration/<ensemble>/locked_operating_points.json
$OSB_RUNS_ROOT/systems/*.parquet                     # decision files per system
$OSB_RUNS_ROOT/reports/<benchmark>_T20/leaderboard.md   # Table 1
$OSB_RUNS_ROOT/reports/<benchmark>_T20/{by_review_type,ranking,pareto}.csv
$OSB_RUNS_ROOT/systems_secondary/*.parquet           # A19 secondary decisions
$OSB_RUNS_ROOT/reports/<benchmark>_T20_secondary/    # their leaderboard
```

**Pre-registration step** (protocol §7, A20): `calibrate_eval.sbatch`
runs `bash slurm/preregister.sh <ensemble>...` after `osb calibrate` and
before any held-out benchmark is evaluated. It copies every
`locked_operating_points.json` and frozen `manifest.json` into
`datasets/manifests/` as `*.<RUN_TAG>.json` (default `killarney`) and
commits exactly those files locally; `osb evaluate` refuses to run on
thresholds that are not committed byte-identical in git HEAD.

* Set `git config --global user.name` / `user.email` first (compute nodes
  read `~/.gitconfig` through `/home`), or the job fails.
* **Push the commit from the login node** (`git push`) afterwards.
* Resubmitting the job after a failure is safe: already pre-registered
  ensembles are not re-calibrated.
* `osb evaluate --skip-preregistration-check` is for development only;
  results produced with it are not protocol results.
* The secondary `pooled_conformal` set (A19) is evaluated into
  `reports/<benchmark>_T20_secondary/` (`osb evaluate --tag secondary`).

## 4. Throughput and budget planning

Rough planning estimates (not measurements; bf16, per GPU, 512 tokens unless
noted) — measure on a 50-step run with `--max-steps 50` before budgeting:

| Backbone | Params | L40S seq/s (train) | H100 seq/s (train) | Notes |
|---|---:|---:|---:|---|
| BioLinkBERT-large / BiomedBERT-large | 340 M | ~180 | ~550 | 512 tokens max |
| DeBERTa-v3-large | 435 M | ~90 | ~280 | slowest encoder; no flash-attn |
| ModernBERT-large (1024 tok) | 395 M | ~110 | ~380 | FlashAttention-2 recommended |
| Qwen3-Reranker-4B LoRA (1536 tok) | 4 B | — | ~45 | H100 only; gradient checkpointing |

With the default sampler an epoch is ≈ 7 × (capped) positives ≈ 1–1.5 M
pairs. `titan_open` therefore needs roughly 4–8 L40S-node-hours per
(backbone, fold) — ~1 day of wall time for the whole array when tasks run in
parallel. Tune `sampler.max_pos_per_review`, `sampler.negatives_per_positive`
and `optim.epochs` in `configs/titan_ensemble.yaml` to fit your allocation.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `sbatch: error: invalid account` | set `OSB_ACCOUNT` in `slurm/env.sh`; check `sacctmgr show assoc user=$USER` |
| `OSError: ... couldn't connect to huggingface.co` in a job | run `bash slurm/prefetch_models.sh` on the login node (jobs are offline) |
| `module: command not found` / wrong module names | adjust `OSB_MODULES` (`module spider python`, `module spider cuda`) |
| `FileNotFoundError: ... run the upstream stage` | a previous stage has not run; see the stage order in docs/PIPELINE.md |
| `LeakageError: FATAL LEAKAGE DETECTED` | do not bypass. Inspect `splits/audit.json`; usually a new source lacks titles/abstracts or a fingerprint column |
| NCBI HTTP 429 | lower `network.ncbi_requests_per_second`; make sure `NCBI_API_KEY` is set |
| OpenAlex quota exhausted | set `OPENALEX_API_KEY` and re-run `enrich` (only SYNERGY records without PMID need it) |
| CUDA OOM | lower `optim.batch_size_per_gpu` and raise `optim.grad_accum` via `TRAIN_ARGS="--override ..."` keeping their product (the recipe's global batch), or enable `gradient_checkpointing` |
| `ValueError: global batch ... review groups` at train start | the recipe does not split over this many GPUs; use fewer GPUs per task (see the DDP note in the quick start) |
| `preregister.sh: ... Refusing to re-lock` | thresholds changed after they were committed. Find out why; set `ALLOW_RELOCK=1` only if no held-out benchmark has been evaluated with the committed version, and record it in the protocol |
| `PreregistrationError` from `osb evaluate` | run `bash slurm/preregister.sh <ensembles>` (from the repo root) and commit; never evaluate held-out data first |
| flash-attn import error | harmless: the model falls back to SDPA attention |
| `tau99 = -inf` in locked thresholds | too few validation positives to certify that sensitivity (needs ≥ 1/α − 1 positives, e.g. ≥ 199 for 99.5%); increase `splits.temporal_ratios.val` |
| Scratch purge warning | copy `$OSB_DATA_ROOT/frozen`, `splits/audit.json`, reports and best checkpoints to `/project` |
