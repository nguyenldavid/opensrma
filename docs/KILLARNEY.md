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
  title/abstract into `cache/pubmed.sqlite` (~1–2 M PMIDs).
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
| calibrate_eval | `slurm/calibrate_eval.sbatch` | CPU, 128 GB | stacking + threshold locking per ensemble (with and without RCT features), cascade, **dev** leaderboard on `val`; held-out benchmarks only with `UNBLIND=1` |

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
```

**Pre-registration and unblinding**: development uses the `val`
leaderboard. When thresholds are final:

```bash
osb preregister && git add datasets/manifests && git commit -m "Pre-register"
```

```bash
UNBLIND=1 sbatch slurm/calibrate_eval.sbatch
```

Every held-out evaluation is appended to `$OSB_RUNS_ROOT/unblinding_log.jsonl`.
Jobs pass `--hardware killarney_l40s|killarney_h100` (set by
`submit_all.sh` from `GPU=`); `auto` detection also works.

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
| CUDA OOM | lower `optim.batch_size_per_gpu` and raise `optim.grad_accum` via `TRAIN_ARGS="--override ..."`, or enable `gradient_checkpointing` |
| flash-attn import error | harmless: the model falls back to SDPA attention |
| `tau99 = -inf` in locked thresholds | too few validation positives to certify that sensitivity (needs ≥ 1/α − 1 positives, e.g. ≥ 199 for 99.5%); increase `splits.temporal_ratios.val` |
| Scratch purge warning | copy `$OSB_DATA_ROOT/frozen`, `splits/audit.json`, reports and best checkpoints to `/project` |
