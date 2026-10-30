# Local single-GPU pilot (RTX 3080 Ti, 12 GB)

A scaled-down, overnight version of the pipeline that produces a
TITAN-Open vs TITAN-SR comparison on **TITAN-SR's own held-out test split**
on one workstation GPU. It is a **pilot**, not the protocol run: the
protocol run (all open datasets, Open-SR-Bench harvest, 3 backbones x 5
folds) is the Killarney job in [KILLARNEY.md](KILLARNEY.md).

## Environment

The machine has Python 3.8 system-wide, so a Python 3.11 base interpreter
is created with conda inside the repo and the project venv sits on top of
it (both directories are git-ignored):

```bash
conda create -y -p .py311 python=3.11
.py311/bin/python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[all]"
osb doctor            # expect "Ready."; flash_attn/NCBI/HF_HOME warnings ok
```

torch >= 2.6 is required: transformers 5.x silently disables PyTorch on
2.5 (and `peft` then fails to import). The cu124 wheel runs on the
installed 535 driver (CUDA 12.2) via CUDA minor-version compatibility.
`flash_attn` is not installed; ModernBERT falls back to SDPA.

## What differs from the protocol

All differences live in overlay configs under `configs/local/`; every
changed key is marked `LOCAL:` with the original value. The protocol YAMLs
are untouched.

| Overlay | Change | Why |
|---|---|---|
| `datasets_3080ti.yaml` | only `titan_sr` + `chan2024` enabled | TITAN-SR's own corpus; avoids ~days of PubMed fetching without an API key |
| `benchmark_3080ti.yaml` | `augment_incomplete_pools: false` | augmentation needs ~10M PubMed fetches; `titan_test` keeps TITAN-SR's candidate lists, i.e. their evaluation setup |
| `benchmark_3080ti.yaml` | `efetch_batch_size: 500` | fewer requests at <= 3 req/s |
| `titan_ensemble_3080ti.yaml` | `n_folds: 1` | one model per backbone; the 5% holdout drives early stopping |
| `titan_ensemble_3080ti.yaml` | smaller micro-batches, larger `grad_accum`, gradient checkpointing | fits 12 GB with the protocol's effective batch sizes |
| `titan_ensemble_3080ti.yaml` | ensemble `titan_open_local` = BioLinkBERT-large + ModernBERT-large | DeBERTa-v3-large (eager attention) does not fit the night |
| `scripts/local_run.sh` | `--max-steps` budgets (`BASELINE_STEPS`, `OPEN_STEPS`) | fixed overnight compute budget |

Unchanged: the split/decontamination audit, the purge set, threshold
locking on validation only, frozen tiers, and the evaluation code.

## Running

```bash
source .venv/bin/activate
B=(--config configs/local/benchmark_3080ti.yaml
   --datasets-config configs/local/datasets_3080ti.yaml)
osb data "${B[@]}" --stages download,ingest,enrich      # network
osb data "${B[@]}" --stages build,split,freeze          # offline (+ audit)
BASELINE_STEPS=... OPEN_STEPS=... bash scripts/local_run.sh
```

Logs go to `logs/`, checkpoints/predictions/reports to `runs/`, data to
`datasets/` (all git-ignored). The final leaderboard is printed by
`osb evaluate` and written under `runs/reports/titan_test_T*/`.

`scripts/overnight.sh` chains the offline stages and `local_run.sh` after a
running enrich, detached from the terminal. Set `NCBI_EMAIL` before any
network stage.

## Pilot run of 2026-10-30 (what actually happened)

* Data: Chan 2024 (8,608 reviews, abstracts included) + TITAN-SR
  manifests (11,270 reviews); 358,842 PMIDs fetched by EFetch in ~25 min
  with 3 concurrent workers and no API key (~500 records/s). The full
  PubMed FTP baseline (~40 GB) would not have been faster or fit the disk.
* Pipeline fixes found on the way (with tests in `tests/pubmed_test.py`):
  `PubMedStore.fetch_missing` re-queried the cache once per PMID
  (quadratic; never finished on 760k ids); the PMC ID Converter answers 403
  to scripted clients from this host, so PMCID->PMID now falls back to
  ESummary (`db=pmc`); truncated responses (`ChunkedEncodingError`) are
  retried; `network.efetch_workers` hides EFetch latency (the session still
  enforces the rate limit). One transient EFetch 400 needed a rerun
  (resumes from the SQLite cache).
* Split: 16,993 reviews -> 6,345 families; the MedCPT cosine graph is
  dense on this corpus (largest family 8,887 reviews). Families touching a
  TITAN-SR test review are held out, so: train 5,297 reviews (225k
  records), val 282 (12k), titan_test 3,635 (348k: 2,065 TITAN-SR PMC/Track
  B + 1,570 Chan), excluded 7,779. Audit: 0 pmid / title / abstract
  matches train-vs-protected and train-vs-val. No Open-SR-Bench eval set
  (harvest disabled), so `open_sr_bench` is empty.
* Throughput (bf16, 3080 Ti): base models ~1.0 s per 128-sample step;
  BioLinkBERT-large ~0.95 s per 32-sample step; ModernBERT-large
  ~0.97 s per 24-sample step at 512 tokens (12 samples/s at 1024).
  Without gradient checkpointing the large models run out of memory.
* Budgets: `BASELINE_STEPS=2800` (~1.6 passes over train),
  `OPEN_STEPS=4000`.

### Results (titan_test, tier T20, 3,635 reviews)

Operating points were locked on val (pooled conformal) and committed in
`1ee734e` before `osb evaluate` ran. Paired family bootstrap, 5,000
replicates, Holm-corrected. Reports: `runs/reports/titan_test_T20/`
(reference `titan_sr_baseline@tau99`, protocol default) and
`runs/reports/titan_test_T20.ref_tauJ/` (reference `@tauJ`).

| Operating point | TITAN-SR baseline sens / spec | TITAN-Open local sens / spec | Paired ΔJ (Open - SR) |
|---|---|---|---|
| tauJ | 0.806 / 0.765 | 0.829 / 0.765 | **+0.023** (+0.017, +0.040) |
| tau93 | 0.884 / 0.678 | 0.918 / 0.623 | -0.021 (point estimates) |
| tau99 | 0.973 / 0.448 | 0.980 / 0.362 | **-0.078** (-0.086, -0.056) |
| tau99.5 | 0.987 / 0.360 | 0.992 / 0.287 | -0.068 (point estimates) |

**Negative result at the primary target.** TITAN-Open is better only at
the Youden point; at >= 99% sensitivity it is significantly worse than the
TITAN-SR re-implementation. Rows marked "point estimates" are differences
of the two leaderboard rows; only tauJ and tau99 have a paired CI from the
two `evaluate` runs.

Observations, not yet explained:

* `modernbert_large` did not learn: mean training loss 0.85 -> 0.80 over
  4,000 steps (BioLinkBERT-large 0.58 -> 0.40); holdout AUC 0.775 vs
  0.869. It is half of a two-member stack fitted on only 282 val reviews.
* Achieved test sensitivity is below every locked target for both systems
  (e.g. tau93 -> 0.884 / 0.918), i.e. val-locked thresholds under-cover
  on titan_test.
* Sanity check: the baseline at tau99.5 gives 36.0% specificity, close to
  the 36.2% at 99.5% sensitivity TITAN-SR reports, but at 98.7% achieved
  sensitivity, so not a like-for-like match.
* The TITAN-Open members saw ~0.4-0.6 passes over train; the baseline
  members ~1.6.

## Caveats when reading the results

* Both systems are trained with a fixed step budget, not to convergence;
  the TITAN-SR baseline normally runs up to 10 epochs.
* One fold per backbone means the stacker sees two members, not 15.
* Disk on this machine is ~20 GB free; keep only `best/` checkpoints.
