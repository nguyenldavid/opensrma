# Open-SR-Bench + TITAN-Open

A leak-audited benchmark and a state-of-the-art screening model for
**title/abstract screening in systematic reviews**, built to be run end to
end on the **Killarney** cluster (Vector Institute / PAICE).

* **Open-SR-Bench** — systematic reviews published *after* frontier-model
  knowledge cut-offs (2025-01-01 → 2026-10-31), harvested automatically from
  Europe PMC, with reconstructed date-bounded search pools, family-level
  splits, a triple-fingerprint zero-leakage audit and frozen T10/T20/T40
  evaluation tiers. **No manual labelling anywhere**: labels come from each
  review's own included-studies list.
* **TITAN-Open** — a decomposed PICOS soft-min cross-encoder ensemble
  (ModernBERT-large, BioLinkBERT-large, DeBERTa-v3-large, optional
  Qwen3-Reranker-4B LoRA) trained on **every open screening dataset**, with
  review-conditional conformal calibration and an optional two-stage LLM
  cascade. Designed to beat [TITAN-SR](https://github.com/InvestmentMDideas/TITAN-SR)
  head-to-head, on TITAN-SR's own test split *and* on the new benchmark.

| Document | What it covers |
|---|---|
| [docs/KILLARNEY.md](docs/KILLARNEY.md) | **Start here on the cluster**: setup, data, training, evaluation, troubleshooting |
| [DATASETS.md](DATASETS.md) | Every training/evaluation dataset: provenance, licence, labels, counts, caveats |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Stage-by-stage reference, data schema, outputs, resumability |
| [docs/METHODS.md](docs/METHODS.md) | Model, losses, calibration, statistics; why it beats TITAN-SR; ablations |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | Study protocol v1.1 (TRIPOD+AI / STARD-AI) with all amendments to v1.0 |
| [docs/LOCAL_GPU.md](docs/LOCAL_GPU.md) | **Single workstation GPU** (RTX 3080 Ti 12 GB and up): profiles, one-command run |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Current status and **what still needs to be done** (prioritised) |
| [CLAUDE.md](CLAUDE.md) | Team context, invariants and conventions for contributors and agents |

## Quick start

### 1. Laptop / any Linux box (CPU is fine)

```bash
git clone https://github.com/nguyenldavid/opensrma && cd opensrma
```

```bash
python -m pip install -e ".[all]"
```

```bash
osb doctor
```

```bash
osb smoke            # synthetic end-to-end run of the offline pipeline (~1 min)
```

```bash
python -m pytest -q  # unit tests + smoke test
```

### 2. One workstation GPU (RTX 3080 Ti or better)

```bash
export NCBI_EMAIL=you@lab.org NCBI_API_KEY=xxxx OSB_HARDWARE=rtx3080ti
bash scripts/run_local.sh      # data -> RCT classifier -> train -> calibrate -> dev report
```

See [docs/LOCAL_GPU.md](docs/LOCAL_GPU.md).

### 3. Killarney (full run)

```bash
ssh killarney.alliancecan.ca
git clone https://github.com/nguyenldavid/opensrma && cd opensrma
```

```bash
$EDITOR slurm/env.sh                       # set OSB_ACCOUNT=aip-<pi>, NCBI_EMAIL, NCBI_API_KEY
```

```bash
bash slurm/setup_env.sh --vllm             # one-time venvs (login node)
```

```bash
bash slurm/prefetch_models.sh              # cache HF models for offline GPU jobs
```

```bash
tmux new -s osb-data 'bash slurm/run_data_online.sh'   # network stages, resumable
```

```bash
WITH_LLM=1 bash slurm/submit_all.sh        # build -> train arrays -> verifier -> calibrate -> evaluate
```

Results land in `$OSB_RUNS_ROOT/reports/<benchmark>_T20/leaderboard.md`.
Full details, resource choices (L40S vs H100) and troubleshooting:
[docs/KILLARNEY.md](docs/KILLARNEY.md).

## The `osb` command

| Command | What it does | Where |
|---|---|---|
| `osb doctor [--network]` | Checks Python, deps, GPUs, keys, paths, reachability | anywhere |
| `osb data --stages online` | `download, ingest, enrich, pools` (network, resumable) | login node |
| `osb data --stages offline` | `build, split, freeze` (BM25 + MedCPT, audit, tiers) | 1 GPU job |
| `osb plan-runs --ensemble E` | Lists (backbone, fold) runs = Slurm array indices | anywhere |
| `osb train --backbone B --fold k` | DDP training of one run (use under `torchrun`) | GPU node |
| `osb predict --backbone B --fold k` | Whole-cohort scoring of val + benchmarks | GPU node |
| `osb calibrate --ensemble E` | Stacking, conformal threshold locking, decision files | CPU |
| `osb llm-sweep` | Budgeted async LLM screening (Conditions A/B, Cao) | CPU / GPU |
| `osb cascade --ensemble E` | Two-stage encoder → LLM verifier cascade | CPU |
| `osb evaluate --benchmark NAME` | IPW equal-review metrics, 5,000 family-bootstrap CIs, Holm | CPU |
| `osb rct corpus\|train\|score\|systems` | Cochrane-recipe RCT classifier (leak-free), RCT filter baseline | CPU |
| `osb audit` | Re-runs every leakage guard on the current splits | CPU |
| `osb preregister` | Copies tier manifests + locked thresholds to `datasets/manifests/` | CPU |
| `osb evaluate --benchmark val` | Development leaderboard (open) | CPU |
| `osb evaluate --benchmark internal_test --unblind` | Held-out leaderboard (blinded; every call logged) | CPU |
| `osb smoke` | Synthetic end-to-end run | anywhere |

Training and prediction take `--hardware {auto,rtx3080ti,gpu_24gb,killarney_l40s,killarney_h100,cpu}`.

Every command takes `--config` (default `configs/benchmark_v1.yaml`) and
`--help`. Individual stages can be run with e.g.
`osb data --stages enrich --only clef_tar,synergy`.

## Repository layout

```text
configs/
  benchmark_v1.yaml        # dates, harvest, cohort, splits, calibration, eval
  datasets.yaml            # which open datasets to ingest (+ priorities)
  titan_ensemble.yaml      # backbones, training recipes, ensembles, baseline
  llm_models.yaml          # endpoints, models, prices, budget
  hardware/                # rtx3080ti, gpu_24gb, killarney_l40s, killarney_h100, cpu
  benchmark_local.yaml     # workstation data profile (extends benchmark_v1.yaml)
  prompts/                 # Condition A / B / Cao (paste verbatim) templates
src/opensrbench/
  sources/                 # one adapter per open dataset + Europe PMC harvest
  pubmed/                  # E-utilities client, XML parser, SQLite cache, OpenAlex
  harvest/                 # Europe PMC client + JATS parser
  cohort/                  # PICO query synthesis, BM25/MedCPT/RRF, pool rebuild
  splits/                  # family graph, assignment, decontamination, guards, tiers
  rct/                     # Cochrane-recipe RCT classifier + leak-free corpus
  text/                    # criteria decomposition, review-type classifier
  modeling/                # inputs, model, losses, samplers, DDP train, predict
  calibration/             # stacking + conformal / equal-review thresholds
  llm/                     # prompts/parsers, async client + budget, runner
  evaluation/              # IPW metrics, bootstrap, agreement, ranking, report
  pipeline.py cascade.py cli.py smoke.py doctor.py
slurm/                     # env.sh, setup, prefetch, data, train array, eval, vLLM
scripts/run_local.sh       # single-GPU end-to-end runner
tests/                     # *_test.py (pytest), incl. end-to-end smoke test
datasets/                  # default data root (generated content is git-ignored)
docs/                      # Killarney guide, pipeline, methods, protocol
```

## Splits and held-out sets (no leakage)

| Split | What | Used for |
|---|---|---|
| `train` | families of legacy + post-cutoff reviews | fitting (5 family folds per backbone; held-out fold = early stopping) |
| `val` | 5% legacy + 10% post-cutoff families | calibration / threshold locking; open development reports |
| `test` | **15% of legacy families (review-level holdout, all sources)** | `internal_test` benchmark (blinded) |
| `eval` | 15% of post-cutoff families | `open_sr_bench` (≤ 10 families per review type) + extended (blinded) |
| `titan_test` | TITAN-SR's own test split + external holdout | head-to-head on their benchmark (blinded) |
| `prospective` | published after the freeze | strictly prospective (blinded) |

Families (overlapping included studies or near-duplicate objectives) are
assigned atomically; train/val records sharing any PMID, title or abstract
fingerprint with a protected split are purged, and the guards re-check this
in `split`, before every training run and in `osb audit`.

## Status and honest caveats

* The pipeline, model and statistics are implemented and tested (unit tests
  plus a synthetic end-to-end run); data stages were validated live against
  PubMed / Europe PMC on small slices. **No benchmark results are claimed
  yet**. What remains is tracked in [docs/ROADMAP.md](docs/ROADMAP.md).
* Tier-2 pool reconstruction uses a topic-local candidate index (PubMed
  Boolean + broad recall query, re-ranked with BM25 + MedCPT) rather than a
  full local PubMed index; each pool's recall of the known included studies
  is recorded (`pool_recall`) and a floor is applied for benchmark
  eligibility. See [docs/PROTOCOL.md](docs/PROTOCOL.md) (A15).
* The Cao et al. (2025) prompt is not reproduced; paste it verbatim into
  `configs/prompts/cao2025.txt` (the runner refuses to run it otherwise).
* Third-party dataset licences apply (see [DATASETS.md](DATASETS.md)).

## Style

Code follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
(module imports, Google docstrings, type hints, 80 columns, `*_test.py`).
Format with `make format` (pyink + isort, Google profiles); lint with
`make lint`.

## Licence

Code: MIT. Derived manifests: CC BY 4.0. Source datasets keep their own
licences (see [DATASETS.md](DATASETS.md)).
