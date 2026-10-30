# CLAUDE.md — Open-SR-Bench / TITAN-Open

Guidance for Claude Code and human contributors working in this repository.

## Who we are and what this is

* We are a **frontier AI research group affiliated with the Vector
  Institute**, working at the top of the field. Hold every change to the
  standard of a top-venue submission (*Nature Medicine*, *Lancet Digital
  Health*, *Annals of Internal Medicine*, *JCE*) and of production research
  code: reproducible, audited, tested, and honestly reported.
* The project has two deliverables:
  1. **Open-SR-Bench** — a post-cutoff, leak-audited benchmark for
     title/abstract screening built fully automatically from open data.
  2. **TITAN-Open** — a decomposed PICOS soft-min cross-encoder ensemble
     with review-conditional conformal calibration (and an optional LLM
     cascade) whose goal is to **decisively beat TITAN-SR** — on TITAN-SR's
     own held-out test split and on Open-SR-Bench — especially at ≥ 99%
     sensitivity where TITAN-SR's specificity collapses (36.2% at 99.5%).
* The protocol is [docs/PROTOCOL.md](docs/PROTOCOL.md) (v1.1). Methods and
  the TITAN-SR comparison are in [docs/METHODS.md](docs/METHODS.md).

## Compute: the Killarney cluster

We have access to **Killarney** (Vector Institute / SciNet, PAICE) through
our Vector affiliation and an AIP RAP (`aip-<pi>` Slurm account).

* Login: `ssh killarney.alliancecan.ca` (CCDB account + SSH key; Vector
  geo-blocking applies). Open OnDemand: ondemand.killarney.vectorinstitute.ai.
* Nodes: 168 × standard (4 × L40S 48 GB, 64 cores, 512 GB) and
  10 × performance (8 × H100 SXM 80 GB, 48 cores, 2 TB).
  Request with `--gpus-per-node=l40s:4` or `--gpus-per-node=h100:8`.
* Storage: `/home` small; `/scratch/$USER` large but purged when inactive;
  `/project/<group>` persistent. Data root defaults to
  `/scratch/$USER/opensrbench` (see `slurm/env.sh`); copy releases you must
  keep to `/project`.
* Treat compute nodes as **offline**: network stages
  (`osb data --stages online`) run on the login node in tmux; models are
  pre-fetched with `slurm/prefetch_models.sh`; jobs set `HF_HUB_OFFLINE=1`.
  `salloc` may not work — use `srun --pty` for interactive sessions.
* Everything is wired through `slurm/env.sh` (account, modules, paths,
  venvs). Full guide: [docs/KILLARNEY.md](docs/KILLARNEY.md).
* Support: ops-help@vectorinstitute.ai (shared directories, quotas).

When asked to "run it on Killarney": edit `slurm/env.sh` (account,
NCBI email/key), then `bash slurm/setup_env.sh --vllm`,
`bash slurm/prefetch_models.sh`, `bash slurm/run_data_online.sh` (login
node, tmux), `WITH_LLM=1 bash slurm/submit_all.sh`. Always submit from the
repo root (Slurm log paths are relative).

## Commands

```bash
python -m pip install -e ".[all]"   # install (venv recommended)
python -m pytest -q                 # all tests (includes ~1 min smoke test)
OSB_SKIP_SMOKE=1 python -m pytest -q  # fast unit tests only
osb smoke                           # synthetic end-to-end pipeline
osb doctor --network                # environment check
make format && make lint            # pyink + isort (Google), pylint
```

## Non-negotiable invariants

1. **Zero leakage.** Never weaken `splits/decontam.py`, the family graph or
   the purge set. The audit must report 0 `pmid` / `title_fp` /
   `abstract_fp` matches between train/val and eval/prospective/titan_test,
   and between train and val. `LeakageError` must stop the pipeline.
2. **Thresholds are locked on validation only** (out-of-fold stacked
   scores), committed to `datasets/manifests/` *before* any held-out
   benchmark is looked at. Never tune anything on `eval`, `prospective` or
   `titan_test`; never choose operating points post hoc.
3. **Benchmark determinism.** Frozen tiers use
   `SHA-256("OPEN-SR-BENCH-V1:" + review_id + ":" + record_id)`; do not
   change salts, ordering or tier logic for a released version — bump the
   version instead. Manifest hashes must reproduce exactly.
4. **Abstention rule.** Any missing/invalid/UNCERTAIN/refused/timeout
   decision counts as *keep* and never as a valid exclusion.
5. **No manual labels.** New data must be labelled automatically from
   published artefacts. Do not hand-edit labels.
6. **Fair baselines.** TITAN-SR is compared through
   `titan_sr_baseline` (faithful re-implementation trained on the same
   splits) and on TITAN-SR's own test split. Report paired family-bootstrap
   ΔJ with Holm correction — never cherry-picked single numbers.
7. **Honest reporting.** Report failures and negative results. Do not
   claim results that were not produced by the pipeline.
8. Never commit data, checkpoints, API keys or emails. Never fabricate
   prompts from papers (e.g. `configs/prompts/cao2025.txt` must be pasted
   verbatim from the publication).

## Architecture map

```text
sources/*.py      SourceAdapter.download() [net] / ingest() [offline] -> schema
pubmed/           E-utilities client + parser + SQLite cache (+ OpenAlex)
harvest/          Europe PMC client + JATS parser (criteria, included studies)
cohort/           query synthesis, BM25/MedCPT/RRF ranking, pool rebuild
pipeline.py       stages: download, ingest, enrich, pools, build, split, freeze
splits/           families (union-find), assign, decontam, tiers
modeling/         inputs (layout + [CRIT] markers), model (soft-min veto),
                  losses (ASL + in-review ranking), sampler, train (DDP), predict
calibration/      stacking (within-review features, cross-fit), thresholds
llm/              prompts/parsers, async client + budget, runner
cascade.py        stage-1 thresholds -> stage-2 verifier
evaluation/       metrics (IPW), bootstrap (family, paired, Holm), report
```

Data flows through two canonical tables (`schema.py`): `reviews` and
`records` (`label_final`, `label_abs`). Every stage reads and writes
parquet under the data root (`datasets/README.md`).

## Conventions (Google Python Style Guide)

* Import modules, not names (`from opensrbench.utils import io`, then
  `io.write_parquet`); `typing`/`collections.abc` excepted.
* Google-style docstrings (Args/Returns/Raises) on public functions; type
  hints everywhere; 4-space indent; 80-column lines; `pyink` + `isort`
  (Google profile) formatting.
* Tests live in `tests/*_test.py` (pytest). Add a test for every bug fix
  and every new adapter/parser. Keep the smoke test passing.
* Configuration lives in YAML under `configs/`; no hard-coded paths, keys
  or hyper-parameters in code. Add new knobs to the YAML with a comment.
* New dataset: subclass `sources.base.SourceAdapter`, decorate with
  `@base.register`, import it in `sources/__init__.py`, add a block to
  `configs/datasets.yaml` and a section to `DATASETS.md`.
* New backbone: add it under `backbones:` in `configs/titan_ensemble.yaml`
  (inherits `defaults`), add it to an ensemble, add it to
  `slurm/prefetch_models.sh` automatically via the ensemble list.
* Network code must go through `utils.http.RateLimitedSession` (NCBI:
  ≤ 10 req/s with key, 3 without) and cache to disk so stages resume.
