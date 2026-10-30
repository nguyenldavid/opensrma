# Pipeline reference

```text
  [network, login node]                         [offline, Slurm]
  download -> ingest -> enrich -> pools   ==>   build -> split -> freeze
                                                    |
          train (array) -> predict -> calibrate -> [llm-sweep] -> cascade -> evaluate
```

All stages read `configs/benchmark_v1.yaml` (`--config`) and write under the
data root (`paths.data_root` / `OSB_DATA_ROOT`) or runs root
(`paths.runs_root` / `OSB_RUNS_ROOT`). Every stage is **idempotent and
resumable**: downloads, PubMed records, Europe PMC full texts, reference
maps and candidate pools are cached and skipped when present.

## Data stages (`osb data --stages ...`)

| Stage | Net | Reads | Writes | Notes |
|---|:-:|---|---|---|
| `download` | ✓ | `configs/datasets.yaml` | `raw/<source>/`, `cache/europepmc/` | git clones, checksummed archives; Europe PMC search + JATS + reference resolution |
| `ingest` | | `raw/` | `interim/<source>/{reviews,records}.parquet` | canonical schema; text may be missing for PMID records |
| `enrich` | ✓ | `interim/` | `cache/pubmed.sqlite`, `interim/enrich/*` | review PMIDs (PMC id converter, DOI, Cochrane CD); all citation texts; DOI→PMID; OpenAlex fallback |
| `pools` | ✓ | `interim/`, PubMed cache | `interim/pools/<review>.json` | Tier 1 (verbatim strategy) / Tier 2 (strict + broad PICO queries) under t_r; texts of all candidates cached |
| `rct_corpus` | ✓ | — | `rct/corpus_pmids.parquet`, PubMed cache | MEDLINE publication-type sample for the RCT classifier (decontaminated later, at `osb rct train`) |
| `build` | | all of the above | `processed/{reviews,records}.parquet`, `build_summary.json` | review de-dup by key/priority; BM25 + MedCPT + RRF pool assembly; text attach; quality filters; fingerprints; criteria lists; eligibility |
| `split` | | `processed/` | `splits/{splits,records}.parquet`, `audit.json` | family graph; atomic assignment (train/val/**test**/eval/prospective/titan_test); purge; guards; **LeakageError on any leak** |
| `freeze` | | `splits/` | `frozen/<bench>/{tiers,review_totals}.parquet`, `manifest.json` | T10/T20/T40, stability subset, SHA-256 per tier |

Groups: `online` = download,ingest,enrich,pools,rct_corpus; `offline` = build,split,freeze;
`all`. Restrict sources with `--only clef_tar,chan2024`. Disable MedCPT with
`--device none` (BM25 only) or set `cohort.dense.enabled: false`.

## Canonical schema (`src/opensrbench/schema.py`)

**reviews**: `review_id` (`<source>:<id>`), `source`, `review_key` (dedupe
key such as `cochrane:CD012255`, `pmc:PMC123`, `doi:...`), `title`,
`objective`, `selection_criteria`, `criteria_list` (JSON list, added in
build), `review_type` (10 strata), `pub_date`, `search_date`,
`pool_complete`, `pool_tier` (`provided`/`tier1_exact`/`tier2_hybrid`/
`augmented`), `pool_recall`, `has_abs_labels`, `domain`, `doi`, `pmid`,
`pmcid`, `extra` (JSON), and build statistics `n_pos`, `n_neg`, `n_total`,
`n_pos_with_abstract`, `eligible_eval`.

**records**: `review_id`, `record_id` (`pmid:<n>`, `doi:<doi>`,
`openalex:<W>` or `<source>:<hash>`), `pmid`, `title`, `abstract`,
`pub_date`, `label_final`, `label_abs`, `retrieval_rank`,
`retrieval_source` (`provided`/`cited`/`included`/`ft_excluded`/`tier1`/
`boolean`/`hybrid`/`augmented`), `publication_types`, `title_fp`,
`abstract_fp`.

**splits**: `review_id`, `family_id`, `split` (`train`/`val`/`eval`/
`prospective`/`titan_test`/`excluded`), `reason`, `in_benchmark`.

**tiers**: `review_id`, `record_id`, `label_final`, `label_abs`,
`sample_hash`, `neg_rank`, `in_tier_t10/t20/t40`, `in_stability_subset`,
`family_id`. **review_totals**: `n_total_pos`, `n_total_neg`, `family_id`,
`review_type` (whole-cohort counts used by the IPW workload).

**decision files** (`runs/systems/*.parquet`): `system`, `set`,
`review_id`, `record_id`, `decision_keep` (1 keep / 0 exclude),
`is_valid_answer`, optional `score`, `cost_usd`. Any system — including
external tools — can be evaluated by writing this format.

## Benchmarks produced by `freeze`

| Name | Reviews | Use |
|---|---|---|
| `open_sr_bench` | ≤ 10 eval families per review type (target 85–100 reviews) meeting eligibility | **primary** frozen benchmark (frontier LLMs on T20) |
| `open_sr_bench_extended` | every eval-split review | encoder-only extended evaluation |
| `internal_test` | every `test`-split review (15% of legacy families, all sources) | review-level internal holdout |
| `val` | the calibration split | **development only** (thresholds are locked on it) |
| `titan_test` | TITAN-SR test split + external holdout | head-to-head on TITAN-SR's own benchmark |
| `prospective` | published after `temporal.freeze_date` | strictly prospective holdout |

## Modelling stages

| Command | Inputs | Outputs |
|---|---|---|
| `osb train --backbone B --fold k` | `splits/`, `configs/titan_ensemble.yaml` | `runs/B/fold<k>/{best/,holdout_preds.parquet,train_log.jsonl,summary.json}` |
| `osb predict --backbone B --fold k --sets val,open_sr_bench,...` | checkpoint | `runs/B/fold<k>/preds.parquet` (whole cohorts) |
| `osb calibrate --ensemble E` | all member `preds.parquet` | `runs/calibration/E/{scored.parquet,locked_operating_points.json}`, `runs/systems/E@tau*.parquet` |
| `osb llm-sweep` | `frozen/<bench>/tiers.parquet`, `configs/llm_models.yaml` | `runs/llm/<model>/<cond>/responses_seed{0,1}.jsonl`, `runs/systems/<model>__<cond>.parquet` |
| `osb cascade --ensemble E` | calibration + verifier decisions | `runs/systems/cascade__E__<verifier>.parquet` |
| `osb rct train / score / systems` | RCT corpus, splits | `rct/model/` (+ model card), `splits/rct_scores.parquet`, `runs/systems/cochrane_rct_filter.parquet` |
| `osb calibrate --ensemble E --rct auto` | + RCT scores | systems named `E+rct@tau*` |
| `osb evaluate --benchmark NAME [--unblind]` | tiers + decision files | `runs/reports/NAME_T<k>/{leaderboard.json,leaderboard.md,by_review_type.csv,ranking.csv,pareto.csv}`; held-out names need `--unblind` and append to `runs/unblinding_log.jsonl` |
| `osb preregister` | frozen manifests, locked thresholds | copies in `datasets/manifests/` (commit them) |
| `osb audit` | splits | guard report (raises on any leak) |

Hardware: `osb train/predict --hardware rtx3080ti|gpu_24gb|killarney_l40s|killarney_h100|cpu|auto`
(see `configs/hardware/`). Spec overrides for ablations: `osb train ... --override loss.rank_type=none
--override architecture=pooled` (values parsed as YAML).

## Evaluating an external system

Write a decision file with the columns above for the records of
`frozen/open_sr_bench/tiers.parquet` where `in_tier_t20` is true, save it as
`runs/systems/<name>.parquet`, then `osb evaluate --benchmark open_sr_bench`.
Missing records are scored as *keep / invalid*.
