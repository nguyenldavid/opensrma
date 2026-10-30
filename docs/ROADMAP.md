# Status and what still needs to be done

_Last updated: 2026-10-30 (end of the second development session)._

## Where things stand

| Area | State |
|---|---|
| Data adapters (CLEF TAR, Chan 2024, SYNERGY, Cohen, SWIFT, Wallace, SIGIR, TITAN-SR, Europe PMC harvest) | implemented; each validated against the real upstream files |
| PubMed / Europe PMC / OpenAlex clients, caches | implemented; live-validated on small slices |
| Pool reconstruction (Tier 1 / Tier 2, augmentation) | implemented; live-validated on 9 harvested reviews (mean pool recall 0.55 at a deliberately tiny K) |
| Families, splits incl. **review-level `test` holdout**, purge, guards, tiers | implemented + unit-tested; zero-leak audit enforced in `split`, `train`, `osb audit` |
| Unblinding ledger + `osb preregister` | implemented + tested |
| TITAN-Open model (encoders + LoRA decoders), losses, sampler, DDP | implemented; tested with tiny random BERT / ModernBERT / DeBERTa / Qwen3 checkpoints on CPU |
| Hardware profiles (RTX 3080 Ti, 24 GB, L40S, H100, CPU), OOM probe | implemented + tested on CPU; **not yet exercised on a real GPU** |
| Cochrane-recipe RCT classifier (+ PubMed-PT and HF backends) | implemented + tested; corpus → decontaminate → train → score validated live on a 124-record toy corpus (full ~80k corpus not yet built) |
| Calibration, conformal thresholds, RCT stacking features | implemented + tested |
| LLM sweep, cascade, evaluation, reports | implemented; LLM sweep not yet run against a live endpoint |
| Killarney Slurm scripts | written; **not yet run on Killarney** |
| Benchmark results | **none yet** — nothing in this repo claims a result |

## P0 — needed before the first real results

1. **Run the data stages at full scale** (3080 Ti machine:
   `bash scripts/run_local.sh` with `STEPS=data`; Killarney:
   `slurm/run_data_online.sh`). Then inspect:
   `datasets/interim/pmc_harvest/harvest_audit.json` (rejection reasons),
   `datasets/processed/build_summary.json` (per-source counts, pool recall),
   `datasets/splits/audit.json` (must show zero matches).
2. **First real GPU runs on the 3080 Ti**: `osb train --backbone
   pubmedbert_base --fold 0 --max-steps 100` then a large backbone
   (`modernbert_large`) to confirm the OOM probe, gradient checkpointing,
   bf16 and FlashAttention/SDPA paths on real weights; record `seq_per_s`
   in `docs/LOCAL_GPU.md`.
3. **Validate the JATS parser automatically**: for Cochrane reviews that are
   both in PMC OA and in CLEF TAR / Chan 2024, compare the parsed included
   studies against those datasets' labels and report parser precision /
   recall per `positive_source` (table / ref_list / section). Tighten rules
   where precision is low. (No manual labelling needed.)
4. **Credentials and placeholders**: NCBI API key; `OPENALEX_API_KEY` for the
   ~4.6k SYNERGY records without PMIDs; paste the Cao et al. 2025 prompt
   verbatim into `configs/prompts/cao2025.txt`; verify OpenRouter model
   slugs/prices in `configs/llm_models.yaml`.
5. **Killarney specifics**: confirm module names (`module spider`), the
   AIP account string, walltime limits and whether compute nodes have
   internet; adjust `slurm/env.sh`; run `sbatch slurm/smoke.sbatch`.
6. **Baseline sanity**: train `titan_sr_baseline` and check it roughly
   reproduces TITAN-SR's reported ranking metrics on `titan_test`
   (threshold-free `ranking.csv`) before comparing against TITAN-Open.

## P1 — scientific quality

* **Pool reconstruction**: report the `pool_recall` distribution; try the
  LLM query generator (`cohort.query_generator: llm`, gpt-oss-20b via vLLM —
  implemented, untested live); consider a full local PubMed index
  (BM25 + MedCPT over the annual baseline) as in protocol v1.0 (current
  topic-local index is amendment A15).
* **Search strategies in supplements** (docx/pdf/xlsx) are not parsed;
  only inline JATS text is. Same for PRISMA flow numbers inside figures.
* **RCT classifier**: run `osb rct corpus/train`; check recall on included
  studies of RCT-only Cochrane reviews (target ≥ 99%); optionally add a
  Cochrane Crowd export (`rct.crowd_csv`) and compare with PT-only labels;
  decide whether the `+rct` stacking variant or the standalone
  `cochrane_rct_filter` is pre-registered as primary.
* **Calibrator regularisation**: with many features and a small validation
  set the stacker can overfit (visible in the synthetic smoke run for
  `+rct`). Tune `calibration.meta_c` by nested family CV on validation only.
* **Small-GPU ranking loss parity**: micro-batches < 16 shrink review groups;
  a cross-micro-batch memory bank would make the listwise loss identical to
  Killarney.
* **Qwen3-Reranker input format**: currently the generic layout; test the
  model's own instruction template.
* **TITAN-SR external holdout** is title-only; wire
  `EUtilsClient.title_to_pmid` to recover abstracts (optional).
* **Statistics**: decide the Holm family exactly in the pre-registration;
  consider BCa intervals; add a power estimate for ΔJ at the expected
  number of benchmark families.
* **LLM sweep**: dry-run with `--limit 50` against vLLM and one commercial
  model; verify reasoning-parameter names per provider; record cost.

## P2 — engineering

* GitHub Actions CI (CPU): `pytest` + `osb smoke`.
* Version lock / container (Apptainer image for Killarney, Docker for the
  workstation).
* Parallelise pool assembly (BM25 per review is single-threaded) and make
  `build` stream by source to cap RAM below 32 GB.
* `osb figures`: render Figure 1 (cohort flow) and Figure 2 (Pareto) from
  the report CSVs.
* Lint with Google's pylintrc (custom `g-` checks) and fix remaining
  warnings; type-check with pytype or mypy.
* Native Windows support (currently Linux / WSL2).

## Done in this session

* Hardware profiles (`configs/hardware/*.yaml`) layered over backbone specs,
  effective-batch-preserving gradient accumulation, CUDA OOM probe,
  bf16 frozen LoRA bases, non-reentrant checkpointing; `titan_open_lite` and
  `titan_open_local` ensembles; `configs/benchmark_local.yaml`;
  `scripts/run_local.sh`; `docs/LOCAL_GPU.md`.
* Review-level internal **test** holdout (15% of legacy families), purged
  and guarded like every protected split; frozen `internal_test` and `val`
  (dev) benchmarks.
* Defence-in-depth guards (`splits/guards.py`) re-run in `split`, `train`
  and `osb audit`; unblinding ledger and `osb preregister`.
* Cochrane RCT Classifier re-implementation (`src/opensrbench/rct/`),
  leak-free MEDLINE publication-type corpus, `requires_rct` parsing,
  `cochrane_rct_filter` baseline system and `+rct` stacking features.
