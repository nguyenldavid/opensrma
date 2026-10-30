# Study Protocol & Technical Specification — v1.1

**Title:** *Open-SR-Bench: A Prospective, Leak-Audited Benchmark and
Decomposed Cross-Encoder Architecture for Automated Title-and-Abstract
Screening in Systematic Reviews*

**Reporting guidelines:** TRIPOD+AI (2024), PRISMA-DTA, STARD-AI.
**Target venues:** *Annals of Internal Medicine*, *Nature Medicine*,
*The Lancet Digital Health*, *Journal of Clinical Epidemiology*.

v1.1 keeps every element of v1.0 and records each change as an amendment
(§9). Every section names the code that implements it.

---

## 1. Objectives and hypotheses

**Primary.** Build an open, reproducible, strictly decontaminated benchmark
(**Open-SR-Bench**) of systematic reviews published after frontier-model
knowledge cut-offs, and evaluate fine-tuned encoder ensembles
(**TITAN-Open**), open-weight generative verifiers and commercial frontier
LLMs on title/abstract screening.

**Secondary.**
1. Whether the decomposed PICOS soft-min cross-encoder with
   review-conditional conformal calibration mitigates the specificity
   collapse of pooled encoders at ≥ 99% sensitivity (TITAN-SR: 36.2%
   specificity at 99.5% sensitivity).
2. The cost–accuracy Pareto frontier of a two-stage cascade (calibrated
   encoder → Condition-B verifier) versus standalone frontier LLMs.
3. (A7) Head-to-head superiority over TITAN-SR on TITAN-SR's own held-out
   test split, using a faithful re-implementation trained on identical data.

## 2. Data

### 2.1 Training corpus (A1)
All openly available screening collections, ingested automatically
(`src/opensrbench/sources/`, [DATASETS.md](../DATASETS.md)): CLEF TAR
2017–2019, Chan et al. 2024, SYNERGY, Cohen 2006, SWIFT/Howard 2016,
Wallace 2010, SIGIR 2017 and the TITAN-SR manifests, plus harvested reviews
assigned to train/val by the family split. No manual labels.

### 2.2 Benchmark inclusion criteria (`sources/pmc_harvest.py`, `harvest/jats.py`)
A review is eligible if it satisfies:
1. **Temporal window**: published 2025-01-01 → 2026-10-31; reviews after
   the freeze form the prospective partition.
2. **Parseable protocol sections**: objective and eligibility criteria
   extractable from JATS.
3. **Resolvable positives**: ≥ 5 included studies resolving to PMIDs with
   non-empty abstracts, from "studies included" reference lists,
   "characteristics of included studies" tables, or included-studies
   results sections (A9 quality filters).
4. **Reconstructible pool**: verbatim PubMed strategy (Tier 1) or PRISMA
   counts / default K for Tier 2; pool size 100–15,000; pool recall of
   known positives ≥ 0.25 (A9).
5. **Taxonomy**: one of the 10 strata (`text/review_type.py`).

### 2.3 Search-pool reconstruction and labels (`cohort/`)
For review r with positives P_r and search date t_r (reported, else
pub-date − 90 d), C_r = P_r ∪ N_r with PDAT ≤ t_r:
* **Tier 1**: verbatim PubMed strategy bounded by `[1900/01/01 : t_r][dp]`
  (ESearch's 10k cap handled by date slicing, A10).
* **Tier 2**: union of a strict PICO Boolean query and the top-K_r
  BM25 + MedCPT neighbours (RRF) of a broad query, K_r = PRISMA yield
  bounded to [500, 5000] (A15).
* **Labels**: y_final = 1 for P_r; y_abs = 1 additionally for studies
  excluded at full text (from "excluded studies" tables).

### 2.4 Families and zero-leakage audit (`splits/`)
* Edge (u, v) if |P_u ∩ P_v| / min(|P_u|, |P_v|) ≥ 0.10 or
  cos(Embed(Obj_u), Embed(Obj_v)) ≥ 0.88 (MedCPT); positives matched by
  title fingerprint or PMID.
* Connected components are assigned atomically: temporal families to
  Train 75% / Val 10% / Eval 15%; legacy families to Train 80% / Val 5% /
  Test 15% (review-level internal holdout, A19); families
  with prospective or TITAN-SR-test members to those partitions (A7).
* Fingerprints: `pmid`, `title_fp = SHA256(norm(title))`,
  `abstract_fp = SHA256(norm(abstract)[:500])` (A2, A3). Train/val records
  matching any held-out fingerprint are purged; train is also purged
  against val (A6). The audit must report **0** intersections, else
  `LeakageError` (A17).

## 3. Frozen sampling and estimators (`splits/tiers.py`, `evaluation/`)

### 3.1 Tiers
Up to 10 eval families per review type (target R = 85–100; release floor
≥ 50 families and ≥ 500 positive appearances). All positives kept; negatives
ordered by h(r, i) = SHA-256("OPEN-SR-BENCH-V1:" ‖ review_id ‖ ":" ‖
record_id); T_k = first min(k, N_r) negatives, k ∈ {10, 20, 40}; **T20** is
primary.

### 3.2 IPW equal-review estimators
With n_r = min(k, N_r) and w_r = N_r / n_r:
* Sens_r = (1/P_r) Σ_{i∈P_r} 1(d̂ = 1)
* Spec_r = (1/n_r) Σ_{i∈N_r^(k)} 1(d̂ = 0 ∧ valid)
* Work_r = [Σ_{P_r} 1(d̂=1) + w_r Σ_{N_r^(k)} 1(d̂=1)] / (P_r + N_r)
* J = mean_{R+} Sens_r + mean_R Spec_r − 1

**Abstention rule**: timeouts, malformed output, refusals, `UNCERTAIN` and
missing decisions (e.g. budget exhaustion, A14) count as keep and invalid.

### 3.3 Inference
95% percentile CIs from B = 5,000 family cluster-bootstrap replicates
(shared across systems); paired ΔJ with two-sided bootstrap p-values,
Holm-Bonferroni adjusted. All thresholds τ93, τ95, τ97, τ99, τ99.5 and τJ
are locked on validation (out-of-fold stacked scores, A5) before unblinding.

## 4. Evaluation conditions (`llm/`, `cascade.py`)

| System | Condition | Specification |
|---|---|---|
| LLMs | A (zero-shot) | reasoning disabled or lowest effort; T = 0; `<decision>INCLUDE|EXCLUDE</decision>`; ≤ 256 output tokens |
| LLMs | B (structured) | reasoning low / 4,096-token budget; per-criterion MET / NOT_MET / UNCLEAR with quotes, then the decision tag |
| Published baseline | Cao et al. 2025 | exact published prompt pasted verbatim (A16); parser: final 500 chars, YYY vs XXX |
| TITAN-SR baseline | `titan_sr_baseline` | faithful re-implementation on identical splits; raw mean-sigmoid scores; same locked targets (A7) |
| TITAN-Open | locked τ points | single deterministic pass; τ93…τ99.5, τJ |
| Cascade | TITAN-Open → verifier | s < τ99.5 exclude; s ≥ τ93 include; gray zone → Condition-B verifier (gpt-oss-20b local by default) |

**Repeat agreement (§4.1):** max(1, round(0.10·|C_r^(T20)|)) records per
review with the lowest SHA256(record_id + ":stability-v1") are re-run with
seed = 1; percentage agreement and Cohen's κ are reported.

## 5. Implementation map

| Protocol element | Code |
|---|---|
| Harvest + JATS parsing | `harvest/europepmc.py`, `harvest/jats.py`, `sources/pmc_harvest.py` |
| Pool reconstruction | `cohort/query.py`, `cohort/ranking.py`, `cohort/reconstruct.py` |
| Families, splits, audit, tiers | `splits/families.py`, `splits/assign.py`, `splits/decontam.py`, `splits/tiers.py` |
| TITAN-Open model + losses | `modeling/model.py`, `modeling/losses.py`, `modeling/sampler.py`, `modeling/train.py` |
| Calibration | `calibration/stacking.py`, `calibration/thresholds.py`, `calibration/run.py` |
| LLM sweep + budget | `llm/prompts.py`, `llm/client.py`, `llm/runner.py` |
| Estimators + bootstrap | `evaluation/metrics.py`, `evaluation/bootstrap.py`, `evaluation/report.py` |
| Killarney execution | `slurm/` + [KILLARNEY.md](KILLARNEY.md) |

## 6. Exhibits

| Exhibit | Content | Source |
|---|---|---|
| Figure 1 | Cohort flow: harvested → kept (with rejection reasons) → families → purge counts (0 leaked) → frozen T20 | `pmc_harvest/harvest_audit.json`, `processed/build_summary.json`, `splits/audit.json`, `frozen/*/manifest.json` |
| Table 1 | Leaderboard on T20: J, Sens, Spec, IPW workload (95% CIs), ΔJ vs reference (Holm), invalid rate, cost, repeat agreement | `reports/open_sr_bench_T20/leaderboard.md`, `runs/llm/*/summary.json` |
| Figure 2 | Sensitivity (90–100%) vs work saved: TITAN-SR baseline, TITAN-Open, LLMs, cascade | `reports/*/pareto.csv` |
| Table 2 | Strata by review type; false negatives by abstract-level vs full-text-only evidence | `reports/*/by_review_type.csv`, records with `label_abs` |
| Table S1 | Same on `titan_test` (TITAN-SR's own split) and `prospective` | `reports/titan_test_T20`, `reports/prospective_T20` |

## 7. Pre-registration procedure

1. Run data stages; commit `frozen/open_sr_bench/manifest.json` to
   `datasets/manifests/`.
2. Train, predict, calibrate; commit every
   `locked_operating_points.json`.
3. Only then run `osb evaluate` on held-out benchmarks.

## 8. Limitations

Harvested labels inherit the reviews' own reporting; included-study
parsing and Tier-2 reconstruction are automatic and imperfect (quantified by
`pool_recall` and the harvest audit); open-access reviews may differ from
paywalled ones; TITAN-SR's external holdout is title-only.

## 9. Amendments to v1.0

| # | Amendment | Rationale |
|---|---|---|
| A1 | Training uses all open screening datasets; benchmark labels come from automated harvest only | no manual labelling; maximum data |
| A2 | Empty normalized text fingerprints to "" (v1.0 hashed "" and its `- {""}` guard never fired) | otherwise every empty-abstract record collides |
| A3 | Abstract window applied after normalization | robust to whitespace/punctuation differences |
| A4 | Conformal threshold = exact order statistic s_(⌊(n+1)α⌋) (v1.0 interpolated with `np.quantile`) | interpolation can violate the finite-sample guarantee |
| A5 | Stacker cross-fitted on validation families; thresholds locked on OOF scores | v1.0 fitted and locked on the same data (optimistic) |
| A6 | Train additionally purged against validation | calibration data must be unseen by every member |
| A7 | TITAN-SR test split held out as `titan_test`; faithful TITAN-SR baseline on identical splits | leakage-free head-to-head on their benchmark and ours |
| A8 | Ensemble folds drawn from train families only; validation reserved for calibration | v1.0 used "val OOF predictions" ambiguously |
| A9 | Harvest quality filters (no SR/MA positives except umbrella reviews; no positives after the review; pool recall ≥ 0.25) | removes citation-parsing artefacts |
| A10 | Search-date parsing from methods text; ESearch date slicing beyond 10k hits | exact t_r ceilings; complete Tier-1 pools |
| A11 | Soft-min computed with masked log-sum-exp; document-first layout with markers after criteria | v1.0 could yield NaN gradients; layout supports decoders |
| A12 | Review-grouped batch sampler | in-review ranking loss needs same-review pairs in a batch |
| A13 | Per-backbone max length (512 for BioLinkBERT/DeBERTa, 1024 ModernBERT, 1536 Qwen3) | v1.0's 1024 exceeds two backbones' positional range |
| A14 | Missing decisions (e.g. budget exhaustion) follow the abstention rule; budget reserved before each call | no silent selection; no overspend |
| A15 | "Local PubMed index" realised as a topic-local candidate index (broad query) re-ranked by BM25 + MedCPT | full-PubMed indexing deferred; recall reported per pool |
| A16 | Cao et al. prompt must be pasted verbatim; the runner refuses a placeholder | no paraphrased "exact" baselines |
| A17 | Leakage audit raises `LeakageError` instead of `assert` | asserts vanish under `python -O` |
| A18 | Pool augmentation for cited-reference sources (TITAN-SR, Chan, SIGIR) with title-fingerprint guards | realistic prevalence without label noise |
| A19 | Review-level internal `test` holdout: 15% of legacy families (`internal_test`), purged/guarded like other protected splits | a held-out estimate on the legacy distribution independent of the post-cutoff harvest |
| A20 | Guards re-run before every training run and in `osb audit` (family, review-key and record disjointness) | defence in depth against stale or edited artefacts |
| A21 | Held-out evaluation requires `--unblind`; every look is logged with commit and threshold hashes; `osb preregister` | makes "leakage by iteration" auditable |
| A22 | Cochrane-recipe RCT classifier (MEDLINE publication-type labels, decontaminated) as stacking features and as a standalone RCT-filter baseline | design classification is the dominant exclusion reason in RCT-only reviews |
| A23 | Hardware profiles may change only batch/memory settings; effective batch preserved by accumulation | identical recipe on Killarney and a 12 GB workstation |
