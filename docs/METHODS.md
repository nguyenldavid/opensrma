# Methods: TITAN-Open, and how it is designed to beat TITAN-SR

## 1. What TITAN-SR does (the baseline)

From the TITAN-SR reproducibility package (JCEPI-D-26-00715):

* input `[CLS] eligibility_criteria [SEP] "title. abstract" [SEP]`,
  truncated to 512 tokens (criteria only — no objective);
* PubMedBERT-base / BioLinkBERT-base (110 M), weighted-mean of sigmoids;
* 10:1 false-negative-weighted BCE, class-balanced `WeightedRandomSampler`,
  AdamW 2e-5, effective batch 128, ≤ 10 epochs, early stopping on AUC;
* training data: PMC reviews whose candidates are the **cited references**
  (~45 per review, ~40% positive), Track B, and (unreleased) Chan data;
* a global validation threshold per target recall on raw ensemble scores.

Reported weakness targeted by this project: at 99.5% sensitivity the
specificity collapses to ~36%.

`configs/titan_ensemble.yaml` contains a faithful re-implementation
(`titan_sr_pubmedbert`, `titan_sr_biolinkbert`, ensemble
`titan_sr_baseline`, combiner `mean_sigmoid`). It is trained on exactly the
same splits as TITAN-Open, which turns every comparison into a paired,
same-data test.

## 2. TITAN-Open: what changes and why

| # | Change | Why it matters at ≥ 99% sensitivity |
|---|---|---|
| D1 | **Realistic pools.** Train on full search pools (CLEF TAR, SYNERGY, Cohen, SWIFT, Wallace) and *augment* cited-reference pools (TITAN-SR PMC/Track B, Chan, SIGIR) with date-bounded retrieved hard negatives. | TITAN-SR never sees the 1–5% prevalence, lexically similar negatives that decide specificity in real screening. |
| D2 | **Superset data.** All open datasets incl. TITAN-SR's own manifests and the Chan corpus (~20k reviews). | More reviews ⇒ better criteria generalisation; no data disadvantage vs TITAN-SR. |
| D3 | **Objective + decomposed criteria input**, 1024 tokens for ModernBERT, 1536 for the decoder. | Criteria-only 512-token inputs truncate exclusion rules; the objective disambiguates PICO. |
| D4 | **Criterion soft-min veto head** with learnable gates (`model.py`). | A single violated criterion can veto a record — matching how screeners exclude — while the global head keeps recall. |
| D5 | **Dual heads**: abstract-stage and final; full-text-only exclusions are ignored by the final head. | Removes false-negative gradients on abstracts humans kept; uses all label levels. |
| D6 | **Asymmetric focal loss** (γ+ = 0.5, γ− = 2) + **in-review listwise ranking**. | Focuses capacity on hard negatives and directly optimises within-review ordering — the quantity a per-review threshold depends on. |
| D7 | **Review-grouped hard-negative sampler**. | Makes the ranking loss meaningful; half the negatives are full-text exclusions or top-ranked retrieval neighbours. |
| D8 | **Large, diverse ensemble**: ModernBERT-large, BioLinkBERT-large, DeBERTa-v3-large × 5 family folds (+ Qwen3-Reranker-4B LoRA in `titan_open_xl`), LLRD, EMA, bf16. | Variance reduction exactly where tails matter; decoder rerankers are the current state of the art for relevance. |
| D9 | **Review-conditional stacking** (within-review z-scores, rank percentiles, fold and cross-model dispersion, cohort size), cross-fitted on validation families. | Raw scores are not comparable across reviews; normalising within each candidate pool is the main lever against the 99.5% specificity collapse. |
| D10 | **Exact split-conformal thresholds** (or equal-review / LCB variants) locked on out-of-fold validation scores. | Finite-sample sensitivity guarantee without optimistic bias; no post-hoc operating points. |
| D11 | **Two-stage cascade** to a local gpt-oss-20b verifier for the gray zone. | Spends LLM compute only where the encoder is uncertain; zero API cost on Killarney. |

## 3. Model details

```text
[CLS] title [SEP] abstract [SEP] objective [SEP] c1 [CRIT] c2 [CRIT] ... cK [CRIT] [SEP]
```

* Criteria come from `text/criteria.py` (enumerations, bullets, sentences;
  role prefixes `Include:` / `Exclude:` / `Population:` ...), at most 12–16.
* Each `[CRIT]` marker *follows* its criterion, so the layout works for
  bidirectional encoders and causal decoders. Marker positions are passed
  explicitly (`crit_positions`).
* z_k = MLP(h_[CRIT_k]); softmin = −(1/t)·log(mean_k exp(−t·z_k)), t = 4.
  Rows without criteria use the global logit; masked entries never enter the
  log-sum-exp (no NaN gradients).
* logit_final = σ(g_f)·global_final + (1 − σ(g_f))·softmin (same for the
  abstract head with its own gate).

Loss (`modeling/losses.py`):
L = ASL(final; ignore FT-excluded) + 0.5·ASL(abstract) + 0.5·L_rank, where
L_rank is listwise: −log softmax of each positive against the negatives of
the same review in the batch (pairwise hinge available).

Optimisation: AdamW (β = 0.9, 0.98), LLRD 0.9, head LR 10×, cosine
schedule with 6% warm-up, EMA 0.999, gradient clipping 1.0, bf16 autocast,
DDP over all GPUs of a node. Early stopping on mean per-review
specificity at 99% recall on the held-out family fold.

## 4. Calibration and operating points

1. Fold logits are averaged per backbone; features are computed per
   (set, review) over the **whole candidate pool**.
2. A class-weighted logistic stacker is cross-fitted on validation families
   (5 folds); thresholds are locked on its out-of-fold scores; the final
   stacker is refit on all validation data and applied to held-out sets.
3. `pooled_conformal`: τ = s_(k), k = ⌊(n+1)α⌋ of the sorted validation
   positive scores ⇒ P(miss) ≤ α for a new exchangeable positive. With too
   few positives (k = 0) τ = −∞ (keep everything) — the pipeline reports
   this rather than pretending to certify the target.
4. Operating points: τ93, τ95, τ97, τ99, τ99.5 and τJ (equal-review
   Youden-optimal), all locked before unblinding.

## 5. Statistics

* Primary endpoint: equal-review Youden's J on T20 with IPW workload
  (protocol §3.2); abstentions/invalid answers count as *keep*.
* 95% percentile CIs from 5,000 family cluster-bootstrap replicates; every
  system shares the same replicates, so paired ΔJ distributions are exact.
* Two-sided bootstrap p-values, Holm-Bonferroni across comparisons.
* Threshold-free per-review ranking metrics (AUC, spec@95/99/99.5 recall,
  WSS@95; ties included) are reported on whole cohorts for direct
  comparison with TITAN-SR's published metric definitions.

## 6. Pre-specified comparisons and ablations

Primary (Holm family): TITAN-Open@τ99 and @τ99.5 vs titan_sr_baseline at
the same targets, on `open_sr_bench` and `titan_test`; cascade vs the best
standalone frontier LLM (Condition B) on cost-adjusted J.

Ablations (single-fold, `--override`):

| Ablation | Override |
|---|---|
| no soft-min head | `--override architecture=pooled` |
| criteria-only 512-token input | `--override input_format=titan_sr` |
| no ranking loss | `--override loss.rank_type=none` |
| pairwise instead of listwise | `--override loss.rank_type=pairwise` |
| BCE instead of ASL | `--override loss.type=weighted_bce --override loss.pos_weight=10` |
| no FT-excluded masking | `--override loss.ft_excluded_policy=negative` |
| class-balanced sampler | `--override sampler.type=class_balanced` |
| no hard negatives | `--override sampler.hard_negative_fraction=0` |
| no augmented pools | rebuild with `cohort.augment_incomplete_pools: false` |
| raw-score thresholds | calibrate with combiner `mean_sigmoid` |

## 7. Targets (hypotheses, not results)

On `open_sr_bench` T20 at τ99.5: specificity ≥ 0.80 (TITAN-SR reported
≈ 0.36), equal-review sensitivity ≥ 0.99 with the conformal guarantee, and
a paired ΔJ vs `titan_sr_baseline` whose 95% CI excludes 0 after Holm
correction. These are pre-registered hypotheses to be tested, not claims.
