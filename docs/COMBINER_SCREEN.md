# Combiner screen for TITAN-Open (pre-registered rule)

Committed **before** the rule is applied (2026-11-01). It decides how the
`titan_open` members are combined into one score (`combiners.titan_open`
in `configs/titan_ensemble.yaml`) for the Killarney pilot (K1). It uses
only the pilot_v2 train-fold holdout and val; no titan_test data is read.

## Why

`scripts/val_diagnostics.py` (V1, V7) ran a dry run of the protocol's
"lock thresholds on validation, evaluate on test", with the train-fold
holdout as the locking set and val as the test set, using the two
TITAN-Open members finished so far (BioLinkBERT-large, BiomedBERT-large).
The configured combiner (cross-fitted logistic stacker on within-review
features) ranked no better than the plain mean of member sigmoids and
gave a much more conservative primary threshold:

| Combiner | val spec at matched sens 0.99 | holdout-locked `equal_review_lcb` @ 0.99: val sens / spec |
|---|---|---|
| mean_sigmoid | 0.380 | 0.996 / 0.281 |
| stacking | 0.345 | 0.9995 / 0.197 |
| TITAN-SR baseline (mean_sigmoid, 2 members) | 0.319 | 0.995 / 0.267 |

Those two-member numbers were seen when this rule was written (disclosed
here); the decision below uses the three-member result, which does not
exist yet (DeBERTa-v3-large is still training).

## Candidates

`stacking` (current configuration) and `mean_sigmoid` (TITAN-SR's
combiner), for `titan_open` only. `titan_sr_baseline` stays
`mean_sigmoid` (faithful re-implementation).

## Procedure

* Members: all three `titan_open` members of the backbone screen outcome,
  pilot_v2 fold 0 runs in `runs_v2` (`scripts/v2_queue.sh`).
* Locking set: the members' `holdout_preds.parquet` (train-fold holdout,
  287 reviews). The stacker is cross-fitted on it (5 folds by family,
  `meta_c` 1.0, positive weight 5.0: the `configs/benchmark_v1.yaml`
  values) and refit on all of it, exactly as `osb calibrate` does on val.
* Scoring set: val (`preds.parquet`, 335 reviews).
* Primary metric: val equal-review specificity at the threshold locked on
  the holdout with `equal_review_lcb`, target 0.99 (the protocol's
  primary method, A19).
* Script: `scripts/val_diagnostics.py` (`V1` for mean_sigmoid, `V7` for
  stacking), results in `<FNX>/val_diagnostics/results.json`.

## Decision rule

1. Keep `stacking` unless `mean_sigmoid` beats it on the primary metric
   by **at least 0.01** and also reaches val equal-review sensitivity
   >= 0.99 at that threshold.
2. If `mean_sigmoid` is chosen, set `combiners.titan_open: mean_sigmoid`
   (and `titan_open_xl`), record it as protocol amendment A22 and commit
   both before any K1 `--finalize` (the titan_test evaluation).
3. The outcome is appended below and is final for this round.

## Outcome

Applied 2026-11-01 15:10, after DeBERTa-v3-large finished (all three
members: BioLinkBERT-large, BiomedBERT-large, DeBERTa-v3-large; DeBERTa
trained EMA-off locally, see `scripts/v2_queue.sh`). Log:
`logs/val_diagnostics_3m.log`, results `<FNX>/val_diagnostics/results.json`.

| Combiner (3 members) | holdout-locked `equal_review_lcb` @ 0.99: val sens / spec | val reviews below 0.99 | val AUC (per review) |
|---|---|---|---|
| mean_sigmoid | **0.9870** / 0.376 | 21 | 0.884 |
| stacking | 0.9959 / 0.306 | 10 | 0.878 |
| TITAN-SR baseline (mean_sigmoid) | 0.9948 / 0.267 | 10 | 0.861 |

`mean_sigmoid` beats `stacking` on specificity by 0.070, but its val
sensitivity at the locked threshold is 0.987, below the required 0.99.
Clause 1 is not met, so **`stacking` is kept**: `combiners.titan_open`
stays `stacking`, no amendment A22, no config change. K1 runs as
configured.

For the record (not part of the decision): TITAN-Open (stacking) vs the
TITAN-SR baseline on val, paired family bootstrap: per-review AUC +0.016
[0.005, 0.028], spec at matched 99% recall +0.030 [0.010, 0.052].
