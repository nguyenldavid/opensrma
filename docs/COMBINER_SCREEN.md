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

(pending: applied when DeBERTa-v3-large finishes)
