# Backbone screen for TITAN-Open (pre-registered rule)

Committed **before** the screen runs. It decides which encoders form
`titan_open` on Killarney. It uses only the train-fold holdout (reviews
held out of training inside the train split); no val, titan_test or
Open-SR-Bench data is read.

## Why

The local pilot's ModernBERT-large member stayed far behind
BioLinkBERT-large on the holdout (per-review AUC 0.775 vs 0.869). A
diagnosis on the holdout found no bug: with frozen backbones the gap is
already 0.174 and fine-tuning adds the same ~0.055 to both models. The
pilot's titan_test results were seen before this screen; the decision
below rests only on holdout evidence.

## Candidates

`biolinkbert_large`, `biomedbert_large`, `deberta_v3_large`,
`modernbert_large`, each with the specs in
`configs/local/titan_ensemble_3080ti.yaml` (512 tokens, gradient
checkpointing, bf16) and identical budgets.

## Procedure

* Fold 0 of the local data, no `--limit-reviews`.
* 1,500 optimizer steps per run, EMA off (`optim.ema_decay=0`, EMA at
  0.999 would leave most of the weights at their initial values after
  1,500 steps), global batch 32 for every candidate
  (16 x 2 with gradient checkpointing, set by `scripts/screen_arm.py`).
* Two seeds per candidate (`seed=1234`, `seed=1235`).
* Metric: per-review holdout AUC of `logit_final`
  (`evaluation.ranking.summarize_ranking`), averaged over the two seeds.
  Secondary, for information only: holdout specificity at 99% recall.
* Script: `scripts/backbone_screen.sh`; results in
  `runs/screen/results.jsonl`.

## Decision rule

1. A candidate **passes** if its mean AUC is no more than **0.05** below
   the best candidate's mean AUC.
2. `titan_open` = the three passing candidates with the highest mean AUC.
   If fewer than three pass, `titan_open` uses only the passing ones.
3. Ties within 0.005 are broken in favour of architectural diversity
   (prefer a non-BERT architecture), then lower training cost.
4. The outcome and the resulting `ensembles.titan_open` change are
   committed before any Killarney training starts. `titan_sr_baseline`
   is not affected.

The screen is short (about 0.2 passes over the training data), so it
measures how well each backbone starts and adapts, not where it would
converge. That is the question the rule answers: whether a member is
worth its share of the Killarney budget.
