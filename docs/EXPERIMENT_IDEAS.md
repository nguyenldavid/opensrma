# Experiment ideas (backlog)

Candidate experiments that came up while preparing the Killarney run.
None of these is part of the protocol yet. Anything adopted must be decided
on train / holdout / validation evidence only and committed (as a protocol
amendment or a pre-registered screen like [BACKBONE_SCREEN.md](BACKBONE_SCREEN.md))
before any Killarney training or unblinding. titan_test is never used to
choose between them.

Status: **idea** (not started), **running**, **done** (result linked).

## Input representation (the 512-token budget)

Measured on pilot_v2 training data (2026-10-31), the 512-token layout
keeps on TITAN-SR reviews a median 72% of the criteria (worst decile
<= 36%) and 87% of the abstract (only 38% of abstracts fit whole); Chan
reviews fit almost entirely. The regex splitter (`text/criteria.py`) labels
98% of criteria only Include/Exclude (< 2% get a PICO role), and Chan
reviews mostly become a single criterion, so the soft-min veto rarely acts
there.

| # | Idea | Status | Notes |
|---|---|---|---|
| I1 | **LLM-structured PICOS criteria**: rewrite each review's criteria once (cached) into short atomic, role-typed items (P / I / C / O / S / exclusions), each item quoting a span of the source text | idea | fixes Chan single-blob criteria, truncation and missing roles at once; gpt-oss-20b (Killarney) or a smaller local model; validate span faithfulness |
| I2 | **Section-aware abstract budget**: use PubMed structured-abstract labels; drop BACKGROUND first, keep METHODS / PARTICIPANTS | idea (sized) | 40% of TITAN-SR record abstracts are structured; BACKGROUND/INTRODUCTION is a median 33 tokens (8% of the abstract), roughly the median truncation (13%). Modest gain, only for structured abstracts; pairs better with I1 |
| I3 | **Per-criterion scoring**: one `[CLS] criterion [SEP] title + abstract` pass per criterion, soft-min over passes | idea | full abstract for every criterion; ~11x compute on TITAN-SR reviews |
| I4 | **Long-context members**: Qwen3-Reranker-4B at 2-4k tokens (titan_open_xl, H100) | idea | removes truncation entirely |
| I5 | **BioClinical-ModernBERT-large** (8k context, PubMed + clinical pretraining) | done: **failed** | screen addendum A: holdout AUC 0.775 at 1024 tokens, 0.764 at 512 vs 0.860 (BioLinkBERT-large); long context +0.011 |
| I6 | DeBERTa-v3-large at 1024 tokens (relative positions only) | idea | cheap screen arm |

## Design-based exclusion rules ("stage 0") and training-set refinement

Many screening exclusions are about study design (case reports, letters,
editorials, narrative reviews, protocols, animal studies). Measured on the
pilot_v2 training split, these cues are **not** safe global exclusions:
PubMed "Case Reports" records are 48% included, "we present a (rare) case"
42%, "-year-old" 35%, "case series" 22%, "Review" 12% (many reviews include
case reports: rare diseases, adverse events, diagnostics). They must be
conditioned on what each review allows.

| # | Idea | Status | Notes |
|---|---|---|---|
| R1 | **Review-conditional design veto** (scan done: **no safe rule**, see R6): LLM reads each review's criteria (cached, quoted) -> allowed/excluded designs; records typed by PubMed publication types + phrase rules (case report, case presentation, we present a rare case, -year-old, case series, perspectives, review, consensus statement, letter / editorial / commentary) + LLM-proposed regexes; a rule becomes a hard veto only if its train inclusion rate has 95% upper bound <= 0.1%, confirmed on val, then locked | idea | exclude-only (abstention rule intact); vetoed positives count against sensitivity; report as a TITAN-Open component with an ablation |
| R2 | **Refined training set**: drop R1-vetoed negatives from training so the sampler spends its budget on hard negatives | idea | the same filter must run at inference |
| R3 | **Cue features for the stacker** (soft): phrase / publication-type flags x review-allowed designs | idea | usable even where a hard veto is unsafe |
| R4 | First check for R1-R3: are cue-matching negatives over-represented among the high-scoring negatives that cost specificity at 99%? (val predictions only) | done: **mostly no** | pilot_v2 val, 3 members: hard negatives (scoring >= the review's lowest positive) are 28-32% of negatives; case-report / letter / 'we present a case' / '-year-old' / case-series / consensus cues are 0-0.5% of hard negatives (below their share of all negatives: the cross-encoders already demote them). Only PubMed 'Review' is sizeable (8-11% of hard negatives, 209-315), but 163 val positives are also typed Review; in RCT-only reviews just 14-26 hard negatives are Reviews. Little headroom for R1/R2 on TITAN-SR + Chan |
| R6 | Design-cue scan (`scripts/design_cue_scan.py`): 201 cues (phrases, every PubMed publication type, MeSH incl. Animals-without-Humans, language, journal, structured-abstract labels) x 6 review conditions on pilot_v2 train/val | done: **negative** | 0 of 1,206 cue x condition cells has a train inclusion upper bound <= 0.1%, none even <= 1%. Best cells (e.g. systematic review / meta-analysis / editorial / chart review in RCT-only reviews) still have 1.7-5% inclusion in train and remove 0-4 hard val negatives each. Causes: cue imprecision (RCT abstracts mentioning case reports) and label noise (N1) |
| N1 | **Positive-label noise in TITAN-SR**: 7.1% of TITAN-SR train/val positives are reviews, editorials, letters, guidelines or comments by publication type / title (e.g. 'Albumin in adult cardiac surgery: a narrative review' included in an RCT review); Chan 2024: 0.3% (title only) | diagnostic done (V2) | val, stricter flag (non-primary PT or title, minus records also typed RCT / trial / case report / observational): 4.9% of positives (TITAN-SR 5.7%, Chan 2.6%). They rank low (mean within-review percentile 0.34-0.36 vs 0.55 for other positives) and are the lowest-scoring positive in 30-34 of 328 reviews; leaving them out of scoring would raise oracle spec@99 by 0.019-0.027 (diagnostic only, labels unchanged). But only 2 sit in RCT-only reviews: most are in scoping / broad reviews whose criteria may admit them, so a training filter limited to reviews that exclude reviews touches very few positives. Expected gain from N1 training: small |
| R5 | Publication types are present for only ~50% of records: find which sources lack them and backfill from the PubMed cache | done (V5): partly feasible | pilot_v2: Chan 2024 records have no PMID and no publication types (0%); TITAN-SR 74%. Exact title-fingerprint match against the local PubMed cache (1.35M titles) recovers 28% of Chan records (all with publication types); the rest needs an ESearch title lookup (network; ~490k titles, ~14 h at 10 req/s with an API key) |

## Bi-encoder (sentence-embedding) PICO matching on CPU

| # | Idea | Status | Notes |
|---|---|---|---|
| B1 | **Per-criterion bi-encoder veto**: embed each criterion (MedCPT query encoder) and each abstract sentence (MedCPT article encoder) on CPU; per criterion take the best-matching sentence; a record is ruled out only if some criterion has no sentence above a recall-calibrated threshold | done: **negative** (regex criteria) | pilot_v2 val, threshold fit on 250 train reviews at 99.5% pooled recall: per-review AUC 0.60-0.71 (cross-encoders ~0.86); at 99.5-99.8% equal-review sensitivity only 1.6-5.1% of negatives ruled out (best: objective vs whole document, MedCPT). NeuML/pubmedbert-base-embeddings slightly worse than MedCPT. B2 (LLM PICO items) remains the real test |
| B2 | B1 with LLM-structured PICO items (I1) instead of regex fragments | idea | the PICO-decomposition version of B1 |
| B3 | Bi-encoder scores as stacker features or as a cheap stage-0 high-recall filter before the cross-encoders | idea | |

## Calibration, splits, ensemble

| # | Idea | Status | Notes |
|---|---|---|---|
| C1 | Source-stratified ("Mondrian") calibration: lock thresholds per benchmark on val reviews from matching sources | idea | addresses val -> titan_test shift |
| C2 | Minimum number of val reviews per calibration stratum (>= 400; >= 800 for review-level conformal at 99.5%) | idea | protocol amendment |
| C3 | Family graph: minimum number of shared included studies for overlap edges and a recalibrated MedCPT cosine threshold (largest family still 7,252 of 16,993 reviews on pilot_v2) | idea | changes leakage protection: PI decision + amendment |
| C4 | Matched-sensitivity secondary comparison (specificity where both systems reach equal-review sensitivity 0.99 / 0.995) | idea | removes the threshold under-coverage confound |
| C5 | gpt-oss-20b verifier cascade on the gray zone | idea | Killarney (H100) |
| T1 | BioLinkBERT-large on pilot_v2 for 3 epochs: holdout AUC 0.859 / 0.860 / 0.857 (spec@99 0.65); BiomedBERT-large 0.857 / 0.858 / 0.857 (spec@99 0.652): flat after epoch 0 and equal to the 1,500-step screen; more steps are not the bottleneck | done | points to input representation (I1-I4) |
| C6 | EMA warm-up (or EMA off) for short runs; log encoder LR and denser loss | idea | small recipe fixes |

## Data

| # | Idea | Status | Notes |
|---|---|---|---|
| D1 | **Full-pool training data**: add the legacy benchmarks (CLEF TAR, SYNERGY, SWIFT, Cohen, SAS) to training | idea (data built) | `data_legacy` (2026-11-01, build/split/freeze in 10 min on CPU): 17,052 reviews after build (19,939 after dedupe), largest family 7,327 (pilot_v2 7,252: the giant family barely grows). Train gains only 75 reviews but +468k records (CLEF 242k, SYNERGY 124k, SWIFT 94k) at ~1.8% prevalence, i.e. real full screening pools, vs. pilot_v2 train 247k records at 31% prevalence. 4 CLEF reviews enter val; titan_test keeps 3,635 reviews but 24 CLEF reviews replace 24 Chan reviews, so it is **not** the pilot_v2 titan_test (Killarney K1 stays on pilot_v2). Test: one screen arm trained on legacy train vs pilot_v2 train, scored on the val reviews common to both; the sampler's negatives-per-review cap decides how much of the pool is used |

## Validation diagnostics (2026-11-01, `scripts/val_diagnostics.py`)

pilot_v2 val (335 reviews, 13,157 records, 4,227 positives), members
finished so far: titan_sr_pubmedbert, titan_sr_biolinkbert,
biolinkbert_large, biomedbert_large (DeBERTa-v3-large still training).
Deployable thresholds are locked on the train-fold holdout (287 reviews)
and applied to val, the dry run of "lock on val, evaluate on titan_test".
No titan_test data is read.

| # | Experiment | Result |
|---|---|---|
| V1 | TITAN-Open (2 members) vs TITAN-SR, oracle per-review metrics, paired family bootstrap | AUC 0.885 vs 0.862 (+0.023, 95% CI 0.013-0.036); spec@99 0.717 vs 0.677 (+0.039, 0.020-0.061); spec@99.5 +0.039 (0.020-0.061) |
| V1 | Same, one global threshold locked on the holdout (equal_review, target 0.99) | val sensitivity 0.994 vs 0.994, specificity 0.328 vs 0.300; equal_review_lcb: 0.996 / 0.281 vs 0.995 / 0.267. Within-review z-score combiners collapse (spec ~0.09) under a global threshold; mean_sigmoid stays the combiner |
| C4 | Matched sensitivity on val (in-sample threshold, descriptive) | spec at equal-review sens 0.99: Open 0.380 vs SR 0.319; at 0.995: 0.315 vs 0.260; at 0.95: 0.580 vs 0.501 |
| V3 | Error analysis (Open) | spec@99 Chan 0.835 vs TITAN-SR 0.640; falls with positives per review (0.85 for <= 3 positives, 0.56 for > 14; Spearman -0.46), noisy-positive share (-0.42), prevalence (-0.38) and criteria length (-0.26; truncation +0.23). RCT-only reviews 0.85 vs 0.66 others. The 10 worst reviews (spec@99 = 0) are small high-prevalence TITAN-SR pools (5-63 records, ~50% included), mostly scoping / methods reviews |
| V6 | Review-adaptive thresholds: global threshold on the ensemble logit re-centred on the review's own unlabelled scores (minus max / top-3 mean / quantiles, partial shifts) | **negative**: best holdout variant gains 0.02-0.07 val specificity only by losing sensitivity (val 0.987-0.988 vs target 0.99); at matched sensitivity no variant beats the plain logit. The gap between oracle per-review (0.72) and global (0.33) specificity is not closed by simple re-centring |
| V4 | MedCPT bi-encoder z-scores as an extra ensemble feature (B3), weight chosen on the holdout | running (MedCPT scoring of val + holdout on CPU) |
