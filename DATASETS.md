# Datasets

Open-SR-Bench uses **every openly available title/abstract screening
collection** we could ingest automatically, plus an automated harvest of
post-cutoff systematic reviews. No step requires manual labelling: labels
are the review authors' own inclusion decisions, and protocol text
(objective + criteria) comes from the source files or from the review's
structured PubMed abstract.

All sources are configured in [`configs/datasets.yaml`](configs/datasets.yaml)
and downloaded by `osb data --stages download`. Counts below are what the
adapters produce from the current upstream releases (before cross-source
de-duplication, decontamination and text filtering).

## Summary

| Source (`key`) | Reviews | Records | Final incl. | Abstract-stage labels | Pool type | Role | Licence |
|---|---:|---:|---:|---|---|---|---|
| CLEF eHealth TAR 2017–2019 (`clef_tar`) | 128 | 591,558 | 3,632 | yes (12,355) | full Boolean search | train/val | research use (see repo) |
| Chan et al. 2024, Mendeley V1 (`chan2024`) | 8,608 | 544,151 | 93,332 | yes (284,693) | partial (median 48/review) → **augmented** | train/val | CC BY 4.0 |
| SYNERGY (`synergy`) | 43 entries (31 after removing its 12 Cohen 2006 copies) | 187,542 (177,912 non-Cohen) | 4,036 (3,438) | no | full search | train/val | CC0 |
| Cohen et al. 2006 (`cohen2006`) | 15 | 18,733 | 846 | yes (2,401) | full search | train/val | public research data |
| SWIFT-Review / Howard et al. 2016 (`swift2016`) | 5 | 96,134 | 6,032 | no | full search | train/val | CC0 |
| Wallace et al. 2010 (`sas2010`) | 3 | 10,266 | 695 | no | full search | train/val | public (GitHub) |
| SIGIR 2017 SysRev collection (`sigir2017`) | 176 | 7,326 | 2,997 | FT-excluded refs | cited refs only → **augmented** | train/val | public (GitHub) |
| TITAN-SR manifests: PMC + Track B (`titan_sr`) | 11,248 | 413,988 | 157,040 | no | cited refs only → **augmented** | train/val; **their test split held out** | CC BY 4.0 (manifests) |
| TITAN-SR external holdout (`titan_sr`) | 22 | 142,504 | 617 | no | titles only | `titan_test` benchmark | CC BY 4.0 |
| **Open-SR-Bench harvest** (`pmc_harvest`) | ~15–25k expected | rebuilt pools | from included-studies lists | FT-excluded tables when present | Tier 1 exact / Tier 2 hybrid | **eval** (+ train/val per family split) | PMC OA licences |

"Augmented" = the source lists only cited/screened subsets, so the `pools` stage
adds date-bounded retrieved negatives (`cohort.augment_k`, default 1,000 per
review) to turn it into a realistic screening task (prevalence ≈ 1–5%
instead of ≈ 40%).

## Label semantics (canonical schema)

| Column | Meaning |
|---|---|
| `label_final` | 1 = included in the final review (primary endpoint) |
| `label_abs` | 1 = passed title/abstract screening (always ≥ `label_final`) |

Records with `label_abs = 1, label_final = 0` were excluded only at full
text. The TITAN-Open final head ignores them in its loss
(`loss.ft_excluded_policy: ignore`) so the model is never penalised for
keeping an abstract that humans also kept; the abstract head learns them as
positives.

## Per-source details

### CLEF eHealth TAR 2017–2019 — `clef_tar`
* **What**: Cochrane DTA (2017/2018), plus intervention, prognosis and
  qualitative (2019) reviews; the complete PubMed results of each review's
  Boolean search with abstract- and content-level relevance.
* **Where**: <https://github.com/CLEF-TAR/tar> (git clone).
* **Parsing**: all `*abs*` / `*content*` qrels per year; for topics that
  recur, the most recent year wins (later releases fixed errors).
  Objectives and criteria come from the protocol XML (Types of studies,
  Participants, Index tests / Interventions, Target conditions, Reference
  standards, Outcomes). `99999999` placeholder PMIDs are dropped.
* **Caveats**: ~9% of topics lack protocol text; it is filled from the
  Cochrane review's structured PubMed abstract during `enrich`.
* **Cite**: Kanoulas et al., CLEF 2017/2018/2019 TAR overviews.

### Chan, He, Leung & Verspoor (2024) — `chan2024`
* **What**: title/abstract screening data for Cochrane reviews with
  background, objective and selection criteria inline. Label `1` = final
  inclusion, `0.5` = full-text exclusion, `0` = (estimated) T/A exclusion.
* **Where**: Mendeley Data V1, DOI 10.17632/7sgmg89zb6.1 (466 MB zip, SHA-256
  verified on download). Associated article: Res Synth Methods (2025),
  doi:10.1017/rsm.2025.1. This is the "Chan" component of TITAN-SR's
  all-source model that TITAN-SR's package does not ship.
* **Parsing**: streamed in 50k-row chunks from all five members (dev, val,
  random/heart/HIV test); reviews keyed by Cochrane CD number.
* **Caveats**: no PMIDs → cross-source matching and decontamination use
  the title/abstract fingerprints. The authors describe 0-labels as
  approximations of screening outcomes. Pools are partial (median 48
  records per review, 52% passed T/A screening), so the pool-augmentation
  stage adds retrieved negatives; augmented candidates whose title matches
  a labelled record are discarded so no positive re-enters as a negative.

### SYNERGY — `synergy`
* **What**: 26 fully labelled screening datasets (De Bruin et al., 2023).
* **Where**: <https://github.com/asreview/synergy-dataset>; review DOIs and
  eligibility criteria from `datasets.toml`.
* **Parsing**: every `*_ids.csv`; records keyed by PMID, else DOI (resolved
  to PMID through PubMed; 34,293 records), else OpenAlex id (4,631). Software-engineering reviews
  are dropped (`biomedical_only: true`). SYNERGY's copies of the Cohen 2006
  reviews share the `cohen2006` review key, so the richer copy is kept.
* **Caveats**: records without PMID need OpenAlex (`OPENALEX_API_KEY` for a
  personal quota); unresolved ones are dropped if they have no text.

### Cohen et al. (2006) drug-class reviews — `cohen2006`
* **What**: 15 AHRQ/DERP drug-class reviews with abstract-triage and
  article-triage decisions per PubMed citation.
* **Where**: <https://dmice.ohsu.edu/cohenaa/epc-ir-data/epc-ir.clean.tsv>.
* **Caveats**: no criteria text is distributed; a templated objective /
  criterion per drug class is used and flagged
  (`extra.criteria_quality = templated`).

### SWIFT-Review datasets (Howard et al., 2016) — `swift2016`
* **What**: four OHAT environmental-health reviews (BPA, PFOS/PFOA,
  transgenerational inheritance, fluoride) and one CAMARADES pre-clinical
  review (neuropathic pain).
* **Where**: Springer supplementary files of doi:10.1186/s13643-016-0263-z.
* **Caveats**: templated objectives/criteria (flagged); Fluoride/Neuropain
  ship their own text, the others are PMID lists.

### Wallace et al. (2010) — `sas2010`
* **What**: proton-beam, micro-nutrients and COPD reviews with local
  titles/abstracts. <https://github.com/bwallace/citation-screening>.
* **Caveats**: templated objectives/criteria (flagged); no PMIDs.

### SIGIR 2017 SysRev collection (Scells et al.) — `sigir2017`
* **What**: Cochrane reviews with their included studies and the studies
  excluded at full text. <https://github.com/ielab/SIGIR2017-SysRev-Collection>.
* **Parsing**: `citations.json` (`included` true/false); non-included
  citations are treated as full-text exclusions (`label_abs = 1`).
* **Caveats**: no search negatives → pool augmentation.

### TITAN-SR reproducibility package — `titan_sr`
* **What**: TITAN-SR's released PMC (7,723 reviews) and Track B (3,525)
  labels, criteria and `stratified_split_v2` review-level split, plus the
  22-review title-only external holdout.
* **Where**: <https://github.com/InvestmentMDideas/TITAN-SR> (manifests
  CC BY 4.0; abstracts reconstructed from PubMed).
* **Why**: (1) our training data is a strict superset of TITAN-SR's;
  (2) reviews in TITAN-SR's **test** split (any source, matched by PMC id,
  Track B id or Cochrane CD number) and the external holdout are routed to
  a separate `titan_test` benchmark and decontaminated from training, so we
  can compare on *their* benchmark leakage-free.
* **Caveats**: PMC pools are cited references (~40% prevalence) → pool
  augmentation; the external holdout has titles only.

### Open-SR-Bench harvest — `pmc_harvest`
* **What**: open-access systematic reviews (all 10 strata) published in the
  evaluation window, from Europe PMC (`harvest.query` in the config; the
  title query alone returns ~44.6k hits for 2025-01-01..2026-10-31).
* **Automated labelling** (protocol criteria 1–5):
  1. Temporal window from `firstPublicationDate`.
  2. Objective + eligibility criteria parsed from JATS sections.
  3. Included studies from Cochrane "studies included" reference lists,
     "characteristics of included studies" tables (xref ranges expanded,
     author–year and bracketed numbers matched), or results sections;
     references resolved to PMIDs via `<pub-id>`, Europe PMC's reference
     API (DOI/title alignment) and PubMed DOI lookup; ≥ 5 resolved
     positives required. "Excluded studies" tables give `label_abs = 1`.
  4. Pools rebuilt under the search-date ceiling t_r: Tier 1 executes a
     verbatim PubMed strategy found in the text; Tier 2 unions a strict
     PICO Boolean query with the top-K_r BM25 + MedCPT neighbours of a
     broad query (K_r from PRISMA counts, bounded to [500, 5000]).
  5. Review type from deterministic keyword rules.
* **Quality filters**: harvested positives that are themselves SRs/MAs
  (except for umbrella reviews) or that post-date the review are dropped;
  benchmark eligibility requires pool size in [100, 15000] and
  `pool_recall ≥ 0.25`. A pilot on 15 reviews kept 9 (60%).

## Collections considered but not ingested

| Collection | Reason |
|---|---|
| Alharbi et al. 2019 SR-updates | distributed as a Python pickle (unsafe to load untrusted pickles); reviews overlap Cochrane sources already ingested |
| Hannousse et al. 2022 | computer-science reviews (non-biomedical) |
| CSMeD-FT | full-text screening task (out of scope); CSMeD's T/A part is the union of sources above |
| Miwa et al. 2015 | not publicly downloadable |

## De-duplication and decontamination

1. **Review level** (`build`): one copy per `review_key` (Cochrane CD
   number, DOI, PMC id, ...), keeping the source with the lowest
   `priority` in `configs/datasets.yaml`.
2. **Family level** (`split`): reviews whose included studies overlap
   (overlap coefficient ≥ 0.10, matched by title fingerprint or PMID) or
   whose objectives are near-duplicates (MedCPT cosine ≥ 0.88) form one
   family; families are split atomically.
3. **Record level** (`split`): train/val records matching any `pmid`,
   `title_fp` or `abstract_fp` of the eval, prospective or TITAN-SR test
   splits are purged, and train is also purged against val. The audit in
   `splits/audit.json` must report **0** matches or the pipeline stops.
