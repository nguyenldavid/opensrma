"""Automatic, leak-free training corpus for the RCT classifier.

Labels come from MEDLINE publication types assigned by NLM indexers
("Randomized Controlled Trial"[pt]); no manual labelling. For every year
the corpus samples RCTs plus four negative strata that make the task hard:
non-randomised trials, observational studies, reviews, and other human
studies. Sampling is spread over months so no period dominates.

Decontamination: before training, every corpus record matching a PMID,
title or abstract fingerprint of the validation or any protected split
(test, eval, prospective, titan_test) is dropped, so the classifier can be
used as a stacking feature without touching held-out data.

Optionally a Cochrane Crowd export (CSV with title, abstract, label) can be
added via `rct.crowd_csv`.
"""

import logging

import pandas as pd
import tqdm

from opensrbench.pubmed import eutils
from opensrbench.pubmed import store as store_lib
from opensrbench.splits import decontam
from opensrbench.utils import hashing

_BASE = "medline[sb] AND hasabstract"
_NOT_RCT = 'NOT "randomized controlled trial"[pt]'
STRATA = {
    "rct": '"randomized controlled trial"[pt]',
    "trial_nonrandomized": (
        '("clinical trial"[pt] OR "controlled clinical '
        f'trial"[pt]) {_NOT_RCT}'
    ),
    "observational": (
        '("observational study"[pt] OR "cohort studies"[mh] OR '
        '"case-control studies"[mh] OR "cross-sectional '
        f'studies"[mh]) {_NOT_RCT}'
    ),
    "review": (
        '("review"[pt] OR "systematic review"[pt] OR '
        f'"meta-analysis"[pt]) {_NOT_RCT}'
    ),
    "other": f'humans[mh] {_NOT_RCT} NOT "clinical trial"[pt]',
}


def fetch_corpus(
    client: eutils.EUtilsClient,
    store: store_lib.PubMedStore,
    years: tuple[int, int],
    per_year_rct: int,
    per_year_negative: int,
    batch_size: int = 200,
) -> pd.DataFrame:
    """Samples PMIDs per stratum/month and caches their records.

    Returns:
      Frame pmid, stratum (labels are derived from the fetched records).
    """
    rows = []
    for year in tqdm.tqdm(range(years[0], years[1] + 1), desc="rct corpus"):
        for stratum, query in STRATA.items():
            n_year = per_year_rct if stratum == "rct" else per_year_negative
            per_month = max(1, n_year // 12)
            for month in range(1, 13):
                last = (
                    31
                    if month in (1, 3, 5, 7, 8, 10, 12)
                    else (29 if month == 2 else 30)
                )
                term = (
                    f'({query}) AND {_BASE} AND ("{year}/{month:02d}/01"'
                    f'[dp] : "{year}/{month:02d}/{last}"[dp])'
                )
                _, pmids = client.esearch(term, per_month)
                rows.extend({"pmid": p, "stratum": stratum} for p in pmids)
    frame = pd.DataFrame(rows).drop_duplicates("pmid")
    store.fetch_missing(client, frame["pmid"], batch_size)
    return frame


def attach_labels(
    frame: pd.DataFrame, store: store_lib.PubMedStore
) -> pd.DataFrame:
    """Adds title/abstract/label from cached PubMed records."""
    texts = store.frame(frame["pmid"])
    out = frame.merge(texts, on="pmid", how="inner")
    out["label"] = out["publication_types"].map(
        lambda types: int(
            any(
                str(t).lower() == "randomized controlled trial"
                for t in (types if types is not None else [])
            )
        )
    )
    out = out[
        (out["title"].fillna("") != "") & (out["abstract"].fillna("") != "")
    ]
    return out[["pmid", "title", "abstract", "label", "stratum"]]


def decontaminate(
    corpus: pd.DataFrame,
    records: pd.DataFrame,
    split_of: pd.Series,
    protected: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """Drops corpus records matching fingerprints of protected/val reviews."""
    corpus = decontam.add_fingerprints(
        corpus.assign(pmid=corpus["pmid"].astype(str))
    )
    split = records["review_id"].map(split_of)
    held = records[split.isin(tuple(protected) + ("val",))]
    keys = {
        c: set(held[c].dropna().astype(str)) - {""}
        for c in decontam.FINGERPRINT_COLUMNS
    }
    hit = pd.Series(False, index=corpus.index)
    for col in decontam.FINGERPRINT_COLUMNS:
        hit |= corpus[col].astype(str).isin(keys[col]) & (
            corpus[col].astype(str) != ""
        )
    clean = corpus[~hit].reset_index(drop=True)
    audit = {
        "dropped": int(hit.sum()),
        "kept": int(len(clean)),
        "final": decontam.audit_separation(clean, held),
    }
    logging.info("RCT corpus decontamination: %s", audit)
    return clean, audit


def split_corpus(
    corpus: pd.DataFrame, salt: str = "rct-v1"
) -> dict[str, pd.DataFrame]:
    """Deterministic 80/10/10 train/calibration/test split by PMID hash."""
    u = (
        corpus["pmid"]
        .astype(str)
        .map(lambda p: hashing.stable_unit(f"{salt}:{p}"))
    )
    return {
        "train": corpus[u < 0.8],
        "calib": corpus[(u >= 0.8) & (u < 0.9)],
        "test": corpus[u >= 0.9],
    }
