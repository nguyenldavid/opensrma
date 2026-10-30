"""Canonical review/record schema shared by every pipeline stage.

Two tables describe the whole corpus:

reviews (one row per systematic review)
  review_id            Globally unique "<source>:<local id>".
  source               Source adapter name (e.g. "clef_tar").
  review_key           De-duplication key across sources, e.g.
                       "cochrane:CD012255".
  title                Review title.
  objective            Review objective / aims.
  selection_criteria   Eligibility (inclusion + exclusion) criteria text.
  review_type          One of REVIEW_TYPES.
  pub_date             ISO publication date (may be null for legacy data).
  search_date          ISO literature-search cut-off t_r (may be null).
  pool_complete        True if negatives come from a full search result.
  pool_tier            "provided" | "tier1_exact" | "tier2_hybrid" |
                       "augmented".
  has_abs_labels       True if abstract-stage labels are known.
  domain               "biomedical" | "other".
  doi, pmid, pmcid     Review identifiers when known.
  extra                JSON string with source-specific metadata.

records (one row per candidate citation within a review)
  review_id            Foreign key into reviews.
  record_id            Unique within review: "pmid:<n>" or "<source>:<hash>".
  pmid                 PubMed ID string or null.
  title, abstract      Citation text (attached at build time if missing).
  pub_date             ISO publication date of the citation (nullable).
  label_final          1 = included in the final review (primary endpoint).
  label_abs            1 = passed title/abstract screening (auxiliary).
  retrieval_rank       Rank within the reconstructed pool (nullable).
  retrieval_source     "provided" | "tier1" | "boolean" | "bm25" | "dense" | ...
"""

import json
from typing import Any, Iterable

import pandas as pd

from opensrbench.utils import hashing

REVIEW_TYPES = (
    "intervention_ma",
    "intervention_no_ma",
    "nma",
    "dta",
    "prognostic",
    "dose_response",
    "prevalence",
    "etiology",
    "qualitative",
    "rapid_scoping_umbrella",
)

REVIEW_COLUMNS = (
    "review_id",
    "source",
    "review_key",
    "title",
    "objective",
    "selection_criteria",
    "review_type",
    "pub_date",
    "search_date",
    "pool_complete",
    "pool_tier",
    "has_abs_labels",
    "domain",
    "doi",
    "pmid",
    "pmcid",
    "extra",
)  # yapf: disable

RECORD_COLUMNS = (
    "review_id",
    "record_id",
    "pmid",
    "title",
    "abstract",
    "pub_date",
    "label_final",
    "label_abs",
    "retrieval_rank",
    "retrieval_source",
)  # yapf: disable

SPLITS = ("train", "val", "eval", "prospective", "titan_test", "excluded")


def pmid_record_id(pmid: object) -> str:
    """Returns the canonical record id for a PubMed citation."""
    return f"pmid:{normalize_pmid(pmid)}"


def text_record_id(source: str, title: object, abstract: object) -> str:
    """Stable record id for citations that have no PMID."""
    payload = (
        hashing.normalize_for_fingerprint(title)
        + "|"
        + hashing.normalize_for_fingerprint(abstract)
    )
    return f"{source}:{hashing.short_id(payload)}"


def normalize_pmid(value: object) -> str | None:
    """Extracts a bare numeric PMID from ints, floats, strings or URLs."""
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:  # NaN.
            return None
        value = int(value)
    text = str(value).strip().rstrip("/")
    if not text:
        return None
    text = text.rsplit("/", 1)[-1]
    if text.lower().startswith("pmid:"):
        text = text[5:]
    text = text.strip()
    return text if text.isdigit() and int(text) > 0 else None


def make_reviews_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Builds a reviews dataframe with all canonical columns present."""
    df = pd.DataFrame(list(rows))
    defaults = {
        "pool_complete": True,
        "pool_tier": "provided",
        "has_abs_labels": False,
        "domain": "biomedical",
        "review_type": None,
    }
    for col in REVIEW_COLUMNS:
        if col not in df.columns:
            df[col] = defaults.get(col)
    df["extra"] = df["extra"].apply(
        lambda v: v if isinstance(v, str) else json.dumps(v or {})
    )
    for col in ("title", "objective", "selection_criteria"):
        df[col] = df[col].fillna("").astype(str)
    df["pool_complete"] = df["pool_complete"].fillna(True).astype(bool)
    df["has_abs_labels"] = df["has_abs_labels"].fillna(False).astype(bool)
    return df[list(REVIEW_COLUMNS)]


def make_records_frame(
    rows: Iterable[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Builds a records dataframe with canonical columns and dtypes.

    Missing `label_abs` defaults to `label_final` (a final inclusion always
    passed abstract screening). Rows are de-duplicated on
    (review_id, record_id), keeping the highest labels.
    """
    df = (
        rows.copy()
        if isinstance(rows, pd.DataFrame)
        else pd.DataFrame(list(rows))
    )
    for col in RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["pmid"] = df["pmid"].map(normalize_pmid)
    df["label_final"] = df["label_final"].fillna(0).astype(int)
    df["label_abs"] = df["label_abs"].fillna(df["label_final"]).astype(int)
    df["label_abs"] = df[["label_abs", "label_final"]].max(axis=1)
    df["retrieval_source"] = df["retrieval_source"].fillna("provided")
    df["retrieval_rank"] = pd.to_numeric(df["retrieval_rank"], errors="coerce")
    df = df.sort_values(["label_final", "label_abs"], ascending=False)
    df = df.drop_duplicates(["review_id", "record_id"], keep="first")
    return df[list(RECORD_COLUMNS)].reset_index(drop=True)


def validate(reviews: pd.DataFrame, records: pd.DataFrame) -> None:
    """Raises ValueError if the two tables violate schema invariants."""
    missing = set(REVIEW_COLUMNS) - set(reviews.columns)
    if missing:
        raise ValueError(f"reviews missing columns: {sorted(missing)}")
    missing = set(RECORD_COLUMNS) - set(records.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")
    if reviews["review_id"].duplicated().any():
        dupes = reviews.loc[reviews["review_id"].duplicated(), "review_id"]
        raise ValueError(f"duplicate review_id: {dupes.head().tolist()}")
    if records.duplicated(["review_id", "record_id"]).any():
        raise ValueError("duplicate (review_id, record_id) pairs")
    orphans = set(records["review_id"]) - set(reviews["review_id"])
    if orphans:
        raise ValueError(f"{len(orphans)} records reference unknown reviews")
    if not records["label_final"].isin([0, 1]).all():
        raise ValueError("label_final must be 0/1")
    if (records["label_abs"] < records["label_final"]).any():
        raise ValueError("label_abs must be >= label_final")
