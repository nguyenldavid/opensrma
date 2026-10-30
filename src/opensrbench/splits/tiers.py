"""Frozen nested negative tiers T10/T20/T40 (protocol Section 3.1, 4.1).

For each evaluation review, 100% of positives are kept and negatives are
ordered by SHA-256("OPEN-SR-BENCH-V1:" || review_id || ":" || record_id); the
first min(k, N_r) negatives form tier T_k. Tiers are nested by construction
(T10 ⊂ T20 ⊂ T40). A second hash ("<record_id>:stability-v1") selects the
10% repeat-agreement subset.
"""

import hashlib

import numpy as np
import pandas as pd

from opensrbench.utils import io


def _hashes(
    prefix: str, review_ids: pd.Series, record_ids: pd.Series, suffix: str = ""
) -> np.ndarray:
    return np.array(
        [
            hashlib.sha256(f"{prefix}{r}:{i}{suffix}".encode()).hexdigest()
            for r, i in zip(review_ids, record_ids)
        ]
    )


def assign_frozen_tiers(
    records: pd.DataFrame, salt: str, tiers: tuple[int, ...] = (10, 20, 40)
) -> pd.DataFrame:
    """Adds sample_hash, neg_rank and in_tier_t{k} columns.

    Args:
      records: Records with review_id, record_id, label_final.
      salt: Benchmark salt, e.g. "OPEN-SR-BENCH-V1".
      tiers: Negative tier sizes.

    Returns:
      Copy of `records` sorted by review, label (positives first) and hash.
    """
    df = records.copy()
    df["sample_hash"] = _hashes(f"{salt}:", df["review_id"], df["record_id"])
    df = df.sort_values(
        ["review_id", "label_final", "sample_hash"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    neg = df["label_final"] == 0
    df["neg_rank"] = -1
    df.loc[neg, "neg_rank"] = df[neg].groupby("review_id").cumcount()
    for k in tiers:
        df[f"in_tier_t{k}"] = (df["label_final"] == 1) | (
            neg & (df["neg_rank"] < k)
        )
    return df


def stability_subset(
    tier_records: pd.DataFrame,
    fraction: float = 0.10,
    suffix: str = "stability-v1",
) -> pd.Series:
    """Flags max(1, round(fraction * |C_r|)) lowest-hash records per review."""
    df = tier_records[["review_id", "record_id"]].copy()
    df["h"] = [
        hashlib.sha256(f"{i}:{suffix}".encode()).hexdigest()
        for i in df["record_id"]
    ]
    df["rank"] = df.groupby("review_id")["h"].rank(method="first")
    size = df.groupby("review_id")["record_id"].transform("count")
    quota = np.maximum(1, np.round(fraction * size))
    return pd.Series((df["rank"] <= quota).values, index=tier_records.index)


def manifest(
    tier_df: pd.DataFrame, tiers: tuple[int, ...], salt: str, audit: dict
) -> dict:
    """Pre-registration manifest: counts, content hash and leakage audit."""
    out = {"salt": salt, "audit": audit, "tiers": {}}
    for k in tiers:
        sub = tier_df[tier_df[f"in_tier_t{k}"]]
        out["tiers"][f"T{k}"] = {
            "reviews": int(sub["review_id"].nunique()),
            "records": int(len(sub)),
            "positives": int(sub["label_final"].sum()),
            "sha256": io.dataframe_sha256(
                sub, ["review_id", "record_id", "label_final"]
            ),
        }
    return out
