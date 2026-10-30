"""Atomic family-to-split assignment (protocol Section 2.3.2).

Priority per family (first match wins):
  1. prospective  any member published after the benchmark freeze;
  2. titan_test   any member is in TITAN-SR's published test split (so the
                  head-to-head on their benchmark is leakage-free);
  3. temporal     any member published inside the eval window -> train /
                  val / eval by `temporal_ratios`;
  4. legacy       all members pre-window -> `legacy_ratios`.

Assignment is deterministic: families are ordered by SHA-256 of
(salt, family_id) within each review-type stratum and cut at the ratio
boundaries. Members of a held-out family that are not themselves eligible
for that split (e.g. an old review in an eval family) become "excluded":
they are neither trained on nor evaluated.
"""

import numpy as np
import pandas as pd

from opensrbench.utils import dates
from opensrbench.utils import hashing


def _ratio_split(
    families: pd.DataFrame, ratios: dict[str, float], salt: str
) -> dict[str, str]:
    """Stratified, hash-ordered split of families by their primary type."""
    out = {}
    names = [n for n in ("train", "val", "eval") if ratios.get(n, 0) > 0]
    bounds = np.cumsum([ratios[n] for n in names])
    bounds = bounds / bounds[-1]
    for stratum, group in families.groupby("primary_type"):
        order = sorted(
            group["family_id"], key=lambda f: hashing.sha256_hex(f"{salt}:{f}")
        )
        n = len(order)
        # Systematic sampling with a hashed random start: exact proportions
        # in large strata, unbiased (not always "train") in tiny ones.
        offset = hashing.stable_unit(f"{salt}:offset:{stratum}")
        for i, family_id in enumerate(order):
            position = (i + offset) / n
            out[family_id] = names[
                min(
                    int(np.searchsorted(bounds, position, side="right")),
                    len(names) - 1,
                )
            ]
    return out


def assign_splits(
    reviews: pd.DataFrame,
    families: pd.DataFrame,
    cfg: dict,
    salt: str,
    titan_test_keys: set[str] | None = None,
) -> pd.DataFrame:
    """Assigns every review to a split.

    Args:
      reviews: Processed reviews (review_id, review_key, pub_date,
        review_type, eligible_eval).
      families: review_id -> family_id.
      cfg: Merged `temporal` + `splits` config sections.
      salt: Benchmark salt.
      titan_test_keys: review_keys of TITAN-SR test/external reviews.

    Returns:
      Frame review_id, family_id, split, reason.
    """
    df = reviews.merge(families[["review_id", "family_id"]], on="review_id")
    start = dates.parse_date(cfg["eval_window_start"])
    end = dates.parse_date(cfg["eval_window_end"])
    freeze = dates.parse_date(cfg["freeze_date"])
    pub = df["pub_date"].map(
        lambda d: dates.parse_date(d) if isinstance(d, str) else None
    )
    df["in_window"] = pub.map(lambda d: bool(d and start <= d <= end))
    df["prospective"] = pub.map(lambda d: bool(d and d > freeze))
    titan_keys = titan_test_keys or set()
    df["titan_test"] = df["review_key"].isin(titan_keys) & bool(
        cfg.get("titan_sr_test_as_benchmark", True)
    )

    fam = (
        df.groupby("family_id")
        .agg(
            any_prospective=("prospective", "any"),
            any_titan=("titan_test", "any"),
            any_window=("in_window", "any"),
            primary_type=(
                "review_type",
                lambda s: s.mode().iat[0] if not s.mode().empty else "unknown",
            ),
        )
        .reset_index()
    )
    fam["kind"] = np.select(
        [fam["any_prospective"], fam["any_titan"], fam["any_window"]],
        ["prospective", "titan_test", "temporal"],
        default="legacy",
    )
    family_split = dict(zip(fam["family_id"], fam["kind"]))
    family_split.update(
        _ratio_split(
            fam[fam["kind"] == "temporal"], cfg["temporal_ratios"], salt
        )
    )
    family_split.update(
        _ratio_split(fam[fam["kind"] == "legacy"], cfg["legacy_ratios"], salt)
    )
    df["split"] = df["family_id"].map(family_split)
    df["reason"] = df["family_id"].map(dict(zip(fam["family_id"], fam["kind"])))
    # Members that cannot be evaluated in their held-out family are excluded.
    held = {
        "prospective": "prospective",
        "titan_test": "titan_test",
        "eval": "in_window",
    }
    for split, flag in held.items():
        mask = (df["split"] == split) & ~df[flag]
        df.loc[mask, "split"] = "excluded"
        df.loc[mask, "reason"] = f"member_of_{split}_family"
    return df[["review_id", "family_id", "split", "reason"]]


def select_benchmark(
    reviews: pd.DataFrame, splits: pd.DataFrame, per_type: int, salt: str
) -> pd.Series:
    """Selects up to `per_type` eval families per review type.

    Only reviews flagged `eligible_eval` (>= 5 positives with abstracts and a
    pool within the protocol bounds) are candidates.

    Returns:
      Boolean Series indexed by review_id: True if in the frozen benchmark.
    """
    df = reviews.merge(splits, on="review_id")
    df = df[(df["split"] == "eval") & df["eligible_eval"]]
    chosen: set[str] = set()
    for _, group in df.groupby("review_type"):
        fams = sorted(
            group["family_id"].unique(),
            key=lambda f: hashing.sha256_hex(f"{salt}:bench:{f}"),
        )
        chosen.update(fams[:per_type])
    flag = df["family_id"].isin(chosen)
    return pd.Series(flag.values, index=df["review_id"]).reindex(
        reviews["review_id"], fill_value=False
    )
