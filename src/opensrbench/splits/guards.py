"""Defence-in-depth leakage guards, re-run at every stage that consumes splits.

The split stage purges and audits once; these guards re-check the artefacts
whenever they are used (training, calibration, evaluation, `osb audit`), so
a hand-edited parquet file, a stale run directory or a new source can never
silently leak held-out reviews into training.

Checks:
  * family disjointness  every family lives in exactly one non-excluded split;
  * review-key disjointness  the same review (e.g. a Cochrane CD number)
    never appears in two splits;
  * record disjointness  no PMID / title / abstract fingerprint shared
    between trainable splits (train, val) and protected splits, nor between
    train and val.
"""

import pandas as pd

from opensrbench.splits import decontam

TRAINABLE = ("train", "val")
PROTECTED = ("test", "eval", "prospective", "titan_test")


def check_family_disjoint(splits: pd.DataFrame) -> dict[str, int]:
    """Raises LeakageError if a family spans two non-excluded splits."""
    active = splits[splits["split"] != "excluded"]
    per_family = active.groupby("family_id")["split"].nunique()
    bad = per_family[per_family > 1]
    if len(bad):
        raise decontam.LeakageError(
            f"{len(bad)} families span several splits, e.g. "
            f"{bad.index[:3].tolist()}"
        )
    return {"families_checked": int(len(per_family))}


def check_review_keys_disjoint(
    reviews: pd.DataFrame, splits: pd.DataFrame
) -> dict[str, int]:
    """Raises LeakageError if one review_key appears in two splits."""
    df = reviews[["review_id", "review_key"]].merge(
        splits[["review_id", "split"]], on="review_id"
    )
    df = df[df["split"] != "excluded"]
    per_key = df.groupby("review_key")["split"].nunique()
    bad = per_key[per_key > 1]
    if len(bad):
        raise decontam.LeakageError(
            f"{len(bad)} review keys appear in several splits, e.g. "
            f"{bad.index[:3].tolist()}"
        )
    return {"review_keys_checked": int(len(per_key))}


def check_records(
    records: pd.DataFrame,
    splits: pd.DataFrame,
    protected: tuple[str, ...] = PROTECTED,
    check_train_vs_val: bool = True,
) -> dict[str, dict]:
    """Re-audits record fingerprints across splits (raises on any match)."""
    split = records["review_id"].map(splits.set_index("review_id")["split"])
    out = {
        "trainable_vs_protected": decontam.audit_separation(
            records[split.isin(TRAINABLE)], records[split.isin(protected)]
        )
    }
    if check_train_vs_val:
        out["train_vs_val"] = decontam.audit_separation(
            records[split == "train"], records[split == "val"]
        )
    return out


def verify_all(
    reviews: pd.DataFrame,
    records: pd.DataFrame,
    splits: pd.DataFrame,
    protected: tuple[str, ...] = PROTECTED,
    check_train_vs_val: bool = True,
) -> dict:
    """Runs every guard; returns a report or raises LeakageError."""
    return {
        **check_family_disjoint(splits),
        **check_review_keys_disjoint(reviews, splits),
        "records": check_records(
            records, splits, protected, check_train_vs_val
        ),
    }


def check_training_frames(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    records: pd.DataFrame,
    splits: pd.DataFrame,
    protected: tuple[str, ...] = PROTECTED,
) -> dict:
    """Guards the exact frames a training run is about to use.

    Args:
      train: Records the model will be fitted on.
      holdout: Early-stopping records (train-split fold).
      records: All decontaminated records.
      splits: Split assignment.
      protected: Splits that must stay unseen (val is added: it is reserved
        for calibration and must not influence training either).

    Returns:
      Audit counts (all zero).
    """
    split = records["review_id"].map(splits.set_index("review_id")["split"])
    unseen = records[split.isin(tuple(protected) + ("val",))]
    used = pd.concat([train, holdout], ignore_index=True)
    in_wrong_split = (
        used["review_id"].map(splits.set_index("review_id")["split"]) != "train"
    )
    if in_wrong_split.any():
        raise decontam.LeakageError(
            f"{int(in_wrong_split.sum())} training rows come from reviews "
            "outside the train split"
        )
    return decontam.audit_separation(used, unseen)
