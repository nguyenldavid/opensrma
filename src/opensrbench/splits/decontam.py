"""Triple-fingerprint decontamination and the zero-leakage audit (2.3.3).

Every record carries `pmid`, `title_fp` = SHA256(norm(title)) and
`abstract_fp` = SHA256(norm(abstract)[:500]). Train/validation records that
match *any* fingerprint of a protected split (eval, prospective, TITAN-SR
test) are purged before training, and the audit must report exactly zero
intersections or the pipeline stops.
"""

import pandas as pd

from opensrbench.utils import hashing

FINGERPRINT_COLUMNS = ("pmid", "title_fp", "abstract_fp")


class LeakageError(RuntimeError):
    """Raised when the decontamination audit finds any overlap."""


def add_fingerprints(records: pd.DataFrame) -> pd.DataFrame:
    """Adds title_fp / abstract_fp columns (abstract window = 500 chars)."""
    out = records.copy()
    out["title_fp"] = out["title"].map(hashing.text_fingerprint)
    out["abstract_fp"] = out["abstract"].map(
        lambda a: hashing.text_fingerprint(a, max_chars=500)
    )
    return out


def _keys(df: pd.DataFrame) -> dict[str, set[str]]:
    out = {}
    for col in FINGERPRINT_COLUMNS:
        values = df[col].dropna().astype(str)
        out[col] = set(values[values != ""])
    return out


def _matches(df: pd.DataFrame, keys: dict[str, set[str]]) -> pd.Series:
    hit = pd.Series(False, index=df.index)
    for col in FINGERPRINT_COLUMNS:
        hit |= (
            df[col].astype(str).isin(keys[col])
            & df[col].notna()
            & (df[col].astype(str) != "")
        )
    return hit


def audit_separation(
    train: pd.DataFrame, held_out: pd.DataFrame, raise_on_leak: bool = True
) -> dict[str, int]:
    """Counts fingerprint intersections between two record sets.

    Args:
      train: Records used for training / calibration.
      held_out: Protected evaluation records.
      raise_on_leak: Raise LeakageError when any count is non-zero.

    Returns:
      Dict of intersection counts per fingerprint.

    Raises:
      LeakageError: On any overlap when `raise_on_leak`.
    """
    a, b = _keys(train), _keys(held_out)
    audit = {
        f"{col}_matches": len(a[col] & b[col]) for col in FINGERPRINT_COLUMNS
    }
    if raise_on_leak and sum(audit.values()):
        raise LeakageError(f"FATAL LEAKAGE DETECTED: {audit}")
    return audit


def purge(
    records: pd.DataFrame,
    split_of: pd.Series,
    protected: tuple[str, ...],
    purge_train_against_val: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Removes train/val records matching protected-split fingerprints.

    Args:
      records: Records with fingerprint columns and review_id.
      split_of: review_id -> split.
      protected: Splits whose fingerprints must never be trained on.
      purge_train_against_val: Also purge train records that match val, so
        calibration data is unseen by every ensemble member.

    Returns:
      (purged records, audit dict). Records of protected splits are kept.
    """
    split = records["review_id"].map(split_of)
    trainable = split.isin(["train", "val"])
    protected_keys = _keys(records[split.isin(protected)])
    leak = trainable & _matches(records, protected_keys)
    audit = {
        "purged_vs_protected": int(leak.sum()),
        "purged_positives_vs_protected": int(
            (leak & (records["label_final"] == 1)).sum()
        ),
    }
    keep = ~leak
    if purge_train_against_val:
        val_keys = _keys(records[keep & (split == "val")])
        leak_val = keep & (split == "train") & _matches(records, val_keys)
        audit["purged_train_vs_val"] = int(leak_val.sum())
        keep &= ~leak_val
    out = records[keep].copy()
    out_split = out["review_id"].map(split_of)
    held = out[out_split.isin(protected)]
    audit["final_train_vs_protected"] = audit_separation(
        out[out_split.isin(["train", "val"])], held
    )
    if purge_train_against_val:
        audit["final_train_vs_val"] = audit_separation(
            out[out_split == "train"], out[out_split == "val"]
        )
    return out, audit
