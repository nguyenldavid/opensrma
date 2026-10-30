"""Repeat-agreement audit (protocol Section 4.1)."""

import numpy as np
import pandas as pd


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters (nan if undefined)."""
    a, b = np.asarray(a).astype(int), np.asarray(b).astype(int)
    if len(a) == 0:
        return float("nan")
    observed = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return 1.0 if observed == 1.0 else float("nan")
    return float((observed - expected) / (1 - expected))


def repeat_agreement(
    primary: pd.DataFrame, repeat: pd.DataFrame
) -> dict[str, float]:
    """Agreement between seed-0 and seed-1 decisions on the stability subset.

    Both frames need review_id, record_id, decision_keep.
    """
    merged = primary.merge(
        repeat, on=["review_id", "record_id"], suffixes=("_a", "_b")
    )
    a = merged["decision_keep_a"].to_numpy()
    b = merged["decision_keep_b"].to_numpy()
    return {
        "n": int(len(merged)),
        "percent_agreement": float((a == b).mean()) if len(a) else float("nan"),
        "cohens_kappa": cohen_kappa(a, b),
    }
