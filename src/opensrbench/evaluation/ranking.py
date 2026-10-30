"""Threshold-free, per-review ranking metrics (TITAN-SR comparable).

These are *oracle* per-review operating points (the threshold is chosen per
review from its own labels), which is what TITAN-SR reports as
"specificity at 99% recall" and WSS@95. They measure ranking quality; the
deployable, validation-locked operating points live in `metrics.py`.

Ties at the cut-off score are always included (conservative).
"""

import numpy as np
import pandas as pd


def operating_point(
    labels: np.ndarray, scores: np.ndarray, target_recall: float
) -> dict[str, float] | None:
    """Specificity / work saved at the smallest threshold reaching recall.

    Args:
      labels: 0/1 array.
      scores: Scores (higher = keep).
      target_recall: e.g. 0.99.

    Returns:
      Dict with threshold, recall, specificity, work_saved and wss; or None
      when the review has no positives or no negatives.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    pos_scores = np.sort(scores[labels == 1])[::-1]
    needed = int(np.ceil(target_recall * n_pos - 1e-12))
    threshold = pos_scores[max(needed, 1) - 1]
    keep = scores >= threshold
    recall = keep[labels == 1].mean()
    spec = 1.0 - keep[labels == 0].mean()
    work_saved = 1.0 - keep.mean()
    return {
        "threshold": float(threshold),
        "recall": float(recall),
        "specificity": float(spec),
        "work_saved": float(work_saved),
        "wss": float(work_saved - (1.0 - target_recall)),
    }


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC AUC with average ranks for ties (Mann-Whitney U)."""
    labels = np.asarray(labels).astype(int)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float(
        (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def per_review_ranking(
    df: pd.DataFrame, score_col: str = "score", label_col: str = "label_final"
) -> pd.DataFrame:
    """Per-review AUC, spec@95/99/99.5 recall and WSS@95."""
    rows = []
    for review_id, group in df.groupby("review_id"):
        y = group[label_col].to_numpy()
        s = group[score_col].to_numpy()
        row = {"review_id": review_id, "auc": auc(y, s)}
        for target, tag in ((0.95, "95"), (0.99, "99"), (0.995, "995")):
            point = operating_point(y, s, target)
            row[f"spec_at_{tag}_recall"] = (
                point["specificity"] if point else None
            )
            if tag == "95":
                row["wss_at_95"] = point["wss"] if point else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_ranking(
    df: pd.DataFrame, score_col: str = "score"
) -> dict[str, float]:
    """Equal-review means of the per-review ranking metrics."""
    per = per_review_ranking(df, score_col)
    return {
        c: float(per[c].dropna().mean())
        for c in per.columns
        if c != "review_id" and per[c].notna().any()
    }
