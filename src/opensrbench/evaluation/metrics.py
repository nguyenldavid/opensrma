"""IPW equal-review estimators (protocol Section 3.2).

For review r on tier T_k with P_r positives (all kept in the sample),
n_r = min(k, N_r) sampled negatives out of N_r and weight w_r = N_r / n_r:

  Sens_r = mean over positives of 1[keep]
  Spec_r = mean over sampled negatives of 1[exclude and answer valid]
  Work_r = (kept positives + w_r * kept sampled negatives) / (P_r + N_r)

and the primary endpoint is equal-review Youden's
J = mean_{R+} Sens_r + mean_R Spec_r - 1.

Abstention rule: a missing decision, timeout, malformed output, refusal or
UNCERTAIN is scored as keep (decision_keep = 1) and invalid.
"""

import numpy as np
import pandas as pd

REQUIRED = (
    "review_id",
    "family_id",
    "label_final",
    "decision_keep",
    "is_valid_answer",
)


def apply_abstention_rule(
    tier: pd.DataFrame, decisions: pd.DataFrame
) -> pd.DataFrame:
    """Left-joins decisions onto tier records; missing -> keep & invalid."""
    cols = ["review_id", "record_id", "decision_keep", "is_valid_answer"]
    extra = [c for c in ("cost_usd", "score") if c in decisions.columns]
    merged = tier.merge(
        decisions[cols + extra].drop_duplicates(["review_id", "record_id"]),
        on=["review_id", "record_id"],
        how="left",
    )
    missing = merged["decision_keep"].isna()
    merged.loc[missing, "decision_keep"] = 1
    merged.loc[missing, "is_valid_answer"] = False
    merged["decision_keep"] = merged["decision_keep"].astype(int)
    merged["is_valid_answer"] = merged["is_valid_answer"].astype(bool)
    invalid = ~merged["is_valid_answer"]
    merged.loc[invalid, "decision_keep"] = 1
    return merged


def score_review_metrics(
    df_tier: pd.DataFrame, review_totals: pd.DataFrame
) -> pd.DataFrame:
    """Per-review Sens / Spec / IPW workload.

    Args:
      df_tier: Tier records with REQUIRED columns.
      review_totals: review_id, n_total_pos, n_total_neg (whole cohort).

    Returns:
      Frame review_id, family_id, sensitivity, specificity, workload,
      n_pos, n_neg_sampled, invalid_rate.
    """
    missing = set(REQUIRED) - set(df_tier.columns)
    if missing:
        raise ValueError(f"df_tier missing {sorted(missing)}")
    df = df_tier.merge(
        review_totals[["review_id", "n_total_pos", "n_total_neg"]],
        on="review_id",
    )
    pos = df["label_final"] == 1
    keep = df["decision_keep"] == 1
    valid = df["is_valid_answer"].astype(bool)
    df = df.assign(
        pos=pos,
        neg=~pos,
        kept_pos=keep & pos,
        kept_neg=keep & ~pos,
        valid_excl_neg=~keep & valid & ~pos,
        invalid=~valid,
    )
    out = df.groupby("review_id").agg(
        family_id=("family_id", "first"),
        n_pos=("pos", "sum"),
        n_neg_sampled=("neg", "sum"),
        kept_pos=("kept_pos", "sum"),
        kept_neg=("kept_neg", "sum"),
        valid_excl_neg=("valid_excl_neg", "sum"),
        invalid_rate=("invalid", "mean"),
        N_pos=("n_total_pos", "first"),
        N_neg=("n_total_neg", "first"),
    )
    out["sensitivity"] = np.where(
        out["n_pos"] > 0, out["kept_pos"] / out["n_pos"].clip(lower=1), np.nan
    )
    out["specificity"] = np.where(
        out["n_neg_sampled"] > 0,
        out["valid_excl_neg"] / out["n_neg_sampled"].clip(lower=1),
        np.nan,
    )
    weight = np.where(
        out["n_neg_sampled"] > 0,
        out["N_neg"] / out["n_neg_sampled"].clip(lower=1),
        0.0,
    )
    out["workload"] = (out["kept_pos"] + weight * out["kept_neg"]) / (
        out["N_pos"] + out["N_neg"]
    )
    return out.reset_index()[
        [
            "review_id",
            "family_id",
            "sensitivity",
            "specificity",
            "workload",
            "n_pos",
            "n_neg_sampled",
            "invalid_rate",
        ]
    ]


def point_estimates(review_metrics: pd.DataFrame) -> dict[str, float]:
    sens = review_metrics["sensitivity"].dropna().mean()
    spec = review_metrics["specificity"].dropna().mean()
    work = review_metrics["workload"].dropna().mean()
    return {
        "youdens_j": float(sens + spec - 1.0),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "workload": float(work),
        "work_saved": float(1.0 - work),
    }
