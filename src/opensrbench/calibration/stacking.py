"""Review-conditional stacking calibrator (protocol Section 5.4, extended).

Raw logits from different reviews live on different scales: a "hard" review
whose positives score 0.3 and an "easy" one whose negatives score 0.4
cannot share one threshold. The calibrator therefore adds, per backbone,
within-review standardised scores and rank percentiles computed over the
*whole* candidate pool (unlabelled information, available at deployment),
fold-disagreement and cross-backbone rank dispersion, and the cohort size.

A class-weighted logistic regression is cross-fitted on validation
families: thresholds are locked on out-of-fold calibrated scores (so they
are not optimistically biased by the stacker having seen those labels), and
the final stacker is refit on all of validation.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import pipeline as sk_pipeline
from sklearn import preprocessing

KEYS = ["set", "review_id", "record_id"]


def collect_predictions(runs_root: str, members: list[str]) -> pd.DataFrame:
    """Averages fold predictions per backbone into one wide frame.

    Returns:
      Frame keyed by (set, review_id, record_id) with label columns and,
      per backbone b, b__final, b__abs, b__fold_std.
    """
    wide = None
    for backbone in members:
        files = sorted(
            glob.glob(
                os.path.join(runs_root, backbone, "fold*", "preds.parquet")
            )
        )
        if not files:
            raise FileNotFoundError(
                f"no predictions for {backbone} under "
                f"{runs_root}; run `osb predict` first."
            )
        preds = pd.concat(
            [pd.read_parquet(f) for f in files], ignore_index=True
        )
        agg = (
            preds.groupby(KEYS)
            .agg(
                label_final=("label_final", "first"),
                label_abs=("label_abs", "first"),
                **{
                    f"{backbone}__final": ("logit_final", "mean"),
                    f"{backbone}__abs": ("logit_abs", "mean"),
                    f"{backbone}__fold_std": ("logit_final", "std"),
                },
            )
            .reset_index()
        )
        agg[f"{backbone}__fold_std"] = agg[f"{backbone}__fold_std"].fillna(0.0)
        logging.info(
            "%s: %d folds, %s rows", backbone, len(files), f"{len(agg):,}"
        )
        if wide is None:
            wide = agg
        else:
            wide = wide.merge(
                agg.drop(columns=["label_final", "label_abs"]),
                on=KEYS,
                how="inner",
            )
    return wide


def build_features(
    df: pd.DataFrame, members: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Adds within-review features; returns (frame, feature columns)."""
    out = df.copy()
    group = out.groupby(["set", "review_id"])
    cols = []
    pct_cols = []
    for backbone in members:
        for head in ("final", "abs"):
            col = f"{backbone}__{head}"
            g = group[col]
            out[f"{col}_z"] = (out[col] - g.transform("mean")) / (
                g.transform("std").fillna(0.0) + 1e-6
            )
            out[f"{col}_pct"] = g.rank(pct=True)
            cols += [col, f"{col}_z", f"{col}_pct"]
            if head == "final":
                pct_cols.append(f"{col}_pct")
        cols.append(f"{backbone}__fold_std")
    out["ens_mean_z"] = out[[f"{b}__final_z" for b in members]].mean(axis=1)
    out["ens_rank_std"] = (
        out[pct_cols].std(axis=1).fillna(0.0) if len(pct_cols) > 1 else 0.0
    )
    out["log_cohort_size"] = np.log1p(group["record_id"].transform("count"))
    cols += ["ens_mean_z", "ens_rank_std", "log_cohort_size"]
    return out, cols


def _make_model(c: float, positive_weight: float):
    return sk_pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(
            C=c, class_weight={0: 1.0, 1: positive_weight}, max_iter=2000
        ),
    )


def fit_cross_fitted(
    val: pd.DataFrame,
    feature_cols: list[str],
    family_ids: np.ndarray,
    folds: int,
    c: float,
    positive_weight: float,
):
    """Returns (OOF calibrated val scores, model refit on all val)."""
    unique = np.unique(family_ids)
    rng = np.random.default_rng(0)
    assignment = dict(
        zip(rng.permutation(unique), np.arange(len(unique)) % max(2, folds))
    )
    fold = np.array([assignment[f] for f in family_ids])
    x = val[feature_cols].to_numpy(np.float64)
    y = val["label_final"].to_numpy()
    oof = np.zeros(len(val))
    for k in np.unique(fold):
        train = fold != k
        if len(np.unique(y[train])) < 2:
            oof[~train] = 0.5
            continue
        model = _make_model(c, positive_weight).fit(x[train], y[train])
        oof[~train] = model.predict_proba(x[~train])[:, 1]
    final = _make_model(c, positive_weight).fit(x, y)
    return oof, final


def mean_sigmoid(
    df: pd.DataFrame, members: list[str], weights: list[float] | None = None
) -> np.ndarray:
    """TITAN-SR combiner: weighted mean of member sigmoids."""
    weights = weights or [1.0 / len(members)] * len(members)
    total = np.zeros(len(df))
    for backbone, w in zip(members, weights):
        total += (
            w
            / sum(weights)
            * (1.0 / (1.0 + np.exp(-df[f"{backbone}__final"].to_numpy())))
        )
    return total
