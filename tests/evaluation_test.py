"""Tests for IPW metrics, bootstrap, thresholds, agreement and ranking."""

import numpy as np
import pandas as pd

from opensrbench.calibration import thresholds
from opensrbench.evaluation import agreement
from opensrbench.evaluation import bootstrap
from opensrbench.evaluation import metrics
from opensrbench.evaluation import ranking


def _tier():
    # Review A: 2 positives, 4 sampled negatives out of 40 total.
    return pd.DataFrame(
        {
            "review_id": ["A"] * 6,
            "record_id": [f"x{i}" for i in range(6)],
            "family_id": ["F"] * 6,
            "label_final": [1, 1, 0, 0, 0, 0],
        }
    )


def test_ipw_workload_and_abstention():
    decisions = pd.DataFrame(
        {
            "review_id": ["A"] * 5,
            "record_id": ["x0", "x1", "x2", "x3", "x4"],
            "decision_keep": [1, 0, 1, 0, 0],
            "is_valid_answer": [True, True, True, True, False],
        }
    )  # x5 missing -> keep/invalid; x4 invalid -> keep.
    merged = metrics.apply_abstention_rule(_tier(), decisions)
    totals = pd.DataFrame(
        {"review_id": ["A"], "n_total_pos": [2], "n_total_neg": [40]}
    )
    per = metrics.score_review_metrics(merged, totals).iloc[0]
    assert per["sensitivity"] == 0.5
    assert per["specificity"] == 0.25  # Only x3 is a valid exclusion.
    # kept = 1 positive + 3 sampled negatives * (40 / 4).
    assert np.isclose(per["workload"], (1 + 3 * 10) / 42)


def test_bootstrap_ci_contains_point_and_paired_zero():
    per = pd.DataFrame(
        {
            "review_id": list("abcdef"),
            "family_id": list("aabbcd"),
            "sensitivity": [1, 0.9, 0.95, 1, 0.8, 1.0],
            "specificity": [0.5, 0.6, 0.7, 0.4, 0.5, 0.6],
            "workload": [0.3, 0.2, 0.4, 0.5, 0.3, 0.3],
        }
    )
    boot = bootstrap.FamilyBootstrap(per["family_id"].unique(), 2000, 1)
    dist = boot.distributions(per)
    point = metrics.point_estimates(per)
    _, lo, hi = bootstrap.percentile_ci(point["youdens_j"], dist["youdens_j"])
    assert lo <= point["youdens_j"] <= hi
    comp = bootstrap.paired_difference(dist["youdens_j"], dist["youdens_j"], 0)
    assert comp["delta"] == 0 and comp["p_value"] == 1.0


def test_holm_is_monotone_step_down():
    adj = bootstrap.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.isclose(adj["a"], 0.03)
    assert np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)


def test_pooled_conformal_is_exact_order_statistic():
    scores = np.arange(1, 201, dtype=float)  # n = 200 positives.
    # k = floor(201 * 0.05) = 10 -> the 10th smallest score.
    assert thresholds.pooled_conformal(scores, 0.95) == 10.0
    # alpha so small that k = 0 -> keep everything.
    assert thresholds.pooled_conformal(scores, 0.999) == float("-inf")


def test_equal_review_threshold_meets_target_on_validation():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(30):
        for i in range(40):
            label = int(i < 5)
            rows.append((f"r{r}", label, rng.normal(2.0 * label, 1.0)))
    df = pd.DataFrame(rows, columns=["review", "y", "s"])
    tau = thresholds.equal_review(
        df.s.values, df.y.values, df.review.values, 0.95
    )
    sens = df[df.y == 1].assign(k=df.s >= tau).groupby("review")["k"].mean()
    assert sens.mean() >= 0.95
    lcb = thresholds.equal_review_lcb(
        df.s.values,
        df.y.values,
        df.review.values,
        df.review.values,
        0.95,
        replicates=300,
    )
    assert lcb <= tau


def test_ranking_operating_point_includes_ties():
    point = ranking.operating_point(
        np.array([1, 0, 1, 0]), np.array([0.9, 0.5, 0.5, 0.1]), 1.0
    )
    assert point["recall"] == 1.0 and point["specificity"] == 0.5
    assert ranking.auc(np.array([1, 0]), np.array([0.2, 0.1])) == 1.0


def test_cohens_kappa():
    assert agreement.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert np.isclose(agreement.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)
