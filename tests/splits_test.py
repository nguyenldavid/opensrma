"""Tests for families, split assignment, decontamination and tiers."""

import pandas as pd
import pytest

from opensrbench.splits import assign
from opensrbench.splits import decontam
from opensrbench.splits import families
from opensrbench.splits import tiers


def _records(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "review_id",
            "record_id",
            "pmid",
            "title",
            "abstract",
            "label_final",
        ],
    )
    return decontam.add_fingerprints(df)


def test_union_find_components():
    forest = families.UnionFind(["a", "b", "c", "d"])
    forest.union("a", "b")
    forest.union("c", "b")
    groups = sorted(sorted(g) for g in forest.components().values())
    assert groups == [["a", "b", "c"], ["d"]]


def test_overlap_edges_use_min_denominator():
    keys = {
        "u": {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"},
        "v": {"1", "x"},
        "w": {"y", "z"},
    }
    assert families.overlap_edges(keys, 0.10) == [("u", "v")]
    assert families.overlap_edges(keys, 0.6) == []


def test_build_families_links_by_title_fingerprint_across_sources():
    reviews = pd.DataFrame(
        {
            "review_id": ["a", "b", "c"],
            "title": ["x", "y", "z"],
            "objective": ["", "", ""],
        }
    )
    records = _records(
        [
            ("a", "pmid:1", "1", "Shared Trial", "", 1),
            ("b", "chan:1", None, "shared trial.", "", 1),
            ("c", "pmid:2", "2", "Other", "", 1),
        ]
    )
    fams, stats = families.build_families(reviews, records, 0.1, 0.99, None)
    fam = fams.set_index("review_id")["family_id"]
    assert fam["a"] == fam["b"] != fam["c"]
    assert stats["families"] == 2


def test_purge_removes_all_fingerprint_matches_and_audit_is_zero():
    records = _records(
        [
            ("tr", "pmid:1", "1", "same pmid", "a1", 0),
            ("tr", "pmid:2", "2", "Shared Title", "a2", 0),
            ("tr", "pmid:3", "3", "t3", "Shared abstract", 1),
            ("tr", "pmid:4", "4", "unique", "unique", 1),
            ("ev", "pmid:1", "1", "different", "b1", 1),
            ("ev", "pmid:5", "5", "shared title!", "b2", 0),
            ("ev", "pmid:6", "6", "t6", "shared ABSTRACT", 0),
        ]
    )
    split_of = pd.Series({"tr": "train", "ev": "eval"})
    clean, audit = decontam.purge(records, split_of, ("eval",))
    assert clean.loc[clean["review_id"] == "tr", "record_id"].tolist() == [
        "pmid:4"
    ]
    assert audit["purged_vs_protected"] == 3
    assert sum(audit["final_train_vs_protected"].values()) == 0


def test_audit_raises_on_leak():
    a = _records([("x", "pmid:1", "1", "t", "a", 0)])
    with pytest.raises(decontam.LeakageError):
        decontam.audit_separation(a, a)


def test_assign_splits_is_atomic_and_prioritised():
    reviews = pd.DataFrame(
        {
            "review_id": ["old1", "new1", "old2", "fut", "titan"],
            "review_key": ["k1", "k2", "k3", "k4", "tk"],
            "pub_date": [
                "2019-01-01",
                "2025-06-01",
                "2018-01-01",
                "2027-01-01",
                "2020-01-01",
            ],
            "review_type": ["dta"] * 5,
        }
    )
    fams = pd.DataFrame(
        {
            "review_id": reviews["review_id"],
            "family_id": ["f1", "f1", "f2", "f3", "f4"],
        }
    )
    cfg = {
        "eval_window_start": "2025-01-01",
        "eval_window_end": "2026-10-31",
        "freeze_date": "2026-10-31",
        "temporal_ratios": {"train": 0.0, "val": 0.0, "eval": 1.0},
        "legacy_ratios": {"train": 1.0},
        "titan_sr_test_as_benchmark": True,
    }
    out = assign.assign_splits(reviews, fams, cfg, "salt", {"tk"}).set_index(
        "review_id"
    )["split"]
    assert out["new1"] == "eval"
    assert out["old1"] == "excluded"  # Old member of an eval family.
    assert out["old2"] == "train"
    assert out["fut"] == "prospective"
    assert out["titan"] == "titan_test"


def test_frozen_tiers_nested_deterministic_and_keep_all_positives():
    rows = [("r1", f"n{i}", 0) for i in range(50)] + [
        ("r1", "p1", 1),
        ("r1", "p2", 1),
    ]
    rows += [("r2", f"m{i}", 0) for i in range(5)] + [("r2", "q", 1)]
    df = pd.DataFrame(rows, columns=["review_id", "record_id", "label_final"])
    a = tiers.assign_frozen_tiers(df, "SALT")
    b = tiers.assign_frozen_tiers(df.sample(frac=1, random_state=3), "SALT")
    pd.testing.assert_frame_equal(a, b)
    for k in (10, 20, 40):
        sub = a[a[f"in_tier_t{k}"]]
        assert sub["label_final"].sum() == 3
        assert (sub[sub.review_id == "r1"].label_final == 0).sum() == k
        assert (sub[sub.review_id == "r2"].label_final == 0).sum() == 5
    assert set(a[a.in_tier_t10].record_id) <= set(a[a.in_tier_t20].record_id)
    flags = tiers.stability_subset(a[a.in_tier_t20])
    per = a[a.in_tier_t20].assign(f=flags).groupby("review_id")["f"].sum()
    assert per["r1"] == 2 and per["r2"] == 1
