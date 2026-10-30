"""Tests for the internal test split, leakage guards and the ledger."""

import pandas as pd
import pytest

from opensrbench.evaluation import ledger
from opensrbench.splits import assign
from opensrbench.splits import decontam
from opensrbench.splits import guards


def _splits(rows):
    return pd.DataFrame(rows, columns=["review_id", "family_id", "split"])


def test_legacy_families_get_a_review_level_test_split():
    n = 400
    reviews = pd.DataFrame(
        {
            "review_id": [f"r{i}" for i in range(n)],
            "review_key": [f"k{i}" for i in range(n)],
            "pub_date": ["2019-05-01"] * n,
            "review_type": ["dta", "intervention_ma"] * (n // 2),
        }
    )
    fams = pd.DataFrame(
        {
            "review_id": reviews["review_id"],
            "family_id": [f"f{i}" for i in range(n)],
        }
    )
    cfg = {
        "eval_window_start": "2025-01-01",
        "eval_window_end": "2026-10-31",
        "freeze_date": "2026-10-31",
        "temporal_ratios": {"train": 0.75, "val": 0.10, "eval": 0.15},
        "legacy_ratios": {"train": 0.80, "val": 0.05, "test": 0.15},
    }
    out = assign.assign_splits(reviews, fams, cfg, "salt", set())
    share = out["split"].value_counts(normalize=True)
    assert abs(share["test"] - 0.15) < 0.02
    assert abs(share["val"] - 0.05) < 0.02
    assert "eval" not in share  # Legacy reviews never enter the benchmark.
    guards.check_family_disjoint(out)


def test_family_guard_raises_when_family_spans_splits():
    bad = _splits(
        [
            ("a", "f1", "train"),
            ("b", "f1", "test"),
            ("c", "f2", "train"),
            ("d", "f2", "excluded"),
        ]
    )
    with pytest.raises(decontam.LeakageError):
        guards.check_family_disjoint(bad)
    guards.check_family_disjoint(bad.iloc[2:])


def test_review_key_guard_raises_on_shared_review():
    reviews = pd.DataFrame(
        {
            "review_id": ["a", "b"],
            "review_key": ["cochrane:CD1", "cochrane:CD1"],
        }
    )
    with pytest.raises(decontam.LeakageError):
        guards.check_review_keys_disjoint(
            reviews, _splits([("a", "f1", "train"), ("b", "f2", "test")])
        )


def test_training_frame_guard_catches_val_and_test_overlap():
    records = decontam.add_fingerprints(
        pd.DataFrame(
            {
                "review_id": ["tr", "tr", "va", "te"],
                "record_id": ["pmid:1", "pmid:2", "pmid:3", "pmid:4"],
                "pmid": ["1", "2", "3", "4"],
                "title": ["alpha", "beta", "gamma", "delta"],
                "abstract": ["", "", "", ""],
                "label_final": [1, 0, 1, 0],
            }
        )
    )
    splits = _splits(
        [("tr", "f1", "train"), ("va", "f2", "val"), ("te", "f3", "test")]
    )
    train = records[records["review_id"] == "tr"]
    assert (
        sum(
            guards.check_training_frames(
                train, train.iloc[:0], records, splits
            ).values()
        )
        == 0
    )
    leaky = pd.concat(
        [train, records[records["review_id"] == "va"].assign(review_id="tr")]
    )
    with pytest.raises(decontam.LeakageError):
        guards.check_training_frames(leaky, train.iloc[:0], records, splits)
    with pytest.raises(decontam.LeakageError):
        guards.check_training_frames(records, train.iloc[:0], records, splits)


def test_ledger_blocks_until_unblinded_and_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ledger.BlindedError):
        ledger.require_unblind("internal_test", unblind=False)
    ledger.require_unblind("val", unblind=False)  # Dev set is open.
    locked = (
        tmp_path
        / "runs"
        / "calibration"
        / "e"
        / ("locked_operating_points.json")
    )
    locked.parent.mkdir(parents=True)
    locked.write_text('{"thresholds": {}}')
    entry = ledger.record_unblinding(
        str(tmp_path / "runs"), "internal_test", 20, ["e@tau99"]
    )
    assert not next(iter(entry["locked_threshold_files"].values()))[
        "preregistered"
    ]
    frozen = tmp_path / "frozen" / "internal_test"
    frozen.mkdir(parents=True)
    (frozen / "manifest.json").write_text("{}")
    written = ledger.preregister(
        str(tmp_path / "runs"), str(tmp_path / "frozen")
    )
    assert len(written) == 2
    entry = ledger.record_unblinding(
        str(tmp_path / "runs"), "internal_test", 20, ["e@tau99"]
    )
    assert next(iter(entry["locked_threshold_files"].values()))["preregistered"]
    log = (tmp_path / "runs" / "unblinding_log.jsonl").read_text()
    assert log.count("internal_test") == 2
