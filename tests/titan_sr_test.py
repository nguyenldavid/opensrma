"""Tests for the TITAN-SR manifest adapter (synthetic fixture, no real data)."""

import json
import os

import pandas as pd
import pytest

from opensrbench import config
from opensrbench.sources import titan_sr


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("24128044.pdf", {"pmid": "24128044"}),
        (
            "1-s2.0-S0020138317306034-main.pdf",
            {"pii": "S0020-1383(17)30603-4"},
        ),
        ("1-s2.0-S014067361730001X-main.pdf", {"pii": "S0140-6736(17)30001-X"}),
        ("s00259-017-3683-7.pdf", {"doi": "10.1007/s00259-017-3683-7"}),
        ("oncotarget-v8i14-15520.pdf", {"doi": "10.18632/oncotarget.15520"}),
        ("e871.full.pdf", {}),
        ("gfw437.pdf", {}),
        ("review 12.pdf", {}),
    ],
)
def test_trackb_identifiers(filename, expected):
    assert titan_sr.trackb_identifiers(filename) == expected


def _write_fixture(root: str) -> None:
    repo = os.path.join(root, "raw", "titan_sr", "repo", "data")
    os.makedirs(os.path.join(repo, "manifests"))
    os.makedirs(os.path.join(repo, "splits"))
    trackb = ["11111111.pdf", "1-s2.0-S0020138317306034-main.pdf", "x.pdf"]
    pd.DataFrame(
        {
            "review_id": ["PMC1", "PMC1"],
            "candidate_pmid": ["101", "102"],
            "label": ["1", "0"],
            "source": ["pmc", "pmc"],
        }
    ).to_csv(
        os.path.join(repo, "manifests", "pmc_train_labels.csv.gz"), index=False
    )
    pd.DataFrame(
        {
            "review_id": [r for r in trackb for _ in (0, 1)],
            "candidate_pmid": ["201", "202", "203", "204", "205", "206"],
            "label": ["1", "0"] * 3,
            "source": ["trackb"] * 6,
        }
    ).to_csv(
        os.path.join(repo, "manifests", "trackb_train_labels.csv.gz"),
        index=False,
    )
    pd.DataFrame(
        {
            "review_id": ["PMC1"] + trackb,
            "source": ["pmc"] + ["trackb"] * 3,
            "eligibility_criteria": ["We included trials."] * 4,
        }
    ).to_csv(os.path.join(repo, "manifests", "criteria.csv.gz"), index=False)
    assignments = {"pmc_PMC1": {"split": "train"}}
    assignments.update({f"tb_{r}": {"split": "test"} for r in trackb})
    with open(
        os.path.join(repo, "splits", "stratified_split_v2.json"), "w"
    ) as fh:
        json.dump({"assignments": assignments}, fh)


def test_ingest_sets_trackb_review_identifiers(tmp_path):
    """Regression: Track B reviews had no PMID/DOI, so enrich never filled
    their title/objective and they formed one giant cosine family."""
    _write_fixture(str(tmp_path))
    paths = config.Paths(root=str(tmp_path), runs=str(tmp_path / "runs"))
    adapter = titan_sr.TitanSrAdapter(
        {"include_external_holdout": False}, paths
    )
    reviews, records = adapter.ingest()
    rv = reviews.set_index("review_key")
    pmid_review = rv.loc["trackb:11111111.pdf"]
    assert pmid_review["pmid"] == "11111111"
    assert pd.isna(pmid_review["doi"])
    elsevier = rv.loc["trackb:1-s2.0-S0020138317306034-main.pdf"]
    assert json.loads(elsevier["extra"])["pii"] == "S0020-1383(17)30603-4"
    assert pd.isna(elsevier["pmid"]) and pd.isna(elsevier["doi"])
    unknown = rv.loc["trackb:x.pdf"]
    assert pd.isna(unknown["pmid"]) and pd.isna(unknown["doi"])
    pmc = rv.loc["pmc:PMC1"]
    assert pmc["pmcid"] == "PMC1" and pd.isna(pmc["pmid"])
    assert json.loads(pmc["extra"]) == {
        "titan_source": "pmc",
        "titan_split": "train",
    }
    # Labels are untouched by the identifier recovery.
    assert len(records) == 8
    assert int(records["label_final"].sum()) == 4
