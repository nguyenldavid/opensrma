"""Tests for the Cochrane-recipe RCT classifier and design parsing."""

import json

import numpy as np
import pandas as pd

from opensrbench.rct import classifier
from opensrbench.rct import corpus
from opensrbench.splits import decontam
from opensrbench.text import design

_FILL = (
    "patients outcomes were measured at baseline and follow up in the "
    "hospital clinical sample with analysis of the data groups "
) * 6


def _corpus(n, seed):
    rng = np.random.default_rng(seed)
    titles, abstracts, labels = [], [], []
    for i in range(n):
        rct = rng.random() < 0.3
        lead = (
            "participants were randomly assigned to treatment or placebo "
            "in this double blind randomized controlled trial"
            if rct
            else rng.choice(
                [
                    "a retrospective cohort study of registry data",
                    "a case control study in two hospitals",
                    "a narrative review of the literature",
                ]
            )
        )
        titles.append(f"Effect of drug {i} on outcome")
        abstracts.append(f"{lead}. {_FILL}")
        labels.append(int(rct))
    return titles, abstracts, np.array(labels)


def test_cochrane_classifier_separates_and_honours_recall_cutoff(tmp_path):
    t, a, y = _corpus(600, 0)
    ct, ca, cy = _corpus(300, 1)
    tt, ta, ty = _corpus(300, 2)
    model = classifier.CochraneRctClassifier(min_df=1).fit(
        t, a, y, ct, ca, cy, target_recall=0.95
    )
    metrics = model.evaluate(tt, ta, ty)
    assert metrics["auc"] > 0.95 and metrics["recall"] >= 0.9
    model.save(str(tmp_path))
    loaded = classifier.CochraneRctClassifier.load(str(tmp_path))
    np.testing.assert_allclose(
        loaded.predict_proba(tt, ta), model.predict_proba(tt, ta)
    )
    assert (
        json.loads((tmp_path / classifier.CARD_FILE).read_text())[
            "target_recall"
        ]
        == 0.95
    )


def test_short_records_are_not_machine_classified():
    t, a, y = _corpus(300, 0)
    model = classifier.CochraneRctClassifier(min_df=1).fit(t, a, y, t, a, y)
    probs = model.predict_proba(
        ["short", "A sufficiently long title"],
        ["x" * 500, "too short abstract"],
    )
    assert np.isnan(probs).all()


def test_requires_rct_parsing():
    rct_only = json.dumps(
        [
            "Include: randomised controlled trials of adults",
            "Exclude: observational studies",
        ]
    )
    mixed = json.dumps(["Include: RCTs and prospective cohort studies"])
    assert design.requires_rct(rct_only, "intervention_ma")
    assert not design.requires_rct(mixed, "intervention_ma")
    assert not design.requires_rct(rct_only, "dta")
    assert design.has_rct_publication_type(
        "Journal Article|Randomized " "Controlled Trial"
    )


def test_rct_corpus_is_decontaminated_against_protected_and_val():
    records = decontam.add_fingerprints(
        pd.DataFrame(
            {
                "review_id": ["tr", "va", "te"],
                "record_id": ["a", "b", "c"],
                "pmid": ["1", "2", "3"],
                "title": ["t1", "t2", "t3"],
                "abstract": ["", "", ""],
                "label_final": [0, 0, 0],
            }
        )
    )
    split_of = pd.Series({"tr": "train", "va": "val", "te": "test"})
    rct_corpus = pd.DataFrame(
        {
            "pmid": ["1", "2", "3", "4"],
            "title": ["t1", "x", "y", "t3"],
            "abstract": ["", "", "", ""],
            "label": [1, 0, 1, 0],
            "stratum": "rct",
        }
    )
    clean, audit = corpus.decontaminate(
        rct_corpus, records, split_of, ("test",)
    )
    assert clean["pmid"].tolist() == ["1"]  # 2: val pmid; 3/4: test pmid/title.
    assert audit["dropped"] == 3 and sum(audit["final"].values()) == 0
