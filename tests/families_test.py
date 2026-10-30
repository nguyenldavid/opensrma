"""Tests for the cosine-edge guard in the family graph."""

import logging

import numpy as np
import pandas as pd

from opensrbench.splits import decontam
from opensrbench.splits import families


def _constant_embed(texts):
    """Every text maps to the same vector: all pairs have cosine 1."""
    return np.ones((len(texts), 4))


def _records(rows):
    return decontam.add_fingerprints(
        pd.DataFrame(
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
    )


def _family_of(frame):
    return frame.set_index("review_id")["family_id"]


def test_has_topic_text_threshold():
    reviews = pd.DataFrame(
        {
            "title": [
                "",
                "  ",
                None,
                "Short",
                "Drug X for disease Y in adults",
            ],
            "objective": ["", " . ", None, "", ""],
        }
    )
    assert families.has_topic_text(reviews).tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_empty_text_reviews_form_no_cosine_clique(caplog):
    """Regression: 2,171 empty-title reviews became one cosine clique."""
    topic = "Drug X versus placebo for disease Y: a systematic review"
    reviews = pd.DataFrame(
        {
            "review_id": ["e1", "e2", "e3", "t1", "t2"],
            "title": ["", "", "", topic, topic],
            "objective": ["", "", "", "", ""],
        }
    )
    records = _records(
        [
            ("e1", "pmid:1", "1", "Trial one", "", 1),
            ("e2", "pmid:2", "2", "Trial two", "", 1),
            ("e3", "pmid:3", "3", "Trial three", "", 1),
            ("t1", "pmid:4", "4", "Trial four", "", 1),
            ("t2", "pmid:5", "5", "Trial five", "", 1),
        ]
    )
    with caplog.at_level(logging.WARNING):
        fams, stats = families.build_families(
            reviews, records, 0.1, 0.88, _constant_embed
        )
    fam = _family_of(fams)
    assert len({fam["e1"], fam["e2"], fam["e3"]}) == 3
    assert fam["e1"] != fam["t1"]
    # Reviews with topic text still link by cosine.
    assert fam["t1"] == fam["t2"]
    assert stats["cosine_edges"] == 1
    assert stats["no_cosine_text"] == 3
    assert stats["largest_family"] == 2
    assert "3 reviews have < 20 chars" in caplog.text


def test_empty_text_reviews_still_join_families_by_overlap():
    reviews = pd.DataFrame(
        {
            "review_id": ["e1", "e2", "e3", "t1"],
            "title": ["", "", "", "Drug X for disease Y: a systematic review"],
            "objective": ["", "", "", ""],
        }
    )
    records = _records(
        [
            ("e1", "pmid:1", "1", "Shared trial A", "", 1),
            ("e2", "pmid:1", "1", "Shared trial A", "", 1),
            ("e3", "pmid:2", "2", "Shared trial B", "", 1),
            ("t1", "pmid:2", "2", "Shared trial B", "", 1),
        ]
    )
    fams, stats = families.build_families(
        reviews, records, 0.1, 0.88, _constant_embed
    )
    fam = _family_of(fams)
    # e1-e2 share a study; e3 shares one with a topical review.
    assert fam["e1"] == fam["e2"]
    assert fam["e3"] == fam["t1"]
    assert fam["e1"] != fam["e3"]
    assert stats["overlap_edges"] == 2
    assert stats["cosine_edges"] == 0
    # Identical outcome with the cosine criterion disabled.
    fams_off, _ = families.build_families(reviews, records, 0.1, 0.88, None)
    off = _family_of(fams_off)
    assert (fam == off.reindex(fam.index)).all()


_TOPIC_A = "Drug X versus placebo for disease Y: a systematic review"
_TOPIC_B = "Exercise therapy for chronic low back pain: a meta-analysis"


def _keyed_embed(texts):
    """Topic A -> e0, topic B -> e1; every other text -> e1 (like topic B)."""
    vectors = np.zeros((len(texts), 4))
    for i, text in enumerate(texts):
        vectors[i, 0 if _TOPIC_A in text else 1] = 1.0
    return vectors


def test_cosine_guard_keeps_embeddings_aligned_with_ids():
    """Interleaved empty reviews: emb[topical] must follow the id order."""
    reviews = pd.DataFrame(
        {
            "review_id": ["t1", "e1", "t2", "e2", "t3"],
            "title": [_TOPIC_A, "", _TOPIC_B, "", _TOPIC_A],
            "objective": ["", "", "", "", ""],
        }
    )
    records = _records(
        [
            (r, f"pmid:{i}", str(i), f"Trial {i}", "", 1)
            for i, r in enumerate(reviews["review_id"], start=1)
        ]
    )
    fams, stats = families.build_families(
        reviews, records, 0.1, 0.88, _keyed_embed
    )
    fam = _family_of(fams)
    # Only t1 and t3 share topic A. The empty reviews embed like t2 but are
    # guarded, so a misaligned slice would link t2 to e1/e2 or break t1-t3.
    assert fam["t1"] == fam["t3"]
    assert len({fam["t1"], fam["t2"], fam["e1"], fam["e2"]}) == 4
    assert stats["cosine_edges"] == 1
    assert stats["no_cosine_text"] == 2


def test_cosine_guard_with_one_or_no_topical_review():
    for titles, n_topical in (([_TOPIC_A, "", ""], 1), (["", "", ""], 0)):
        reviews = pd.DataFrame(
            {
                "review_id": ["r1", "r2", "r3"],
                "title": titles,
                "objective": ["", "", ""],
            }
        )
        records = _records(
            [
                ("r1", "pmid:1", "1", "Trial one", "", 1),
                ("r2", "pmid:2", "2", "Trial two", "", 1),
                ("r3", "pmid:3", "3", "Trial three", "", 1),
            ]
        )
        fams, stats = families.build_families(
            reviews, records, 0.1, 0.88, _constant_embed
        )
        assert stats["cosine_edges"] == 0
        assert stats["no_cosine_text"] == 3 - n_topical
        assert stats["families"] == 3
        assert fams["family_size"].eq(1).all()
