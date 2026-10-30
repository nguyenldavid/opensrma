"""Tests for hashing, dates, criteria splitting and review typing."""

import datetime

from opensrbench.text import criteria
from opensrbench.text import review_type
from opensrbench.utils import dates
from opensrbench.utils import hashing


def test_fingerprint_normalizes_case_punctuation_and_accents():
    a = hashing.text_fingerprint("Effect of Café-au-lait, on X!")
    b = hashing.text_fingerprint("effect of cafe au lait on x")
    assert a == b and len(a) == 64


def test_fingerprint_of_empty_text_is_empty_string():
    assert hashing.text_fingerprint("") == ""
    assert hashing.text_fingerprint("  ...  ") == ""
    assert hashing.text_fingerprint(None) == ""
    assert hashing.text_fingerprint(float("nan")) == ""


def test_abstract_window_is_applied_after_normalization():
    base = "a" * 500
    assert hashing.text_fingerprint(
        base + "b", 500
    ) == hashing.text_fingerprint(base + " -- c", 500)


def test_parse_date_variants():
    assert dates.parse_date("2024-03-05") == datetime.date(2024, 3, 5)
    assert dates.parse_date("March 2024") == datetime.date(2024, 3, 1)
    assert dates.parse_date("March 2024", end_of_period=True) == (
        datetime.date(2024, 3, 31)
    )
    assert dates.parse_date("5 March 2024") == datetime.date(2024, 3, 5)
    assert dates.parse_date("March 5, 2024") == datetime.date(2024, 3, 5)
    assert dates.parse_date("2023", end_of_period=True) == datetime.date(
        2023, 12, 31
    )
    assert dates.parse_date("2024-02-31") == datetime.date(2024, 2, 29)
    assert dates.parse_date("") is None


def test_split_criteria_assigns_polarity():
    text = (
        "Inclusion criteria were as follows: (1) adults with asthma; (2) "
        "randomised trials. Exclusion criteria included: (1) animal "
        "studies; (2) conference abstracts."
    )
    out = criteria.split_criteria(text)
    assert out == [
        "Include: adults with asthma",
        "Include: randomised trials",
        "Exclude: animal studies",
        "Exclude: conference abstracts",
    ]


def test_split_criteria_caps_count_and_handles_empty():
    assert criteria.split_criteria("") == []
    many = " ".join(
        f"({i}) criterion number {i} is long enough" for i in range(1, 30)
    )
    assert len(criteria.split_criteria(many, max_criteria=5)) == 5


def test_review_type_priority_rules():
    assert (
        review_type.classify("Network meta-analysis of diagnostic accuracy")
        == "nma"
    )
    assert review_type.classify("An umbrella review of meta-analyses") == (
        "rapid_scoping_umbrella"
    )
    assert (
        review_type.classify("Diagnostic accuracy of CT for appendicitis")
        == "dta"
    )
    assert review_type.classify("Statins for stroke: a meta-analysis") == (
        "intervention_ma"
    )
    assert review_type.classify("Exercise for back pain") == (
        "intervention_no_ma"
    )
    assert review_type.classify("X for Y", hint="Intervention") == (
        "intervention_ma"
    )
