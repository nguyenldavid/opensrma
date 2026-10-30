"""Study-design requirements parsed from eligibility criteria."""

import json
import re

_RCT = re.compile(
    r"\b(randomi[sz]ed(?: controlled| clinical)? (?:clinical )?"
    r"trials?|RCTs?)\b",
    re.I,
)
_OTHER_DESIGNS = re.compile(
    r"\b(non-?randomi[sz]ed|quasi-?randomi[sz]ed|observational|cohort|"
    r"case-?control|cross-?sectional|before-?and-?after|interrupted time "
    r"series|controlled before|case series|any (study )?design)\b",
    re.I,
)
_INTERVENTION_TYPES = frozenset(
    {"intervention_ma", "intervention_no_ma", "nma", "dose_response"}
)


def requires_rct(criteria_list_json: str, review_type: str) -> bool:
    """True if a review only admits randomised trials.

    An intervention-type review requires RCTs when its inclusion-side
    criteria mention randomised trials and none of them admits other
    designs (exclusion criteria listing observational designs are fine).

    Args:
      criteria_list_json: JSON list from `text.criteria.split_criteria`.
      review_type: Review stratum.
    """
    if review_type not in _INTERVENTION_TYPES:
        return False
    try:
        criteria = json.loads(criteria_list_json or "[]")
    except (TypeError, json.JSONDecodeError):
        return False
    include = " ".join(c for c in criteria if not c.startswith("Exclude:"))
    return bool(_RCT.search(include)) and not _OTHER_DESIGNS.search(include)


def has_rct_publication_type(publication_types: str) -> bool:
    """MEDLINE publication type "Randomized Controlled Trial" present."""
    return "randomized controlled trial" in (publication_types or "").lower()
