"""Rule-based classifier for the 10 methodological review strata.

Deterministic keyword rules replace manual stratum annotation. Rules are
checked in priority order on the title first (most reliable), then on the
objective + abstract. The order resolves common overlaps, e.g. a "network
meta-analysis of diagnostic accuracy" is an NMA, and an "umbrella review of
meta-analyses" is an umbrella review.
"""

import re

_RULES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "rapid_scoping_umbrella",
        re.compile(
            r"\b(umbrella review|scoping review|rapid review|overview of "
            r"(systematic )?reviews|mapping review|evidence map)",
            re.I,
        ),
    ),
    (
        "qualitative",
        re.compile(
            r"\b(qualitative (evidence )?synthesis|meta-?synthesis|"
            r"thematic synthesis|meta-?ethnograph|qualitative systematic "
            r"review|framework synthesis|qualitative studies)",
            re.I,
        ),
    ),
    (
        "nma",
        re.compile(r"\bnetwork meta-?analy|\bmixed treatment comparison", re.I),
    ),
    (
        "dta",
        re.compile(
            r"\b(diagnostic (test )?accuracy|diagnostic performance|"
            r"sensitivity and specificity|accuracy of .{0,60}(for|in) "
            r"(the )?(diagnos|detect)|screening accuracy)",
            re.I,
        ),
    ),
    (
        "prognostic",
        re.compile(
            r"\b(prognos|prediction model|predictive model|risk "
            r"prediction|risk score|predictive (value|performance))",
            re.I,
        ),
    ),
    ("dose_response", re.compile(r"\bdose[- ]response", re.I)),
    (
        "prevalence",
        re.compile(
            r"\b(prevalence|incidence|burden of|epidemiology of|"
            r"proportion of)",
            re.I,
        ),
    ),
    (
        "etiology",
        re.compile(
            r"\b(risk factors?|association (between|of|with)|associated "
            r"with|exposure to|aetiolog|etiolog|causal)",
            re.I,
        ),
    ),
)
_META = re.compile(r"\bmeta-?analy|\bpooled analysis", re.I)


def classify(
    title: str, objective: str = "", abstract: str = "", hint: str | None = None
) -> str:
    """Assigns one of `schema.REVIEW_TYPES`.

    Args:
      title: Review title.
      objective: Review objective text.
      abstract: Review abstract (or criteria) text.
      hint: Optional source-provided type ("dta", "intervention",
        "prognosis", "qualitative"), used when no title rule fires.

    Returns:
      A stratum key from `schema.REVIEW_TYPES`.
    """
    title = title or ""
    body = f"{objective or ''} {abstract or ''}"
    for text in (title, body):
        for label, pattern in _RULES:
            if pattern.search(text):
                return label
    hint = (hint or "").lower()
    if hint.startswith("dta") or hint.startswith("diagnos"):
        return "dta"
    if hint.startswith("prognos"):
        return "prognostic"
    if hint.startswith("qualit"):
        return "qualitative"
    if hint.startswith("interv"):
        # Cochrane intervention reviews almost always pool their results.
        return "intervention_ma"
    if _META.search(title) or _META.search(body):
        return "intervention_ma"
    return "intervention_no_ma"
