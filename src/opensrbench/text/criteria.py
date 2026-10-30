"""Decomposition of free-text eligibility criteria into atomic criteria.

The decomposed cross-encoder scores every criterion separately and vetoes a
record through a soft-min, so it needs criteria as a list. This module does
that automatically (no annotation): it detects PICOS / inclusion / exclusion
headings, splits enumerations ("(1) ...; (2) ..."), bullets and sentences,
and prefixes each criterion with its polarity/role so the model can tell
"Exclude: animal studies" from "Include: animal studies".
"""

import re

_HEADINGS = (
    (
        "Exclude",
        re.compile(
            r"\b(exclusion criteria|were excluded|exclude[ds]?"
            r"\b|not eligible|ineligible)",
            re.I,
        ),
    ),
    (
        "Population",
        re.compile(
            r"\b(types of participants|participants|"
            r"population|patients)\s*[:\-]",
            re.I,
        ),
    ),
    (
        "Intervention",
        re.compile(
            r"\b(types of interventions?|interventions?|"
            r"exposures?|index tests?)\s*[:\-]",
            re.I,
        ),
    ),
    (
        "Comparator",
        re.compile(
            r"\b(comparators?|comparisons?|control|"
            r"reference standards?)\s*[:\-]",
            re.I,
        ),
    ),
    (
        "Outcome",
        re.compile(
            r"\b(types of outcome measures|outcomes?|target "
            r"conditions?)\s*[:\-]",
            re.I,
        ),
    ),
    (
        "Study design",
        re.compile(
            r"\b(types of studies|study designs?|" r"design)\s*[:\-]", re.I
        ),
    ),
    (
        "Include",
        re.compile(
            r"\b(inclusion criteria|were included|eligible "
            r"if|included if|we included)",
            re.I,
        ),
    ),
)
_ENUM_SPLIT = re.compile(
    r"(?:^|\s|;)\(?(?:\d{1,2}|[ivx]{1,4}|[a-h])\)\s+"
    r"|(?:^|\n)\s*(?:[-•*▪●◦]|\d{1,2}[.)])\s+",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")
_WS = re.compile(r"\s+")


def _role_of(text: str, default: str) -> str:
    for role, pattern in _HEADINGS:
        if pattern.search(text[:80]):
            return role
    return default


def _is_heading(fragment: str) -> bool:
    """True for short "Exclusion criteria included:"-style lead-ins."""
    return (
        len(fragment.split()) <= 6
        and fragment.endswith(":")
        and bool(_role_of(fragment, ""))
    )


def split_criteria(
    criteria_text: str,
    max_criteria: int = 12,
    min_chars: int = 8,
    max_chars: int = 400,
) -> list[str]:
    """Splits a criteria paragraph into role-prefixed atomic criteria.

    Args:
      criteria_text: Raw eligibility-criteria text.
      max_criteria: Keep at most this many criteria (the longest-first
        truncation keeps informative ones; order is preserved).
      min_chars: Drop fragments shorter than this.
      max_chars: Truncate fragments longer than this.

    Returns:
      List like ["Population: adults with type 2 diabetes", "Exclude: ..."].
      Empty input returns [].
    """
    if not isinstance(criteria_text, str) or not criteria_text.strip():
        return []
    text = criteria_text.replace("\r", "\n")
    # First split by enumerations / bullets, then by sentences.
    parts = [p for p in _ENUM_SPLIT.split(text) if p and p.strip()]
    fragments: list[str] = []
    for part in parts:
        fragments.extend(s for s in _SENTENCE_SPLIT.split(part) if s.strip())
    out: list[str] = []
    role = "Include"
    for fragment in fragments:
        fragment = _WS.sub(" ", fragment).strip(" ;,.-")
        if _is_heading(fragment):
            # Heading-only fragments ("Exclusion criteria included:") set the
            # role of the criteria that follow instead of becoming criteria.
            role = _role_of(fragment, role)
            continue
        if len(fragment) < min_chars:
            continue
        new_role = _role_of(fragment, role)
        if new_role in ("Include", "Exclude"):
            role = new_role
        fragment = fragment[:max_chars]
        out.append(f"{new_role}: {fragment}")
    if len(out) > max_criteria:
        keep = sorted(
            sorted(range(len(out)), key=lambda i: -len(out[i]))[:max_criteria]
        )
        out = [out[i] for i in keep]
    return out
