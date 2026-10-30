"""Prompt contracts and decision parsers (protocol Section 4).

Every system receives the same four fields: objective, selection_criteria,
title and abstract.
"""

import re

_TAG = re.compile(
    r"<decision>\s*(INCLUDE|EXCLUDE|UNCERTAIN)\s*</decision>", re.I
)
PLACEHOLDER_MARK = "PASTE_THE_VERBATIM"


class PromptNotConfiguredError(RuntimeError):
    """Raised when a template still contains its placeholder notice."""


def load_template(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        template = fh.read()
    if PLACEHOLDER_MARK in template:
        raise PromptNotConfiguredError(
            f"{path} still contains the placeholder; paste the verbatim "
            "published prompt first (see the file)."
        )
    return template


def render(
    template: str,
    objective: str,
    selection_criteria: str,
    title: str,
    abstract: str,
) -> str:
    """Fills the four protocol fields (other braces are left untouched)."""
    fields = {
        "objective": objective or "(not reported)",
        "selection_criteria": selection_criteria or "(not reported)",
        "title": title or "(no title)",
        "abstract": abstract or "(no abstract)",
    }
    out = template
    for key, value in fields.items():
        out = out.replace("{" + key + "}", value)
    return out


def parse_tag(text: str | None) -> tuple[int, bool]:
    """Parses <decision>...</decision>; returns (keep, valid).

    The last tag wins. EXCLUDE -> (0, True); INCLUDE -> (1, True);
    UNCERTAIN, no tag or None -> (1, False) per the abstention rule.
    """
    if not text:
        return 1, False
    matches = _TAG.findall(text)
    if not matches:
        return 1, False
    last = matches[-1].upper()
    if last == "EXCLUDE":
        return 0, True
    if last == "INCLUDE":
        return 1, True
    return 1, False


def parse_cao(text: str | None) -> tuple[int, bool]:
    """Cao et al. parser: final 500 chars, YYY = include/uncertain, XXX = out.

    Both or neither marker -> (1, False).
    """
    if not text:
        return 1, False
    tail = text[-500:]
    has_y, has_x = "YYY" in tail, "XXX" in tail
    if has_y and not has_x:
        return 1, True
    if has_x and not has_y:
        return 0, True
    return 1, False


PARSERS = {"tag": parse_tag, "cao": parse_cao}
