"""Automatic PubMed query synthesis from a review's own PICO text.

Tier-2 reconstruction must not look at the review's positives (that would
leak labels into pool construction), so queries are built only from the
review title, objective and criteria.

`HeuristicQueryGenerator` (default, deterministic, no model) splits the
title/objective into concept chunks at PICO connectives ("for", "in",
"versus", "among", ...) and relies on PubMed Automatic Term Mapping for each
chunk. `strict` ANDs the concepts (precise, like the authors' search);
`broad` ORs them (high recall) and is later re-ranked by BM25 + MedCPT.

`LlmQueryGenerator` asks an OpenAI-compatible endpoint (e.g. gpt-oss-20b
served by vLLM on Killarney) for a Boolean query; it is optional.
"""

import re

_BOILERPLATE = re.compile(
    r"(:|-|–|—)?\s*(an?\s+)?(updated\s+)?(systematic(\s+literature)?"
    r"\s+review|meta-?analys[ie]s|network\s+meta-?analysis|scoping\s+review|"
    r"umbrella\s+review|rapid\s+review|review\s+of\s+the\s+literature|"
    r"literature\s+review|pooled\s+analysis|dose-?response|meta-?regression|"
    r"bayesian|a\s+protocol|protocol|evidence\s+synthesis|trial\s+sequential"
    r"\s+analysis)(\s+(and|&)\s+)?",
    re.I,
)
_CONNECTIVES = re.compile(
    r"\s+(?:for|in|among|amongst|versus|vs\.?|compared\s+(?:with|to)|on|of|"
    r"with|without|after|during|following|and|or|to|from|by|as|at|"
    r"undergoing|receiving|treated\s+with)\s+",
    re.I,
)
_STOP = frozenset(
    "a an the of in on for with to and or by at from as is are be effect "
    "effects efficacy effectiveness safety impact role association "
    "associations relationship use using based outcomes outcome patients "
    "people adults study studies evidence clinical among between versus vs "
    "comparison comparative randomized randomised controlled trials trial "
    "review systematic meta analysis analyses prevalence incidence risk "
    "factor factors accuracy diagnostic prognostic predictive value "
    "performance management intervention interventions".split()
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")


def _clean_chunk(chunk: str) -> str:
    tokens = [t for t in _TOKEN.findall(chunk) if t.lower() not in _STOP]
    return " ".join(tokens[:5])


def concept_chunks(
    title: str, objective: str = "", max_chunks: int = 4
) -> list[str]:
    """Splits review text into PICO-ish concept phrases.

    Args:
      title: Review title (preferred; it is the most compact PICO summary).
      objective: Fallback text when the title is uninformative.
      max_chunks: Maximum number of concepts.

    Returns:
      Concept phrases ordered by specificity (longest first).
    """
    text = _BOILERPLATE.sub(" ", title or "")
    text = re.sub(r"[()\[\]?:;,]", " ", text)
    chunks = [_clean_chunk(c) for c in _CONNECTIVES.split(text)]
    chunks = [c for c in chunks if len(c) >= 3]
    if len(chunks) < 2 and objective:
        sentence = re.split(r"(?<=[.!?])\s+", objective.strip())[0]
        sentence = re.sub(
            r"^(to|we)\s+\w+\s+(the\s+)?", "", sentence, flags=re.I
        )
        chunks.extend(_clean_chunk(c) for c in _CONNECTIVES.split(sentence))
        chunks = [c for c in dict.fromkeys(chunks) if len(c) >= 3]
    chunks = sorted(dict.fromkeys(chunks), key=len, reverse=True)
    return chunks[:max_chunks]


def _atm(chunk: str) -> str:
    """Wraps a concept for PubMed Automatic Term Mapping (no field tags)."""
    words = chunk.split()
    if len(words) == 1:
        return f"{words[0]}"
    return "(" + " ".join(words) + ")"


class HeuristicQueryGenerator:
    """Deterministic PICO-chunk query builder."""

    def queries(
        self, title: str, objective: str = "", criteria: str = ""
    ) -> dict[str, str]:
        """Returns {"strict": ..., "broad": ...} PubMed queries ("" if none)."""
        del criteria  # Unused by the heuristic; kept for interface parity.
        chunks = concept_chunks(title, objective)
        if not chunks:
            return {"strict": "", "broad": ""}
        strict = " AND ".join(_atm(c) for c in chunks[:3])
        broad = " OR ".join(_atm(c) for c in chunks[:2])
        if len(chunks) >= 3:
            # Broad = any two of the top-three concepts.
            a, b, c = (_atm(x) for x in chunks[:3])
            broad = f"({a} AND {b}) OR ({a} AND {c}) OR ({b} AND {c})"
        return {"strict": strict, "broad": broad}


_LLM_PROMPT = """You are an expert medical information specialist.
Write ONE PubMed Boolean search strategy for the systematic review below.
Use MeSH terms ([Mesh]) and title/abstract terms ([tiab]) grouped by PICO
concept with OR inside concepts and AND between concepts. Do not use date or
publication-type limits. Output only the query on a single line.

Title: {title}
Objective: {objective}
Eligibility criteria: {criteria}
"""


class LlmQueryGenerator:
    """Query builder backed by an OpenAI-compatible chat endpoint."""

    def __init__(self, chat_fn):
        """Args: chat_fn: callable(prompt: str) -> str (see llm.client)."""
        self._chat = chat_fn
        self._fallback = HeuristicQueryGenerator()

    def queries(
        self, title: str, objective: str = "", criteria: str = ""
    ) -> dict[str, str]:
        fallback = self._fallback.queries(title, objective)
        try:
            text = self._chat(
                _LLM_PROMPT.format(
                    title=title,
                    objective=objective[:1500],
                    criteria=criteria[:2500],
                )
            )
        except Exception:  # pylint: disable=broad-except
            return fallback
        line = " ".join((text or "").strip().splitlines()).strip().strip("`")
        if line.count("(") != line.count(")") or len(line) < 10:
            return fallback
        return {"strict": line, "broad": fallback["broad"] or line}
