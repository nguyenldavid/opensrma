"""JATS XML parsing of systematic reviews (protocol criteria 2-4).

Extracts, without any human annotation:
  * objective and eligibility criteria (criterion 2);
  * the included studies, via (a) Cochrane-style "studies included" reference
    lists, (b) "characteristics of included studies" tables, or (c) the
    citations inside "included studies / study characteristics" results
    sections (criterion 3), plus the "excluded at full text" studies;
  * a verbatim PubMed search strategy, PRISMA identification counts and the
    literature-search cut-off date (criterion 4).

Numeric citation ranges such as "[12-15]" are expanded using ref-list order,
because JATS only tags the end points with <xref>.
"""

import dataclasses
import datetime
import re

from lxml import etree

from opensrbench.utils import dates

_WS = re.compile(r"\s+")
_OBJECTIVE_TITLE = re.compile(r"\b(objectives?|aims?|purpose|goals?)\b", re.I)
_OBJECTIVE_SENTENCE = re.compile(
    r"\b(aims?|aimed|objectives?|purpose|we (sought|aimed|set out)|"
    r"this (systematic review|meta-analysis|review|study) (aims?|aimed|"
    r"sought|evaluates?|assesse[sd]|examine[sd]|investigate[sd])|"
    r"to (assess|evaluate|investigate|determine|compare|examine|explore|"
    r"synthesi[sz]e|quantify|estimate) (the|whether|if))",
    re.I,
)
_CRITERIA_TITLE = re.compile(
    r"(eligib|inclusion|exclusion|selection criteria|criteria for "
    r"(considering|inclusion)|study selection|picos?\b)",
    re.I,
)
_CRITERIA_PARAGRAPH = re.compile(
    r"(inclusion criteria|exclusion criteria|were (considered )?eligible|"
    r"were included if|were excluded if|eligibility criteria)",
    re.I,
)
_INCLUDED_LIST_TITLE = re.compile(r"(studies included|included studies)", re.I)
_EXCLUDED_LIST_TITLE = re.compile(r"(studies excluded|excluded studies)", re.I)
_INCLUDED_TABLE = re.compile(
    r"(\b(included|selected|eligible)\b.{0,40}\bstud(y|ies)\b|"
    r"\bstud(y|ies)\b.{0,40}\b(included|selected)\b|"
    r"characteristics of (the )?(\w+ ){0,2}(stud|trial)|"
    r"summary of (the )?(included )?(stud|trial))",
    re.I,
)
_EXCLUDED_TABLE = re.compile(
    r"(excluded (stud|trial|article|full)|reasons? for exclusion|"
    r"excluded after full)",
    re.I,
)
_INCLUDED_SECTION = re.compile(
    r"(included stud|study characteristics|characteristics of (the )?"
    r"(included )?(stud|trial)|description of (the )?(included )?stud|"
    r"basic (information|characteristics) of|overview of (the )?(included )?"
    r"stud)",
    re.I,
)
_METHODS_SECTION = re.compile(
    r"\b(methods?|statistical analysis|risk of "
    r"bias|quality assessment|data synthesis)\b",
    re.I,
)
_FIELD_TAG = re.compile(
    r"\[(mh|mesh|mesh terms?|mesh:noexp|majr|tiab|ti|ab|tw|pt|sh|all fields|"
    r"title/abstract|title|text word|publication type|nm|ot|dp|la)\]",
    re.I,
)
_PRISMA_PUBMED = re.compile(
    r"(pubmed|medline)[^.;\d]{0,40}?[(\[]\s*n\s*=\s*([\d,]{2,7})\s*[)\]]", re.I
)
_PRISMA_TOTAL = re.compile(
    r"([\d,]{3,7})\s+(records|articles|studies|citations|references|"
    r"publications|reports|hits)\s+(were\s+|was\s+)?(initially\s+)?"
    r"(identified|retrieved|found|obtained|yielded)",
    re.I,
)
_PRISMA_TOTAL_2 = re.compile(
    r"(identified|retrieved|yielded|found|returned)\s+(a total of\s+)?"
    r"([\d,]{3,7})\s+(records|articles|studies|citations|references|"
    r"publications|reports|hits)",
    re.I,
)
_SEARCH_DATE = re.compile(
    r"(search(ed|es)?|inception|retrieved|conducted|performed|updated)"
    r"[^.]{0,200}?\b(to|until|through|up to|till|on|before|between "
    r"[^.]{0,40}? and)\s+((\d{1,2}(st|nd|rd|th)?\s+)?[A-Z][a-z]+\.?\s+"
    r"(\d{1,2}(st|nd|rd|th)?,?\s+)?\d{4}|\d{4}[-/]\d{1,2}([-/]\d{1,2})?)",
    re.I,
)


@dataclasses.dataclass
class Reference:
    """One <ref> entry."""

    rid: str
    order: int
    pmid: str | None = None
    doi: str | None = None
    title: str = ""
    year: str | None = None


@dataclasses.dataclass
class ParsedReview:
    """Everything the harvester needs from one review's JATS XML."""

    pmcid: str = ""
    pmid: str | None = None
    doi: str | None = None
    title: str = ""
    abstract: str = ""
    pub_date: str | None = None
    objective: str = ""
    selection_criteria: str = ""
    references: dict[str, Reference] = dataclasses.field(default_factory=dict)
    included_rids: list[str] = dataclasses.field(default_factory=list)
    excluded_rids: list[str] = dataclasses.field(default_factory=list)
    positive_source: str = "none"
    search_strategy: str | None = None
    prisma_pubmed: int | None = None
    prisma_total: int | None = None
    search_date: str | None = None
    methods_text: str = ""


def _text(node) -> str:
    if node is None:
        return ""
    return _WS.sub(" ", "".join(node.itertext())).strip()


def _local(tag) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _strip_namespaces(root: etree._Element) -> None:
    for node in root.iter():
        if isinstance(node.tag, str) and "}" in node.tag:
            node.tag = _local(node.tag)


def _sec_title(sec) -> str:
    title = sec.find("title")
    return _text(title)


def _sec_body_text(sec) -> str:
    """Text of a section excluding nested tables/figures."""
    parts = []
    for node in sec.iter("p", "list-item", "title"):
        if node.getparent() is not None and _local(node.getparent().tag) in (
            "caption",
        ):
            continue
        parts.append(_text(node))
    return _WS.sub(" ", " ".join(dict.fromkeys(parts))).strip()


def _parse_date_node(node) -> datetime.date | None:
    if node is None:
        return None
    year = node.findtext("year")
    if not year or not year.strip().isdigit():
        return None
    month = node.findtext("month") or "1"
    day = node.findtext("day") or "1"
    month = int(month) if month.strip().isdigit() else 1
    day = int(day) if day.strip().isdigit() else 1
    return dates.parse_date(f"{int(year):04d}-{month:02d}-{day:02d}")


def _references(root) -> dict[str, Reference]:
    refs: dict[str, Reference] = {}
    for order, ref in enumerate(root.iter("ref")):
        rid = ref.get("id")
        if not rid:
            continue
        entry = Reference(rid=rid, order=order)
        for pub_id in ref.iter("pub-id"):
            kind = (pub_id.get("pub-id-type") or "").lower()
            value = (pub_id.text or "").strip()
            if kind == "pmid" and value.isdigit():
                entry.pmid = value
            elif kind == "doi" and value:
                entry.doi = value.lower()
        title = ref.find(".//article-title")
        if title is None:
            title = ref.find(".//chapter-title")
        entry.title = _text(title)
        citation_text = _text(ref)
        if not entry.title:
            # Unstructured <mixed-citation>: keep the full string; resolution
            # matches known titles as substrings of it.
            entry.title = citation_text[:400]
        if not entry.doi:
            match = _DOI.search(citation_text)
            if match:
                entry.doi = match.group(0).rstrip(".,;)").lower()
        entry.year = ref.findtext(".//year")
        refs[rid] = entry
    return refs


def _xref_rids(node, refs: dict[str, Reference]) -> list[str]:
    """Collects bibliographic xref targets, expanding "12-15" style ranges."""
    order = sorted(refs.values(), key=lambda r: r.order)
    position = {r.rid: i for i, r in enumerate(order)}
    out: list[str] = []
    xrefs = [x for x in node.iter("xref") if x.get("ref-type") == "bibr"]
    for i, xref in enumerate(xrefs):
        rids = (xref.get("rid") or "").split()
        out.extend(r for r in rids if r in refs)
        tail = (xref.tail or "").strip()
        if (
            tail[:1] in ("-", "–", "—")
            and i + 1 < len(xrefs)
            and rids
            and rids[-1] in position
        ):
            nxt = (xrefs[i + 1].get("rid") or "").split()
            if nxt and nxt[0] in position:
                lo, hi = position[rids[-1]], position[nxt[0]]
                if 0 < hi - lo <= 200:
                    out.extend(order[j].rid for j in range(lo + 1, hi))
    return list(dict.fromkeys(out))


_AUTHOR_YEAR = re.compile(
    r"([A-Z][A-Za-z'\u00C0-\u017F\-]{1,30})"
    r"(?:\s+et\s+al\.?|\s+and\s+[A-Z][a-z]+)?,?\s*"
    r"\(?((?:19|20)\d{2})[a-z]?\)?"
)
_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")
_BRACKET_NUMBER = re.compile(r"\[(\d{1,3})\]")


def _table_text_rids(table, root, refs: dict[str, Reference]) -> list[str]:
    """Resolves study names written as plain text in a table's cells.

    Many journals type "Smith et al. 2021" or "[12]" into table cells
    without <xref> markup. Author-year strings are matched to the first
    author's surname + year of each <ref>; bracketed numbers to <label>s.
    """
    by_author_year: dict[tuple[str, str], list[str]] = {}
    by_label: dict[str, str] = {}
    for ref in root.iter("ref"):
        rid = ref.get("id")
        if rid not in refs:
            continue
        surname = ref.findtext(".//surname") or ""
        year = (ref.findtext(".//year") or "")[:4]
        if surname and year:
            key = (surname.strip().lower(), year)
            by_author_year.setdefault(key, []).append(rid)
        label = (ref.findtext("label") or "").strip().strip(".[]")
        if label.isdigit():
            by_label[label] = rid
    out = []
    for cell in table.iter("td", "th"):
        text = _text(cell)
        for surname, year in _AUTHOR_YEAR.findall(text):
            matches = by_author_year.get((surname.lower(), year), [])
            if len(matches) == 1:
                out.append(matches[0])
        for number in _BRACKET_NUMBER.findall(text):
            if number in by_label:
                out.append(by_label[number])
    return list(dict.fromkeys(out))


def _find_objective(root) -> str:
    abstract = root.find(".//front//abstract")
    if abstract is not None:
        for sec in abstract.iter("sec"):
            if _OBJECTIVE_TITLE.search(_sec_title(sec)):
                text = _sec_body_text(sec)
                title = _sec_title(sec)
                return (
                    text[len(title) :].strip()
                    if text.startswith(title)
                    else text
                )
        sentences = re.split(r"(?<=[.!?])\s+", _text(abstract))
        hits = [s for s in sentences if _OBJECTIVE_SENTENCE.search(s)]
        if hits:
            return hits[0]
    body = root.find(".//body")
    if body is not None:
        paragraphs = [_text(p) for p in body.iter("p")][:40]
        for paragraph in paragraphs:
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                if re.search(
                    r"\b(the aim|objective of this|we aimed|this "
                    r"(systematic )?review aims?)",
                    sentence,
                    re.I,
                ):
                    return sentence
    return ""


def _find_criteria(body) -> str:
    chunks = []
    for sec in body.iter("sec"):
        title = _sec_title(sec)
        if _CRITERIA_TITLE.search(title) and not re.search(
            r"search|quality|risk of bias|data extraction", title, re.I
        ):
            text = _sec_body_text(sec)
            if text.startswith(title):
                text = text[len(title) :].strip()
            if len(text) > 40:
                chunks.append(text)
    if not chunks:
        for paragraph in body.iter("p"):
            text = _text(paragraph)
            if _CRITERIA_PARAGRAPH.search(text) and len(text) > 60:
                chunks.append(text)
    return " ".join(dict.fromkeys(chunks))[:6000]


_QUOTES = str.maketrans(
    {c: '"' for c in "\u2033\u201c\u201d\u201e\u00ab\u00bb\u2036"}
)


def normalize_strategy(text: str) -> str:
    """Cleans a copied search block into PubMed-executable syntax."""
    text = text.translate(_QUOTES).replace("\u2019", "'")
    text = re.sub(r"^.*?(pubmed|medline)\s*[:\-]\s*", "", text, flags=re.I)
    # Drop any caption/prose before the first query token.
    start = re.search(r"[(\"#]", text)
    if start:
        text = text[start.start() :]
    text = re.sub(r"\s*\(?n\s*=\s*[\d,]+\)?\s*$", "", text)
    text = re.sub(r"\s*\[\s*", "[", text)
    text = re.sub(r"\s*\]", "]", text)
    text = re.sub(r'"\s+', '"', re.sub(r'\s+"', ' "', text))
    text = _WS.sub(" ", text).strip()
    # Concatenated top-level sub-queries "(..) AND (..)(..) AND (..)" are
    # alternatives: wrap each one and OR them so precedence is explicit.
    segments, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char == "(":
            if (
                depth == 0
                and i > start
                and text[start:i].rstrip().endswith(")")
            ):
                segments.append(text[start:i].strip())
                start = i
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
    segments.append(text[start:].strip())
    if len(segments) > 1:
        text = " OR ".join(f"({seg})" for seg in segments if seg)
    return text


def _find_search_strategy(root) -> str | None:
    """Returns the most PubMed-like Boolean block, or None."""
    best, best_tags = None, 1
    for node in root.iter("p", "preformat", "td", "list-item", "disp-quote"):
        text = _text(node)
        if len(text) < 30 or len(text) > 20000:
            continue
        tags = len(_FIELD_TAG.findall(text))
        if tags <= best_tags or not re.search(r"\b(AND|OR)\b", text):
            continue
        candidate = normalize_strategy(text)
        if candidate.count("(") == candidate.count(")"):
            best, best_tags = candidate, tags
    return best


def _to_int(value: str) -> int | None:
    value = value.replace(",", "")
    return int(value) if value.isdigit() else None


def parse_jats(xml: bytes, pmcid: str = "") -> ParsedReview:
    """Parses a review's JATS XML.

    Args:
      xml: Raw XML bytes (Europe PMC fullTextXML).
      pmcid: PMCID for bookkeeping.

    Returns:
      A populated `ParsedReview`; fields that could not be found are empty.
    """
    parser = etree.XMLParser(
        recover=True, huge_tree=True, resolve_entities=False, no_network=True
    )
    root = etree.fromstring(xml, parser=parser)
    out = ParsedReview(pmcid=pmcid)
    if root is None:
        return out
    _strip_namespaces(root)
    meta = root.find(".//front/article-meta")
    if meta is not None:
        for node in meta.iter("article-id"):
            kind = node.get("pub-id-type")
            if kind == "pmid":
                out.pmid = (node.text or "").strip()
            elif kind == "doi":
                out.doi = (node.text or "").strip().lower()
            elif kind in ("pmc", "pmcid") and not out.pmcid:
                out.pmcid = "PMC" + (node.text or "").strip().lstrip("PMC")
        out.title = _text(meta.find(".//title-group/article-title"))
        pub_dates = [_parse_date_node(n) for n in meta.iter("pub-date")]
        pub_dates = [d for d in pub_dates if d]
        out.pub_date = dates.to_iso(min(pub_dates)) if pub_dates else None
        out.abstract = _text(meta.find("abstract"))
    out.objective = _find_objective(root)
    body = root.find(".//body")
    if body is None:
        return out
    out.selection_criteria = _find_criteria(body)
    out.references = _references(root)
    refs = out.references

    included: list[str] = []
    excluded: list[str] = []
    # (a) Cochrane-style grouped reference lists.
    for ref_list in root.iter("ref-list"):
        title = _text(ref_list.find("title"))
        rids = [r.get("id") for r in ref_list.iter("ref") if r.get("id")]
        if _INCLUDED_LIST_TITLE.search(title):
            included.extend(rids)
        elif _EXCLUDED_LIST_TITLE.search(title):
            excluded.extend(rids)
    if included:
        out.positive_source = "ref_list"
    # (b) Characteristics-of-included-studies tables.
    if not included:
        for table in root.iter("table-wrap"):
            caption = (
                _text(table.find("caption")) + " " + _text(table.find("label"))
            )
            if _EXCLUDED_TABLE.search(caption):
                excluded.extend(_xref_rids(table, refs))
            elif _INCLUDED_TABLE.search(caption):
                rids = _xref_rids(table, refs)
                included.extend(rids or _table_text_rids(table, root, refs))
        if included:
            out.positive_source = "table"
    # (c) Results sections describing the included studies.
    if len(set(included)) < 5:
        methods_rids: set[str] = set()
        for sec in body.iter("sec"):
            title = _sec_title(sec)
            if _INCLUDED_SECTION.search(title):
                included.extend(_xref_rids(sec, refs))
            elif _METHODS_SECTION.search(title):
                methods_rids.update(_xref_rids(sec, refs))
        included = [r for r in included if r not in methods_rids]
        if included and out.positive_source == "none":
            out.positive_source = "section"
        elif included:
            out.positive_source += "+section"
    excluded_set = set(excluded)
    out.included_rids = [
        r for r in dict.fromkeys(included) if r not in excluded_set
    ]
    out.excluded_rids = list(dict.fromkeys(excluded))

    full_text = _text(body)
    out.methods_text = " ".join(
        _sec_body_text(s)
        for s in body.iter("sec")
        if _METHODS_SECTION.search(_sec_title(s))
    )[:20000]
    out.search_strategy = _find_search_strategy(root)
    match = _PRISMA_PUBMED.search(full_text)
    out.prisma_pubmed = _to_int(match.group(2)) if match else None
    match = _PRISMA_TOTAL.search(full_text)
    if match:
        out.prisma_total = _to_int(match.group(1))
    else:
        match = _PRISMA_TOTAL_2.search(full_text)
        out.prisma_total = _to_int(match.group(3)) if match else None
    candidates = []
    for match in _SEARCH_DATE.finditer(out.methods_text or full_text):
        parsed = dates.parse_date(match.group(4), end_of_period=True)
        if parsed and parsed.year >= 1990:
            candidates.append(parsed)
    if out.pub_date and candidates:
        pub = dates.parse_date(out.pub_date)
        candidates = [c for c in candidates if c <= pub]
    out.search_date = dates.to_iso(max(candidates)) if candidates else None
    return out
