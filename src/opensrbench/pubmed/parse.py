"""PubMed EFetch XML parsing into flat citation dictionaries."""

import datetime
from typing import Any

from lxml import etree

from opensrbench.utils import dates

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}  # yapf: disable


def _text(node: etree._Element | None) -> str:
    """Returns all descendant text of a node, whitespace-collapsed."""
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _parse_pubmed_date(node: etree._Element | None) -> datetime.date | None:
    """Parses <PubDate>/<ArticleDate>/<PubMedPubDate> elements."""
    if node is None:
        return None
    year = node.findtext("Year")
    if not year:
        medline = node.findtext("MedlineDate") or ""
        return dates.parse_date(medline)
    month_text = (node.findtext("Month") or "1").strip()
    if month_text.isdigit():
        month = int(month_text)
    else:
        month = _MONTHS.get(month_text[:3].lower(), 1)
    day_text = (node.findtext("Day") or "1").strip()
    day = int(day_text) if day_text.isdigit() else 1
    return dates.parse_date(f"{int(year):04d}-{month:02d}-{day:02d}")


def parse_article(article: etree._Element) -> dict[str, Any] | None:
    """Parses one <PubmedArticle> element.

    Args:
      article: The element.

    Returns:
      Dict with pmid, title, abstract, abstract_sections, pub_date,
      entrez_date, publication_types, mesh_terms, doi, pmcid, journal and
      language; or None when the element carries no PMID.
    """
    citation = article.find("MedlineCitation")
    if citation is None:
        return None
    pmid = citation.findtext("PMID")
    if not pmid:
        return None
    art = citation.find("Article")
    title = _text(art.find("ArticleTitle")) if art is not None else ""
    sections = []
    if art is not None:
        for abstract_text in art.findall("Abstract/AbstractText"):
            label = abstract_text.get("Label") or ""
            sections.append(
                {
                    "label": label,
                    "category": abstract_text.get("NlmCategory") or "",
                    "text": _text(abstract_text),
                }
            )
    abstract = " ".join(
        (f"{s['label']}: {s['text']}" if s["label"] else s["text"])
        for s in sections
    ).strip()

    pub_dates = []
    if art is not None:
        pub_dates.append(
            _parse_pubmed_date(art.find("Journal/JournalIssue/PubDate"))
        )
        for node in art.findall("ArticleDate"):
            pub_dates.append(_parse_pubmed_date(node))
    pub_dates = [d for d in pub_dates if d]
    entrez = None
    pubmed_data = article.find("PubmedData")
    ids: dict[str, str] = {}
    if pubmed_data is not None:
        for node in pubmed_data.findall("History/PubMedPubDate"):
            if node.get("PubStatus") in ("entrez", "pubmed"):
                entrez = entrez or _parse_pubmed_date(node)
        for node in pubmed_data.findall("ArticleIdList/ArticleId"):
            if node.text:
                ids[node.get("IdType", "")] = node.text.strip()
    pub_types = [
        _text(n)
        for n in citation.findall("Article/PublicationTypeList/PublicationType")
    ]
    mesh = [
        _text(n)
        for n in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
    ]
    return {
        "pmid": pmid.strip(),
        "title": title,
        "abstract": abstract,
        "abstract_sections": sections,
        "pub_date": dates.to_iso(min(pub_dates) if pub_dates else entrez),
        "entrez_date": dates.to_iso(entrez),
        "publication_types": pub_types,
        "mesh_terms": mesh,
        "doi": ids.get("doi"),
        "pmcid": ids.get("pmc"),
        "journal": _text(art.find("Journal/Title")) if art is not None else "",
        "language": (art.findtext("Language") or "") if art is not None else "",
    }


def parse_efetch_xml(payload: bytes) -> list[dict[str, Any]]:
    """Parses an EFetch `retmode=xml` response into citation dicts."""
    parser = etree.XMLParser(
        recover=True, huge_tree=True, resolve_entities=False, no_network=True
    )
    root = etree.fromstring(payload, parser=parser)
    if root is None:
        return []
    out = []
    for article in root.iter("PubmedArticle"):
        parsed = parse_article(article)
        if parsed:
            out.append(parsed)
    return out


_OBJECTIVE_LABELS = (
    "objective",
    "objectives",
    "aim",
    "aims",
    "purpose",
    "goal",
    "goals",
    "research question",
)
_CRITERIA_LABELS = (
    "selection criteria",
    "eligibility criteria",
    "inclusion criteria",
    "study selection",
    "eligibility",
    "criteria",
    "study eligibility criteria",
    "inclusion and exclusion criteria",
)


def review_fields_from_sections(
    sections: list[dict[str, str]],
) -> dict[str, str]:
    """Extracts objective and selection criteria from a structured abstract.

    Cochrane and most journal SR abstracts label their sections, e.g.
    "OBJECTIVES" and "SELECTION CRITERIA", which gives protocol text with no
    human annotation.

    Args:
      sections: `abstract_sections` from `parse_article`.

    Returns:
      Dict with "objective" and "selection_criteria" (possibly empty).
    """
    objective, criteria, methods = [], [], []
    for section in sections:
        label = section["label"].strip().lower()
        if any(label.startswith(x) for x in _OBJECTIVE_LABELS):
            objective.append(section["text"])
        elif any(x in label for x in _CRITERIA_LABELS):
            criteria.append(section["text"])
        elif section["category"].upper() == "METHODS" or "method" in label:
            methods.append(section["text"])
    if not criteria and methods:
        criteria = methods
    return {
        "objective": " ".join(objective).strip(),
        "selection_criteria": " ".join(criteria).strip(),
    }
