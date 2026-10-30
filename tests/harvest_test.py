"""Tests for JATS parsing, CLEF parsing, PubMed parsing and query synthesis."""

import os

from opensrbench.cohort import query
from opensrbench.harvest import jats
from opensrbench.pubmed import parse
from opensrbench.sources import clef_tar

_JATS = b"""<?xml version="1.0"?>
<article><front><article-meta>
<article-id pub-id-type="pmid">111</article-id>
<article-id pub-id-type="doi">10.1/ABC</article-id>
<title-group><article-title>Drug X for disease Y: a systematic review and
meta-analysis</article-title></title-group>
<pub-date pub-type="epub"><day>10</day><month>3</month><year>2025</year></pub-date>
<abstract><sec><title>Objective</title><p>To assess drug X in disease Y.</p>
</sec></abstract></article-meta></front>
<body>
<sec><title>Methods</title>
 <sec><title>Search strategy</title><p>We searched PubMed from inception to
 31 August 2024. PubMed: ("disease Y"[MeSH Terms] OR y[tiab]) AND (x[tiab] OR
 "drug x"[tiab])</p></sec>
 <sec><title>Eligibility criteria</title><p>We included randomised trials of
 adults with disease Y receiving drug X. We excluded animal studies.</p></sec>
 <sec><title>Statistical analysis</title><p>Random effects
 <xref ref-type="bibr" rid="r9">9</xref>.</p></sec>
</sec>
<sec><title>Results</title><p>The search identified 1,234 records were
 identified.</p>
<table-wrap><caption><p>Characteristics of the included studies</p></caption>
<table><tr><td><xref ref-type="bibr" rid="r1">1</xref>&#8211;<xref
 ref-type="bibr" rid="r4">4</xref></td></tr><tr><td><xref ref-type="bibr"
 rid="r6">6</xref></td></tr></table></table-wrap>
<table-wrap><caption><p>Studies excluded after full-text review</p></caption>
<table><tr><td><xref ref-type="bibr" rid="r7">7</xref></td></tr></table>
</table-wrap></sec></body>
<back><ref-list>
<ref id="r1"><element-citation><article-title>T1</article-title><pub-id pub-id-type="pmid">1</pub-id></element-citation></ref>
<ref id="r2"><element-citation><article-title>T2</article-title><pub-id pub-id-type="pmid">2</pub-id></element-citation></ref>
<ref id="r3"><element-citation><article-title>T3</article-title><pub-id pub-id-type="doi">10.1/three</pub-id></element-citation></ref>
<ref id="r4"><element-citation><article-title>T4</article-title><pub-id pub-id-type="pmid">4</pub-id></element-citation></ref>
<ref id="r5"><element-citation><article-title>T5</article-title></element-citation></ref>
<ref id="r6"><element-citation><article-title>T6</article-title><pub-id pub-id-type="pmid">6</pub-id></element-citation></ref>
<ref id="r7"><element-citation><article-title>T7</article-title><pub-id pub-id-type="pmid">7</pub-id></element-citation></ref>
<ref id="r9"><element-citation><article-title>T9</article-title><pub-id pub-id-type="pmid">9</pub-id></element-citation></ref>
</ref-list></back></article>"""


def test_parse_jats_extracts_protocol_fields():
    parsed = jats.parse_jats(_JATS, "PMC1")
    assert parsed.pmid == "111" and parsed.doi == "10.1/abc"
    assert parsed.pub_date == "2025-03-10"
    assert parsed.objective == "To assess drug X in disease Y."
    assert "randomised trials" in parsed.selection_criteria
    # Range 1-4 expanded via ref-list order, plus 6; r9 (methods) excluded.
    assert parsed.included_rids == ["r1", "r2", "r3", "r4", "r6"]
    assert parsed.excluded_rids == ["r7"]
    assert parsed.positive_source == "table"
    assert parsed.search_date == "2024-08-31"
    assert parsed.prisma_total == 1234
    assert "[MeSH Terms]" in parsed.search_strategy


def test_normalize_strategy_wraps_concatenated_queries():
    out = jats.normalize_strategy(
        "Table: (a[tiab]) AND (b[tiab])(a[tiab]) AND (c[tiab])"
    )
    assert out == "((a[tiab]) AND (b[tiab])) OR ((a[tiab]) AND (c[tiab]))"


def test_clef_parsers(tmp_path):
    qrels = tmp_path / "q.qrels"
    qrels.write_text(
        "CD000001 0 123 1\nCD000001 0 99999999 1\nbad line\n"
        "CD000001 0 456 0\n"
    )
    df = clef_tar.parse_qrels(str(qrels))
    assert df["pmid"].tolist() == ["123", "456"]
    assert df["rel"].tolist() == [True, False]
    topic = tmp_path / "CD000001"
    topic.write_text(
        "Topic: CD000001\n\nTitle: Test review\n\nObjective: \n"
        "Assess things.\n\nQuery:\nexp X/\n\nPids:\n  123\n  456\n"
    )
    parsed = clef_tar.parse_topic_file(str(topic))
    assert parsed["topic"] == "CD000001"
    assert parsed["title"] == "Test review"
    assert parsed["objective"] == "Assess things."
    assert parsed["pids"] == ["123", "456"]


def test_pubmed_parse_and_review_fields():
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
    <PMID>42</PMID><Article><Journal><JournalIssue><PubDate><Year>2020</Year>
    <Month>Feb</Month></PubDate></JournalIssue><Title>J</Title></Journal>
    <ArticleTitle>A title</ArticleTitle><Abstract>
    <AbstractText Label="OBJECTIVES" NlmCategory="OBJECTIVE">Obj.</AbstractText>
    <AbstractText Label="SELECTION CRITERIA" NlmCategory="METHODS">RCTs.</AbstractText>
    </Abstract><PublicationTypeList><PublicationType>Review</PublicationType>
    </PublicationTypeList></Article></MedlineCitation><PubmedData>
    <ArticleIdList><ArticleId IdType="doi">10.1/X</ArticleId></ArticleIdList>
    </PubmedData></PubmedArticle></PubmedArticleSet>"""
    (record,) = parse.parse_efetch_xml(xml)
    assert record["pmid"] == "42" and record["pub_date"] == "2020-02-01"
    assert record["abstract"] == "OBJECTIVES: Obj. SELECTION CRITERIA: RCTs."
    fields = parse.review_fields_from_sections(record["abstract_sections"])
    assert fields == {"objective": "Obj.", "selection_criteria": "RCTs."}


def test_query_generator_uses_only_review_text():
    out = query.HeuristicQueryGenerator().queries(
        "SGLT2 inhibitors in heart failure with preserved ejection fraction: "
        "a systematic review and meta-analysis"
    )
    assert "SGLT2 inhibitors" in out["strict"]
    assert "systematic" not in out["strict"].lower()
    assert " AND " in out["strict"] and " OR " in out["broad"]
