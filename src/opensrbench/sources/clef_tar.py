"""CLEF eHealth TAR 2017-2019 (Kanoulas et al.).

Cochrane DTA / intervention / prognosis / qualitative reviews with the full
Boolean-search pool, abstract-level qrels (label_abs) and content-level
qrels (label_final), plus protocol XML with objectives and criteria.

Topics recur across years; for every Cochrane topic the most recent year's
qrels are kept (2019 > 2018 > 2017) because later releases fixed errors.
"""

import collections
import glob
import os
import re

from lxml import etree
import pandas as pd

from opensrbench import schema
from opensrbench.sources import base
from opensrbench.text import review_type
from opensrbench.utils import dates

_TOPIC_ID = re.compile(r"CD\d{6}")
_PLACEHOLDER_PMIDS = {"99999999"}
_OBJECTIVE_TAGS = {"objective", "objectives"}
_CRITERIA_ORDER = (
    ("type_of_study", "Types of studies"),
    ("typesofstudies", "Types of studies"),
    ("participants", "Participants"),
    ("typesofparticipants", "Participants"),
    ("index_tests", "Index tests"),
    ("typesofinterventions", "Interventions"),
    ("target_conditions", "Target conditions"),
    ("reference_standards", "Reference standards"),
    ("typesofoutcomemeasures", "Outcomes"),
)


def _year_of(path: str) -> int:
    match = re.search(r"(20\d\d)-TAR", path)
    return int(match.group(1)) if match else 0


def _type_hint(path: str) -> str:
    for hint in ("DTA", "Intervention", "Prognosis", "Qualitative"):
        if f"/{hint}/" in path:
            return hint
    return "DTA"  # 2017 and 2018 are DTA-only.


def parse_topic_file(path: str) -> dict:
    """Parses a TAR topic file (Topic/Title/Objective/Query/Pids blocks)."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    fields: dict[str, list[str]] = collections.defaultdict(list)
    key = None
    for line in text.splitlines():
        match = re.match(
            r"^\s*(Topic|Title|Objectives?|Query|Pids)\s*:\s*(.*)$", line
        )
        if match:
            key = match.group(1).lower()
            key = "objective" if key.startswith("objective") else key
            if match.group(2).strip():
                fields[key].append(match.group(2).strip())
        elif key and line.strip():
            fields[key].append(line.strip())
    topic = " ".join(fields.get("topic", [])).strip()
    return {
        "topic": topic,
        "title": " ".join(fields.get("title", [])).strip(),
        "objective": " ".join(fields.get("objective", [])).strip(),
        "query": "\n".join(fields.get("query", [])),
        "pids": [p for p in fields.get("pids", []) if p.isdigit()],
    }


def parse_protocol_file(path: str) -> dict:
    """Parses protocol XML into objective + ordered criteria text."""
    parser = etree.XMLParser(
        recover=True, resolve_entities=False, no_network=True
    )
    try:
        root = etree.parse(path, parser=parser).getroot()
    except (etree.XMLSyntaxError, OSError):
        return {}
    if root is None:
        return {}
    sections = {}
    for child in root:
        if isinstance(child.tag, str):
            sections[child.tag.lower()] = " ".join(
                "".join(child.itertext()).split()
            )
    objective = " ".join(v for k, v in sections.items() if k in _OBJECTIVE_TAGS)
    parts = [
        f"{label}: {sections[tag]}"
        for tag, label in _CRITERIA_ORDER
        if sections.get(tag)
    ]
    return {"objective": objective, "selection_criteria": "\n".join(parts)}


def parse_qrels(path: str) -> pd.DataFrame:
    """Reads a TREC-style qrels file -> topic, pmid, rel."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            cols = line.split()
            if len(cols) >= 4 and _TOPIC_ID.fullmatch(cols[0]):
                if cols[2].isdigit() and cols[2] not in _PLACEHOLDER_PMIDS:
                    rows.append((cols[0], cols[2], int(cols[3]) > 0))
    return pd.DataFrame(rows, columns=["topic", "pmid", "rel"])


@base.register
class ClefTarAdapter(base.SourceAdapter):
    """Adapter for github.com/CLEF-TAR/tar."""

    name = "clef_tar"

    def download(self) -> None:
        self.git_clone(self.cfg["url"])

    def _qrel_files(self, repo: str) -> list[tuple[str, str, int]]:
        """Returns (path, level, year) for every abs/content qrels file."""
        out = []
        for path in glob.glob(
            os.path.join(repo, "20*-TAR", "**", "*"), recursive=True
        ):
            name = os.path.basename(path).lower()
            if (
                not os.path.isfile(path)
                or "participant" in path.lower()
                or "qrel" not in name
            ):
                continue
            if any(
                x in name
                for x in ("binary", "combined", "all.", "test.qrels", ".sh")
            ):
                continue
            if "abs" in name:
                level = "abs"
            elif "content" in name or "doc" in name:
                level = "content"
            else:
                continue
            year = _year_of(path)
            if year in {
                int(y) for y in self.cfg.get("years", (2017, 2018, 2019))
            }:
                out.append((path, level, year))
        return out

    def ingest(self):
        repo = os.path.join(self.raw_dir, "repo")
        qrels = []
        for path, level, year in self._qrel_files(repo):
            df = parse_qrels(path)
            df["level"], df["year"] = level, year
            df["hint"] = _type_hint(path)
            qrels.append(df)
        qrels_df = pd.concat(qrels, ignore_index=True)
        latest = qrels_df.groupby("topic")["year"].transform("max")
        qrels_df = qrels_df[qrels_df["year"] == latest]

        # Topic metadata from every year; later years override earlier ones.
        meta: dict[str, dict] = collections.defaultdict(dict)
        topic_files = sorted(
            glob.glob(
                os.path.join(repo, "20*-TAR", "**", "topic*", "*"),
                recursive=True,
            ),
            key=_year_of,
        )
        for path in topic_files:
            if os.path.isfile(path):
                parsed = parse_topic_file(path)
                if _TOPIC_ID.fullmatch(parsed["topic"] or ""):
                    for k, v in parsed.items():
                        if v:
                            meta[parsed["topic"]][k] = v
        protocol_files = sorted(
            glob.glob(
                os.path.join(repo, "20*-TAR", "**", "protoc*", "*"),
                recursive=True,
            ),
            key=_year_of,
        )
        for path in protocol_files:
            match = _TOPIC_ID.search(os.path.basename(path))
            if match and os.path.isfile(path):
                parsed = parse_protocol_file(path)
                for k, v in parsed.items():
                    if v:
                        meta[match.group(0)][k] = v
        pubdates = {}
        for path in glob.glob(
            os.path.join(repo, "**", "pubdates.txt"), recursive=True
        ):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    cols = line.split()
                    if len(cols) >= 3:
                        pubdates[cols[0]] = dates.to_iso(
                            dates.parse_date(
                                f"{cols[2][:4]}-{cols[2][4:6]}-"
                                f"{cols[2][6:8]}"
                            )
                        )

        reviews, records = [], []
        for topic, group in qrels_df.groupby("topic"):
            pivot = group.pivot_table(
                index="pmid", columns="level", values="rel", aggfunc="max"
            )
            pivot = pivot.reindex(columns=["abs", "content"]).fillna(False)
            info = meta.get(topic, {})
            review_id = f"{self.name}:{topic}"
            hint = group["hint"].iloc[0]
            reviews.append(
                {
                    "review_id": review_id,
                    "source": self.name,
                    "review_key": f"cochrane:{topic}",
                    "title": info.get("title", ""),
                    "objective": info.get("objective", ""),
                    "selection_criteria": info.get("selection_criteria", ""),
                    "review_type": review_type.classify(
                        info.get("title", ""),
                        info.get("objective", ""),
                        hint=hint,
                    ),
                    "search_date": pubdates.get(topic),
                    "pool_complete": True,
                    "has_abs_labels": True,
                    "doi": None,
                    "extra": {
                        "year": int(group["year"].iloc[0]),
                        "hint": hint,
                        "query": info.get("query", ""),
                    },
                }
            )
            for pmid, row in pivot.iterrows():
                final = bool(row["content"])
                records.append(
                    {
                        "review_id": review_id,
                        "record_id": schema.pmid_record_id(pmid),
                        "pmid": pmid,
                        "label_final": int(final),
                        "label_abs": int(bool(row["abs"]) or final),
                    }
                )
        return (
            schema.make_reviews_frame(reviews),
            schema.make_records_frame(records),
        )
