"""Smaller legacy collections: SWIFT (Howard 2016), Wallace 2010, SIGIR 2017.

* SWIFT-Review (Howard et al., Syst Rev 2016; CC0): four OHAT environmental
  health reviews and one CAMARADES pre-clinical review.
* Wallace et al. (KDD 2010): proton beam, micro-nutrients and COPD reviews
  with local titles/abstracts (no PMIDs).
* SIGIR 2017 SysRev collection (Scells et al.): 93 Cochrane reviews whose
  citations are the included studies plus the studies excluded at full text.
  The pool is incomplete (no search negatives), so the pool-augmentation
  stage adds realistic retrieved negatives.
"""

import json
import os
import re

import pandas as pd

from opensrbench import schema
from opensrbench.sources import base
from opensrbench.text import review_type

_CD = re.compile(r"CD\d{6}")

_SWIFT_REVIEWS = {
    "BPA": (
        "Bisphenol A (BPA) exposure and obesity",
        "To evaluate whether exposure to bisphenol A is associated with "
        "obesity or related metabolic outcomes in humans or animals.",
    ),
    "PFOS-PFOA": (
        "PFOA or PFOS exposure and immunotoxicity",
        "To evaluate whether exposure to perfluorooctanoic acid "
        "(PFOA) or perfluorooctane sulfonate (PFOS) is associated "
        "with immunotoxicity or immune-related health effects.",
    ),
    "Transgenerational": (
        "Transgenerational inheritance of health effects",
        "To identify evidence that environmental exposures "
        "cause health effects inherited across "
        "generations (transgenerational inheritance).",
    ),
    "Fluoride": (
        "Fluoride exposure and neurobehavioural effects",
        "To evaluate the effects of fluoride exposure on learning, "
        "memory and other neurobehavioural outcomes in animals.",
    ),
    "Neuropain": (
        "Animal models of neuropathic pain",
        "To identify in vivo animal studies that model neuropathic "
        "pain and report pain-related behavioural outcomes.",
    ),
}

_WALLACE_REVIEWS = {
    "proton_beam": (
        "proton_labeled_features",
        "Charged particle (proton beam) radiotherapy for cancer",
        "To compare the benefits and harms of charged-particle "
        "radiation therapy versus other treatments for cancer.",
    ),
    "micro_nutrients": (
        "micro_labeled_features_only",
        "Micronutrients and health outcomes",
        "To assess the effect of micronutrient intake or "
        "supplementation on clinical outcomes.",
    ),
    "copd": (
        "copd_labeled_terms_only",
        "Interventions for chronic obstructive pulmonary disease (COPD)",
        "To assess the effectiveness of interventions for patients with "
        "chronic obstructive pulmonary disease.",
    ),
}


def _templated_review(
    source: str, key: str, title: str, objective: str, **extra
) -> dict:
    """Review row for collections that ship no criteria text."""
    return {
        "review_id": f"{source}:{key}",
        "source": source,
        "review_key": f"{source}:{key.lower()}",
        "title": title,
        "objective": objective,
        "selection_criteria": f"Include: {objective[3:].strip()}",
        "review_type": review_type.classify(title, objective),
        "pool_complete": True,
        "has_abs_labels": False,
        "extra": {"criteria_quality": "templated", **extra},
    }


@base.register
class Swift2016Adapter(base.SourceAdapter):
    """Howard et al. (2016) SWIFT-Review supplementary datasets."""

    name = "swift2016"

    def download(self) -> None:
        for key, url in self.cfg["urls"].items():
            self.fetch(url, f"{key}.xlsx")

    def ingest(self):
        reviews, records = [], []
        sheets = {}
        for key in self.cfg["urls"]:
            sheets.update(
                pd.read_excel(
                    os.path.join(self.raw_dir, f"{key}.xlsx"),
                    sheet_name=None,
                    dtype=str,
                )
            )
        for sheet, df in sheets.items():
            if sheet not in _SWIFT_REVIEWS:
                continue
            title, objective = _SWIFT_REVIEWS[sheet]
            reviews.append(
                _templated_review(self.name, sheet, title, objective)
            )
            review_id = f"{self.name}:{sheet}"
            status_col = next(
                c for c in ("Status", "Included", "Label") if c in df.columns
            )
            for _, row in df.iterrows():
                label = int(
                    str(row[status_col]).strip().lower()
                    not in ("excluded", "nan", "", "0")
                )
                pmid = schema.normalize_pmid(row.get("PMID"))
                if pmid:
                    record_id = schema.pmid_record_id(pmid)
                    title_text, abstract = None, None
                else:
                    title_text = row.get("Title", "")
                    abstract = row.get("Abstract", "")
                    record_id = schema.text_record_id(
                        self.name, title_text, abstract
                    )
                records.append(
                    {
                        "review_id": review_id,
                        "record_id": record_id,
                        "pmid": pmid,
                        "title": title_text,
                        "abstract": abstract,
                        "label_final": label,
                    }
                )
        return (
            schema.make_reviews_frame(reviews),
            schema.make_records_frame(records),
        )


@base.register
class Sas2010Adapter(base.SourceAdapter):
    """Wallace et al. (2010) citation-screening datasets."""

    name = "sas2010"

    def download(self) -> None:
        self.git_clone(self.cfg["url"])

    def ingest(self):
        repo = os.path.join(
            self.raw_dir, "repo", "modeling", "curious_snake", "data"
        )
        reviews, records = [], []
        for key, (labels_file, title, objective) in _WALLACE_REVIEWS.items():
            folder = os.path.join(repo, key)
            if not os.path.isdir(folder):
                continue
            reviews.append(_templated_review(self.name, key, title, objective))
            with open(
                os.path.join(folder, labels_file),
                encoding="utf-8",
                errors="replace",
            ) as fh:
                labels = {}
                for line in fh:
                    cols = line.split()
                    if len(cols) >= 2:
                        labels[cols[0]] = int(cols[1] == "1")
            for doc_id, label in labels.items():

                def read(sub, doc=doc_id):
                    path = os.path.join(folder, sub, doc)
                    if not os.path.exists(path):
                        return ""
                    with open(path, encoding="utf-8", errors="replace") as fh:
                        return fh.read().strip()

                title_text, abstract = read("Titles"), read("Abstracts")
                if not title_text and not abstract:
                    continue
                records.append(
                    {
                        "review_id": f"{self.name}:{key}",
                        "record_id": schema.text_record_id(
                            self.name, title_text, abstract
                        ),
                        "title": title_text,
                        "abstract": abstract,
                        "label_final": label,
                    }
                )
        return (
            schema.make_reviews_frame(reviews),
            schema.make_records_frame(records),
        )


@base.register
class Sigir2017Adapter(base.SourceAdapter):
    """Scells et al. (SIGIR 2017) systematic-review query collection."""

    name = "sigir2017"

    def download(self) -> None:
        self.git_clone(self.cfg["url"])

    def ingest(self):
        repo = os.path.join(self.raw_dir, "repo")
        with open(
            os.path.join(repo, "systematic_reviews.json"), encoding="utf-8"
        ) as fh:
            urls = {r["id"]: r["url"] for r in json.load(fh)}
        with open(os.path.join(repo, "citations.json"), encoding="utf-8") as fh:
            citations = json.load(fh)
        reviews, records = {}, []
        for citation in citations:
            url = urls.get(citation["document_id"], "")
            match = _CD.search(url)
            if not match:
                continue
            cd = match.group(0)
            review_id = f"{self.name}:{cd}"
            doi_match = re.search(r"10\.1002/14651858\.CD\d{6}(\.pub\d+)?", url)
            reviews.setdefault(
                review_id,
                {
                    "review_id": review_id,
                    "source": self.name,
                    "review_key": f"cochrane:{cd}",
                    "pool_complete": False,
                    "has_abs_labels": True,
                    "doi": doi_match.group(0) if doi_match else None,
                    "extra": {"url": url},
                },
            )
            pmid = schema.normalize_pmid(citation.get("url"))
            if not pmid:
                continue
            included = bool(citation.get("included"))
            records.append(
                {
                    "review_id": review_id,
                    "record_id": schema.pmid_record_id(pmid),
                    "pmid": pmid,
                    "label_final": int(included),
                    # Non-included citations are the studies excluded at full
                    # text, i.e. they passed title/abstract screening.
                    "label_abs": 1,
                    "retrieval_source": "cited",
                }
            )
        return (
            schema.make_reviews_frame(reviews.values()),
            schema.make_records_frame(records),
        )
