"""SYNERGY (De Bruin et al., 2023): 26 fully labelled screening datasets.

Each review lists every screened record (PMID / DOI / OpenAlex id) with the
final inclusion label, and `datasets.toml` gives the review DOI and
eligibility criteria. CC0. https://github.com/asreview/synergy-dataset
"""

import glob
import os
import tomllib

import pandas as pd

from opensrbench import schema
from opensrbench.pubmed import openalex
from opensrbench.sources import base
from opensrbench.text import review_type

# Software-engineering SLRs; dropped when biomedical_only is set.
_NON_BIOMEDICAL = {
    "Hall_2012",
    "Radjenovic_2013",
    "Wahono_2015",
    "Kitchenham_2010",
}


@base.register
class SynergyAdapter(base.SourceAdapter):
    """Adapter for github.com/asreview/synergy-dataset."""

    name = "synergy"

    def download(self) -> None:
        self.git_clone(self.cfg["url"])

    def ingest(self):
        repo = os.path.join(self.raw_dir, "repo")
        with open(os.path.join(repo, "datasets.toml"), "rb") as fh:
            meta = {d["key"]: d for d in tomllib.load(fh)["datasets"]}
        reviews, records = [], []
        for path in sorted(
            glob.glob(os.path.join(repo, "datasets", "*", "*_ids.csv"))
        ):
            key = os.path.basename(path)[: -len("_ids.csv")]
            if self.cfg.get("biomedical_only", True) and key in _NON_BIOMEDICAL:
                continue
            info = meta.get(key, {})
            publication = info.get("publication", {})
            review_id = f"{self.name}:{key}"
            criteria = publication.get("eligibility_criteria", "") or ""
            if key.startswith("Cohen_2006_"):
                # Same reviews as the cohen2006 source; share its key so the
                # build stage keeps one copy (cohen2006 has abstract labels).
                review_key = "derp:" + key[len("Cohen_2006_") :].lower()
            elif publication.get("doi"):
                review_key = f"doi:{publication['doi'].lower()}"
            else:
                review_key = review_id
            reviews.append(
                {
                    "review_id": review_id,
                    "source": self.name,
                    "review_key": review_key,
                    "title": (
                        ""
                    ),  # Filled from the review DOI during enrichment.
                    "selection_criteria": criteria,
                    "review_type": review_type.classify("", "", criteria),
                    "pool_complete": True,
                    "has_abs_labels": False,
                    "domain": (
                        "other" if key in _NON_BIOMEDICAL else "biomedical"
                    ),
                    "doi": publication.get("doi"),
                    "extra": {
                        "synergy_key": key,
                        "active": info.get("active", True),
                    },
                }
            )
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            for _, row in df.iterrows():
                pmid = schema.normalize_pmid(row.get("pmid"))
                oa_id = openalex.normalize_openalex_id(row.get("openalex_id"))
                doi = str(row.get("doi") or "").lower().split("doi.org/")[-1]
                if pmid:
                    record_id = schema.pmid_record_id(pmid)
                elif doi.startswith("10."):
                    # Resolved to a PMID by PubMed during `enrich`; OpenAlex
                    # (by DOI) is the fallback.
                    record_id = f"doi:{doi}"
                elif oa_id:
                    record_id = f"openalex:{oa_id}"
                else:
                    continue  # No resolvable identifier: cannot get text.
                label = int(float(row.get("label_included") or 0) > 0)
                records.append(
                    {
                        "review_id": review_id,
                        "record_id": record_id,
                        "pmid": pmid,
                        "label_final": label,
                    }
                )
        return (
            schema.make_reviews_frame(reviews),
            schema.make_records_frame(records),
        )
