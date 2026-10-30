"""TITAN-SR reproducibility-package manifests (CC BY 4.0).

Ingesting TITAN-SR's own PMC and Track B training labels guarantees our
training corpus is a superset of theirs, and its `stratified_split_v2`
assignments let us hold out *their* test reviews so the comparison on their
benchmark is leakage-free.

Notes on the data (see the package's data/README.md):
* PMC: 7,723 reviews, ~45 candidates/review with ~40% positives, i.e. the
  negatives are cited-but-not-included references, not a search pool. The
  pool-augmentation stage therefore adds realistic retrieved negatives.
* Track B: 3,525 reviews identified by PDF filename.
* External holdout: 22 Cochrane reviews with candidate *titles only*.
"""

import json
import os
import re

import pandas as pd

from opensrbench import schema
from opensrbench.sources import base
from opensrbench.text import review_type

_CD = re.compile(r"CD\d{6}")


def _key_for(source: str, local_id: str) -> str:
    if source == "pmc":
        return f"pmc:{local_id.upper()}"
    if source == "chan":
        match = _CD.search(local_id)
        return f"cochrane:{match.group(0)}" if match else f"url:{local_id}"
    return f"trackb:{local_id}"


def titan_split_keys(split_json_path: str) -> pd.DataFrame:
    """Maps TITAN-SR split assignments to our review keys.

    Args:
      split_json_path: data/splits/stratified_split_v2.json.

    Returns:
      Frame with review_key, titan_split, titan_source.
    """
    with open(split_json_path, encoding="utf-8") as fh:
        assignments = json.load(fh)["assignments"]
    rows = []
    for prefixed, info in assignments.items():
        prefix, local = prefixed.split("_", 1)
        source = {"pmc": "pmc", "tb": "trackb", "chan": "chan"}.get(
            prefix, prefix
        )
        rows.append(
            {
                "review_key": _key_for(source, local),
                "titan_split": info.get("split"),
                "titan_source": source,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("review_key")


@base.register
class TitanSrAdapter(base.SourceAdapter):
    """Adapter for github.com/InvestmentMDideas/TITAN-SR manifests."""

    name = "titan_sr"

    def download(self) -> None:
        self.git_clone(self.cfg["url"])

    def ingest(self):
        repo = os.path.join(self.raw_dir, "repo", "data")
        criteria = pd.read_csv(
            os.path.join(repo, "manifests", "criteria.csv.gz"), dtype=str
        )
        criteria_of = dict(
            zip(
                criteria["source"] + ":" + criteria["review_id"],
                criteria["eligibility_criteria"],
            )
        )
        splits = titan_split_keys(
            os.path.join(repo, "splits", "stratified_split_v2.json")
        )
        split_of = dict(zip(splits["review_key"], splits["titan_split"]))
        os.makedirs(self.paths.interim(self.name), exist_ok=True)
        splits.to_parquet(
            os.path.join(
                self.paths.interim(self.name), "titan_split_keys.parquet"
            ),
            index=False,
        )

        reviews, records = {}, []
        for manifest, source in (
            ("pmc_train_labels.csv.gz", "pmc"),
            ("trackb_train_labels.csv.gz", "trackb"),
        ):
            labels = pd.read_csv(
                os.path.join(repo, "manifests", manifest), dtype=str
            )
            for local_id, group in labels.groupby("review_id"):
                review_id = f"{self.name}:{source}:{local_id}"
                key = _key_for(source, local_id)
                text = criteria_of.get(f"{source}:{local_id}", "") or ""
                reviews[review_id] = {
                    "review_id": review_id,
                    "source": self.name,
                    "review_key": key,
                    "selection_criteria": text,
                    "review_type": review_type.classify("", "", text),
                    "pool_complete": False,
                    "has_abs_labels": False,
                    "pmcid": local_id if source == "pmc" else None,
                    "extra": {
                        "titan_source": source,
                        "titan_split": split_of.get(key),
                    },
                }
                for pmid, label in zip(group["candidate_pmid"], group["label"]):
                    pmid = schema.normalize_pmid(pmid)
                    if pmid:
                        records.append(
                            {
                                "review_id": review_id,
                                "record_id": schema.pmid_record_id(pmid),
                                "pmid": pmid,
                                "label_final": int(float(label) > 0),
                                "retrieval_source": "cited",
                            }
                        )
        if self.cfg.get("include_external_holdout", True):
            ext = pd.read_csv(
                os.path.join(
                    repo, "manifests", "external_holdout_labels.csv.gz"
                ),
                dtype=str,
                keep_default_na=False,
            )
            ext_criteria = pd.read_csv(
                os.path.join(repo, "manifests", "external_criteria.csv.gz"),
                dtype=str,
            )
            ext_criteria = dict(
                zip(
                    ext_criteria["review_id"],
                    ext_criteria["selection_criteria"],
                )
            )
            for cd, group in ext.groupby("review_id"):
                review_id = f"{self.name}:external:{cd}"
                reviews[review_id] = {
                    "review_id": review_id,
                    "source": self.name,
                    "review_key": f"cochrane:{cd}",
                    "selection_criteria": ext_criteria.get(cd, ""),
                    "review_type": "intervention_ma",
                    "pool_complete": True,
                    "has_abs_labels": False,
                    "extra": {
                        "titan_source": "external",
                        "titan_split": "external",
                        "text_mode": "title_only",
                    },
                }
                for title, label, year in zip(
                    group["title"], group["label"], group["publication_year"]
                ):
                    records.append(
                        {
                            "review_id": review_id,
                            "record_id": schema.text_record_id(
                                self.name, title, ""
                            ),
                            "title": title,
                            "abstract": "",
                            "pub_date": f"{year[:4]}-01-01" if year else None,
                            "label_final": int(float(label or 0) > 0),
                        }
                    )
        return (
            schema.make_reviews_frame(reviews.values()),
            schema.make_records_frame(records),
        )
