"""Chan, He, Leung & Verspoor (2024) Cochrane screening data (Mendeley V1).

Title/abstract screening data for thousands of Cochrane reviews with the
review's background, objective and selection criteria inline. Labels:
1 = final inclusion, 0.5 = excluded at full text (passed T/A screening),
0 = estimated T/A exclusion. Pools are partial (median ~48 records per
review), so the pool-augmentation stage adds retrieved negatives. Records
carry no PMIDs, so cross-source matching and decontamination rely on the
title/abstract fingerprints.

DOI 10.17632/7sgmg89zb6.1, CC BY 4.0. Associated article:
Res Synth Methods (2025), doi:10.1017/rsm.2025.1.
"""

import collections
import logging
import os
import re
import zipfile

import pandas as pd

from opensrbench import schema
from opensrbench.sources import base
from opensrbench.text import review_type

_CD = re.compile(r"CD\d{6}")
_ZIP_NAME = "20240827_Systematic_Review_Data.zip"


def _cochrane_doi(url: str) -> str | None:
    """10.1002/14651858.CDxxxxxx.pubN from a Cochrane Library URL."""
    match = re.search(r"10\.1002/14651858\.CD\d{6}(\.pub\d+)?", url)
    return match.group(0) if match else None


@base.register
class Chan2024Adapter(base.SourceAdapter):
    """Adapter for the Chan et al. Mendeley archive."""

    name = "chan2024"

    def download(self) -> None:
        self.fetch(self.cfg["url"], _ZIP_NAME, sha256=self.cfg.get("sha256"))

    def _iter_rows(self, chunk_rows: int = 50_000):
        path = os.path.join(self.raw_dir, _ZIP_NAME)
        with zipfile.ZipFile(path) as archive:
            for member in self.cfg.get("members", []):
                logging.info("[chan2024] reading %s", member)
                with archive.open(member) as fh:
                    for chunk in pd.read_csv(
                        fh,
                        chunksize=chunk_rows,
                        dtype=str,
                        keep_default_na=False,
                    ):
                        chunk["_member"] = member
                        yield chunk

    def ingest(self):
        reviews: dict[str, dict] = {}
        records = []
        member_of = collections.defaultdict(set)
        for chunk in self._iter_rows():
            for url, group in chunk.groupby("Review_URL", sort=False):
                match = _CD.search(url)
                local = match.group(0) if match else url
                review_id = f"{self.name}:{local}"
                member_of[review_id].update(group["_member"].unique())
                if review_id not in reviews:
                    first = group.iloc[0]
                    reviews[review_id] = {
                        "review_id": review_id,
                        "source": self.name,
                        "review_key": (
                            f"cochrane:{local}" if match else f"url:{url}"
                        ),
                        "title": first["Review_Title"],
                        "objective": first["Objective"],
                        "selection_criteria": first["Selection_criteria"],
                        "review_type": review_type.classify(
                            first["Review_Title"],
                            first["Objective"],
                            first["Selection_criteria"],
                            hint="intervention",
                        ),
                        # Median ~48 records/review: includes, full-text
                        # exclusions and estimated T/A exclusions, i.e. not
                        # a full search pool -> augmented in `pools`.
                        "pool_complete": False,
                        "has_abs_labels": True,
                        "doi": _cochrane_doi(url),
                        "extra": {"url": url},
                    }
                label = pd.to_numeric(group["label"], errors="coerce").fillna(0)
                frame = pd.DataFrame(
                    {
                        "review_id": review_id,
                        "title": group["Title"].values,
                        "abstract": group["Abstract_clean"].values,
                        "label_final": (label >= 1.0).astype(int).values,
                        "label_abs": (label >= 0.5).astype(int).values,
                    }
                )
                records.append(frame)
        for review_id, members in member_of.items():
            extra = reviews[review_id]["extra"]
            extra["chan_split"] = sorted(members)
        records_df = pd.concat(records, ignore_index=True)
        records_df["record_id"] = [
            schema.text_record_id(self.name, t, a)
            for t, a in zip(records_df["title"], records_df["abstract"])
        ]
        return (
            schema.make_reviews_frame(reviews.values()),
            schema.make_records_frame(records_df),
        )
