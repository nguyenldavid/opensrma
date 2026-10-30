"""Transactional SQLite cache of PubMed citations.

Every PMID is fetched from NCBI at most once across all stages and sources.
The store is resumable: an interrupted fetch keeps every committed batch.
"""

import json
import logging
import os
import sqlite3
import time
from typing import Iterable

import pandas as pd
import tqdm

from opensrbench.pubmed import eutils

_SCHEMA = """
CREATE TABLE IF NOT EXISTS citations (
  pmid TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS missing (
  pmid TEXT PRIMARY KEY,
  fetched_at REAL NOT NULL
);
"""


class PubMedStore:
    """Citation-id -> parsed citation cache backed by SQLite.

    Keys are PMIDs by default; the OpenAlex cache reuses the class with
    `key_field="openalex_id"`. Any client exposing
    `efetch(ids, batch_size) -> Iterator[list[dict]]` can fill it.
    """

    def __init__(self, path: str, key_field: str = "pmid"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.key_field = key_field
        self._conn = sqlite3.connect(path, timeout=120)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PubMedStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def known(self, pmids: Iterable[str]) -> set[str]:
        """Returns PMIDs already cached (found or confirmed missing)."""
        pmids = list(dict.fromkeys(str(p) for p in pmids if p))
        found: set[str] = set()
        for start in range(0, len(pmids), 900):
            chunk = pmids[start : start + 900]
            marks = ",".join("?" * len(chunk))
            for table in ("citations", "missing"):
                rows = self._conn.execute(
                    f"SELECT pmid FROM {table} WHERE pmid IN ({marks})", chunk
                )
                found.update(r[0] for r in rows)
        return found

    def put_many(self, records: list[dict], requested: list[str]) -> None:
        """Commits a fetched batch, marking unreturned PMIDs as missing."""
        now = time.time()
        returned = {r[self.key_field] for r in records}
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO citations VALUES (?, ?, ?)",
                [(r[self.key_field], json.dumps(r), now) for r in records],
            )
            self._conn.executemany(
                "INSERT OR REPLACE INTO missing VALUES (?, ?)",
                [(p, now) for p in requested if p not in returned],
            )

    def fetch_missing(
        self,
        client: "eutils.EUtilsClient",
        pmids: Iterable[str],
        batch_size: int = 200,
    ) -> int:
        """Downloads every PMID not yet cached.

        Args:
          client: E-utilities client.
          pmids: PMIDs required downstream.
          batch_size: EFetch batch size.

        Returns:
          Number of PMIDs requested from NCBI.
        """
        pmids = list(dict.fromkeys(str(p) for p in pmids if p))
        todo = [p for p in pmids if p not in self.known(pmids)]
        if not todo:
            return 0
        logging.info(
            "Fetching %s %s records (%s cached)",
            f"{len(todo):,}",
            self.key_field,
            f"{len(pmids) - len(todo):,}",
        )
        bar = tqdm.tqdm(total=len(todo), unit="rec", desc="fetch")
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            records = next(client.efetch(batch, batch_size=len(batch)), [])
            self.put_many(records, batch)
            bar.update(len(batch))
        bar.close()
        return len(todo)

    def get_many(self, pmids: Iterable[str]) -> dict[str, dict]:
        """Returns cached citations for the given PMIDs."""
        pmids = list(dict.fromkeys(str(p) for p in pmids if p))
        out: dict[str, dict] = {}
        for start in range(0, len(pmids), 900):
            chunk = pmids[start : start + 900]
            marks = ",".join("?" * len(chunk))
            rows = self._conn.execute(
                f"SELECT pmid, payload FROM citations WHERE pmid IN ({marks})",
                chunk,
            )
            out.update((pmid, json.loads(payload)) for pmid, payload in rows)
        return out

    def frame(self, pmids: Iterable[str]) -> pd.DataFrame:
        """Returns key/title/abstract/pub_date/publication_types as a frame."""
        records = self.get_many(pmids)
        cols = [
            self.key_field,
            "title",
            "abstract",
            "pub_date",
            "publication_types",
            "doi",
        ]
        if not records:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(list(records.values()))
        return df.reindex(columns=cols)

    def put_external(self, records: list[dict]) -> None:
        """Inserts citations obtained elsewhere (tests, offline mirrors)."""
        self.put_many(records, [r[self.key_field] for r in records])
