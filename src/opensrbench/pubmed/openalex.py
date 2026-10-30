"""OpenAlex client for citations that have no PMID (e.g. SYNERGY records).

OpenAlex data is CC0. Its API meters usage with daily credits; set
OPENALEX_API_KEY for a personal quota. The pipeline only falls back to
OpenAlex for records that PubMed cannot resolve by DOI.
"""

import os
from typing import Iterable, Iterator

from opensrbench.utils import http

BASE_URL = "https://api.openalex.org/works"


def _abstract_from_inverted_index(index: dict | None) -> str:
    """Rebuilds plain text from OpenAlex's `abstract_inverted_index`."""
    if not index:
        return ""
    positions = []
    for word, offsets in index.items():
        positions.extend((offset, word) for offset in offsets)
    return " ".join(word for _, word in sorted(positions))


def normalize_openalex_id(value: object) -> str | None:
    """Returns "W123..." from a bare id or an https://openalex.org/ URL."""
    if not isinstance(value, str) or not value.strip():
        return None
    tail = value.strip().rstrip("/").rsplit("/", 1)[-1]
    return tail if tail[:1] in ("W", "w") and tail[1:].isdigit() else None


class OpenAlexClient:
    """Batch fetcher compatible with `PubMedStore.fetch_missing`."""

    def __init__(
        self, mailto: str = "", session: http.RateLimitedSession | None = None
    ):
        self._mailto = mailto
        self._session = session or http.RateLimitedSession(8.0)

    def efetch(
        self, ids: Iterable[str], batch_size: int = 50
    ) -> Iterator[list[dict]]:
        """Fetches works by OpenAlex id ("W123") or DOI ("doi:10.x/y").

        Yields:
          Lists of dicts with key (the requested id), openalex_id, pmid, doi,
          title, abstract and pub_date.
        """
        ids = [i for i in dict.fromkeys(ids) if i]
        size = min(batch_size, 50)
        for start in range(0, len(ids), size):
            batch = ids[start : start + size]
            by_doi = [i[4:].lower() for i in batch if i.startswith("doi:")]
            by_id = [i for i in batch if not i.startswith("doi:")]
            out = []
            for field, values in (("openalex", by_id), ("doi", by_doi)):
                if values:
                    out.extend(self._query(field, values))
            yield out

    def _query(self, field: str, values: list[str]) -> list[dict]:
        params = {"filter": f"{field}:" + "|".join(values), "per-page": "50"}
        if self._mailto:
            params["mailto"] = self._mailto
        if os.environ.get("OPENALEX_API_KEY"):
            params["api_key"] = os.environ["OPENALEX_API_KEY"]
        payload = self._session.get(BASE_URL, params=params).json()
        out = []
        for work in payload.get("results", []):
            ids_block = work.get("ids") or {}
            oa_id = normalize_openalex_id(work.get("id"))
            doi = (work.get("doi") or "").lower().split("doi.org/")[-1] or None
            out.append(
                {
                    "key": f"doi:{doi}" if field == "doi" else oa_id,
                    "openalex_id": oa_id,
                    "pmid": (
                        (ids_block.get("pmid") or "").rsplit("/", 1)[-1] or None
                    ),
                    "doi": doi,
                    "title": (
                        work.get("title") or work.get("display_name") or ""
                    ),
                    "abstract": _abstract_from_inverted_index(
                        work.get("abstract_inverted_index")
                    ),
                    "pub_date": work.get("publication_date"),
                    "publication_types": [work.get("type") or ""],
                }
            )
        return out
