"""Europe PMC REST client: search, JATS full text and reference lists.

https://europepmc.org/RestfulWebService
"""

import gzip
import logging
import os
from typing import Iterator

from opensrbench.utils import http

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/"


class EuropePmcClient:
    """Rate-limited Europe PMC client with an on-disk full-text cache."""

    def __init__(
        self,
        cache_dir: str,
        requests_per_second: float = 8.0,
        session: http.RateLimitedSession | None = None,
    ):
        self._session = session or http.RateLimitedSession(requests_per_second)
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def search(
        self, query: str, max_results: int = 100_000, page_size: int = 1000
    ) -> Iterator[dict]:
        """Yields `lite` search hits using cursor-mark paging.

        Args:
          query: Europe PMC query syntax.
          max_results: Stop after this many hits.
          page_size: Hits per page (max 1000).

        Yields:
          Result dicts (pmid, pmcid, doi, title, firstPublicationDate, ...).
        """
        cursor = "*"
        seen = 0
        while seen < max_results:
            payload = self._session.get(
                BASE_URL + "search",
                params={
                    "query": query,
                    "format": "json",
                    "resultType": "lite",
                    "pageSize": str(page_size),
                    "cursorMark": cursor,
                },
            ).json()
            results = payload.get("resultList", {}).get("result", [])
            if seen == 0:
                logging.info("Europe PMC: %s hits", payload.get("hitCount"))
            for result in results:
                yield result
                seen += 1
                if seen >= max_results:
                    return
            next_cursor = payload.get("nextCursorMark")
            if not results or not next_cursor or next_cursor == cursor:
                return
            cursor = next_cursor

    def fulltext_path(self, pmcid: str) -> str:
        return os.path.join(self._cache_dir, f"{pmcid}.xml.gz")

    def fetch_fulltext(self, pmcid: str) -> bytes | None:
        """Returns JATS XML for a PMCID (cached, gzip on disk)."""
        path = self.fulltext_path(pmcid)
        if os.path.exists(path):
            with gzip.open(path, "rb") as fh:
                return fh.read()
        try:
            response = self._session.get(BASE_URL + f"{pmcid}/fullTextXML")
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("full text unavailable for %s: %s", pmcid, exc)
            return None
        content = response.content
        tmp = path + ".part"
        with gzip.open(tmp, "wb") as fh:
            fh.write(content)
        os.replace(tmp, path)
        return content

    def references(self, pmcid: str, page_size: int = 1000) -> list[dict]:
        """Returns Europe PMC's resolved reference list for an article.

        Each reference may carry `id` + `source` ("MED" -> PMID), `doi`,
        `title` and `pubYear`. Used to resolve references whose JATS entry
        has no <pub-id>.
        """
        out: list[dict] = []
        page = 1
        while True:
            payload = self._session.get(
                BASE_URL + f"PMC/{pmcid}/references",
                params={
                    "format": "json",
                    "pageSize": str(page_size),
                    "page": str(page),
                },
            ).json()
            refs = payload.get("referenceList", {}).get("reference", [])
            out.extend(refs)
            if len(refs) < page_size:
                return out
            page += 1
