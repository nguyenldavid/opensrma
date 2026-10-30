"""NCBI E-utilities client (ESearch / EFetch) with date bounding.

See https://www.ncbi.nlm.nih.gov/books/NBK25499/. NCBI requires a contact
email; an API key raises the limit from 3 to 10 requests per second.
"""

import datetime
import logging
from typing import Iterable, Iterator

from opensrbench.pubmed import parse
from opensrbench.utils import dates
from opensrbench.utils import http

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ESEARCH_HARD_CAP = 9999  # PubMed returns at most 10,000 ids per query.
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"


class EUtilsClient:
    """Minimal, rate-limited PubMed client."""

    def __init__(
        self,
        email: str = "",
        api_key: str = "",
        requests_per_second: float = 9.0,
        tool: str = "opensrbench",
        session: http.RateLimitedSession | None = None,
    ):
        """Initializes the client.

        Args:
          email: Contact email sent to NCBI (strongly recommended).
          api_key: Optional NCBI API key.
          requests_per_second: Requested rate; clamped to 3 without a key.
          tool: Tool name sent to NCBI.
          session: Optional pre-built session (used by tests).
        """
        if not email:
            logging.warning(
                "NCBI_EMAIL is not set; NCBI asks every client to "
                "identify itself. export NCBI_EMAIL=you@org"
            )
        rate = requests_per_second if api_key else min(requests_per_second, 3)
        self._session = session or http.RateLimitedSession(rate)
        self._base = {"tool": tool, "email": email}
        if api_key:
            self._base["api_key"] = api_key

    def _params(self, **kwargs) -> dict[str, str]:
        params = dict(self._base)
        params.update({k: str(v) for k, v in kwargs.items() if v is not None})
        return params

    @staticmethod
    def date_bounded(term: str, ceiling: datetime.date | None) -> str:
        """Appends a publication-date ceiling `[1900/01/01 : t_r][dp]`."""
        if ceiling is None:
            return term
        return (
            f'({term}) AND ("1900/01/01"[dp] : '
            f'"{dates.to_pubmed(ceiling)}"[dp])'
        )

    def esearch(
        self,
        term: str,
        max_results: int = ESEARCH_HARD_CAP,
        sort: str | None = None,
    ) -> tuple[int, list[str]]:
        """Runs ESearch and returns (total_count, pmids).

        Args:
          term: PubMed query.
          max_results: Maximum ids to return (capped at 9,999).
          sort: Optional sort order, e.g. "relevance" or "pub_date".

        Returns:
          Tuple of the total hit count and up to `max_results` PMIDs.
        """
        max_results = min(max_results, ESEARCH_HARD_CAP)
        pmids: list[str] = []
        count = None
        retstart = 0
        while retstart < max_results:
            response = self._session.post(
                BASE_URL + "esearch.fcgi",
                data=self._params(
                    db="pubmed",
                    term=term,
                    retmode="json",
                    retstart=retstart,
                    retmax=max_results - retstart,
                    sort=sort,
                ),
            )
            result = response.json().get("esearchresult", {})
            if "ERROR" in result:
                logging.warning(
                    "ESearch error for %r: %s", term[:120], result["ERROR"]
                )
            count = int(result.get("count", 0)) if count is None else count
            batch = result.get("idlist", [])
            pmids.extend(batch)
            retstart += len(batch)
            if not batch or retstart >= count:
                break
        return count or 0, pmids

    def count(self, term: str) -> int:
        """Returns only the hit count of a query."""
        response = self._session.post(
            BASE_URL + "esearch.fcgi",
            data=self._params(
                db="pubmed", term=term, retmode="json", rettype="count"
            ),
        )
        return int(response.json().get("esearchresult", {}).get("count", 0))

    def efetch(
        self, pmids: Iterable[str], batch_size: int = 200
    ) -> Iterator[list[dict]]:
        """Fetches and parses citations in batches.

        Args:
          pmids: PMIDs to fetch.
          batch_size: PMIDs per request (NCBI recommends <= 200 for GET; POST
            tolerates more but 200 keeps responses small).

        Yields:
          Lists of parsed citation dicts, one list per batch.
        """
        pmids = list(dict.fromkeys(p for p in pmids if p))
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            response = self._session.post(
                BASE_URL + "efetch.fcgi",
                data=self._params(
                    db="pubmed", id=",".join(batch), retmode="xml"
                ),
            )
            yield parse.parse_efetch_xml(response.content)

    def doi_to_pmid(
        self, dois: Iterable[str], batch_size: int = 50
    ) -> dict[str, str]:
        """Resolves DOIs to PMIDs with batched `[doi]` ESearch + EFetch.

        Args:
          dois: DOIs (with or without the https://doi.org/ prefix).
          batch_size: DOIs per ESearch.

        Returns:
          Mapping lower-cased DOI -> PMID for DOIs found in PubMed.
        """
        clean = []
        for doi in dois:
            if doi:
                doi = str(doi).strip().lower()
                doi = doi.split("doi.org/", 1)[-1]
                if doi.startswith("10."):
                    clean.append(doi)
        clean = list(dict.fromkeys(clean))
        mapping: dict[str, str] = {}
        for start in range(0, len(clean), batch_size):
            batch = clean[start : start + batch_size]
            term = " OR ".join(f'"{d}"[doi]' for d in batch)
            _, pmids = self.esearch(term, max_results=len(batch) * 3)
            for records in self.efetch(pmids):
                for record in records:
                    if record.get("doi"):
                        mapping[record["doi"].lower()] = record["pmid"]
        return mapping

    def title_to_pmid(self, title: str) -> str | None:
        """Resolves an exact article title to a PMID (None if ambiguous)."""
        cleaned = " ".join(str(title).replace('"', " ").split())
        if len(cleaned) < 20:
            return None
        _, pmids = self.esearch(f'"{cleaned}"[ti]', max_results=3)
        return pmids[0] if len(pmids) == 1 else None

    def pmcid_to_pmid(
        self, pmcids: Iterable[str], batch_size: int = 200
    ) -> dict[str, str]:
        """Maps PMCIDs to PMIDs with the PMC ID Converter API."""
        pmcids = list(dict.fromkeys(p for p in pmcids if p))
        mapping: dict[str, str] = {}
        for start in range(0, len(pmcids), batch_size):
            params = dict(self._base)
            params.pop("api_key", None)
            params.update(
                {
                    "ids": ",".join(pmcids[start : start + batch_size]),
                    "format": "json",
                }
            )
            payload = self._session.get(IDCONV_URL, params=params).json()
            for record in payload.get("records", []):
                if record.get("pmid") and record.get("pmcid"):
                    mapping[record["pmcid"]] = str(record["pmid"])
        return mapping

    def cochrane_to_pmid(self, cd_number: str) -> str | None:
        """Returns the PMID of the latest version of a Cochrane review."""
        _, pmids = self.esearch(
            f'{cd_number}[pg] AND "Cochrane Database Syst Rev"[ta]',
            max_results=20,
        )
        return max(pmids, key=int) if pmids else None
