"""Tests for the E-utilities client."""

import requests

from opensrbench.pubmed import eutils
from opensrbench.pubmed import store
from opensrbench.utils import http


class _Response:

    headers: dict = {}

    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)


class _FakeSession:
    """Answers 403 on the ID Converter and serves ESummary (pmc)."""

    def __init__(self):
        self.calls: list[str] = []

    def get(self, url: str, **kwargs) -> _Response:
        del kwargs
        self.calls.append(url)
        _Response(403, {}).raise_for_status()

    def post(self, url: str, data: dict) -> _Response:
        self.calls.append(url)
        uids = data["id"].split(",")
        result = {"uids": uids}
        for uid in uids:
            result[uid] = {
                "articleids": [
                    {"idtype": "pmid", "value": str(int(uid) + 1)},
                    {"idtype": "pmcid", "value": f"PMC{uid}"},
                ]
            }
        return _Response(200, {"result": result})


def test_pmcid_to_pmid_falls_back_to_esummary_on_403():
    session = _FakeSession()
    client = eutils.EUtilsClient(email="x@example.org", session=session)
    mapping = client.pmcid_to_pmid(["PMC10", "PMC20", "PMC30"], batch_size=2)
    assert mapping == {"PMC10": "11", "PMC20": "21", "PMC30": "31"}
    # The ID Converter is tried once, then every batch goes to ESummary.
    assert sum(eutils.IDCONV_URL == c for c in session.calls) == 1
    assert sum(c.endswith("esummary.fcgi") for c in session.calls) == 2


class _FakeEfetchClient:

    def efetch(self, pmids, batch_size):
        del batch_size
        yield [{"pmid": p, "title": f"t{p}"} for p in pmids]


def test_fetch_missing_reads_cache_once(tmp_path, monkeypatch):
    with store.PubMedStore(str(tmp_path / "cache.sqlite")) as cache:
        calls = []
        original = cache.known
        monkeypatch.setattr(
            cache, "known", lambda ids: calls.append(1) or original(ids)
        )
        assert cache.fetch_missing(_FakeEfetchClient(), ["1", "2", "3"]) == 3
        assert len(calls) == 1


def test_session_retries_truncated_body(monkeypatch):
    session = http.RateLimitedSession(1000, backoff_seconds=0.0)
    replies = [
        requests.exceptions.ChunkedEncodingError("cut"),
        _Response(200, {}),
    ]

    def fake_request(method, url, **kwargs):
        del method, url, kwargs
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(session._session, "request", fake_request)
    assert session.get("https://example.org").status_code == 200
