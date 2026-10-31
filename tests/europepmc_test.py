"""Tests for the Europe PMC client."""

from opensrbench.harvest import europepmc


class _Reply:

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("empty body")
        return self._payload


class _Session:

    def __init__(self, replies):
        self.replies = list(replies)

    def get(self, url, params):
        del url, params
        return _Reply(self.replies.pop(0))


def _client(replies, tmp_path):
    return europepmc.EuropePmcClient(
        cache_dir=str(tmp_path), session=_Session(replies)
    )


def test_references_retries_an_empty_reply(tmp_path):
    refs = {"referenceList": {"reference": [{"id": "1", "source": "MED"}]}}
    client = _client([None, refs], tmp_path)
    assert client.references("PMC1") == [{"id": "1", "source": "MED"}]


def test_references_gives_up_without_crashing(tmp_path):
    client = _client([None, None, None], tmp_path)
    assert not client.references("PMC1")
