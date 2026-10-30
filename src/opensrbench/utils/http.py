"""Polite, rate-limited HTTP session with retries.

NCBI allows 3 requests/s without an API key and 10 requests/s with one; Europe
PMC asks for similar courtesy. Every network call in the package goes through
`RateLimitedSession`, so rate limits are enforced in one place.
"""

import logging
import os
import threading
import time

import requests

_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_USER_AGENT = "opensrbench/1.1 (+https://github.com/nguyenldavid/opensrma)"


class RateLimitedSession:
    """A thread-safe `requests.Session` wrapper with rate limiting and retry.

    Attributes:
      min_interval: Minimum seconds between consecutive requests.
    """

    def __init__(
        self,
        requests_per_second: float = 3.0,
        max_retries: int = 6,
        backoff_seconds: float = 2.0,
        timeout_seconds: float = 120.0,
    ):
        """Initializes the session.

        Args:
          requests_per_second: Upper bound on the request rate.
          max_retries: Attempts after the first failure.
          backoff_seconds: Base for exponential backoff (2, 4, 8, ...).
          timeout_seconds: Per-request timeout.
        """
        self.min_interval = 1.0 / max(requests_per_second, 1e-6)
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._timeout = timeout_seconds
        self._lock = threading.Lock()
        self._last = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})

    def _wait_turn(self) -> None:
        with self._lock:
            delay = self._last + self.min_interval - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Issues a request, retrying transient failures.

        Args:
          method: "GET" or "POST".
          url: Absolute URL.
          **kwargs: Forwarded to `requests.Session.request`.

        Returns:
          The successful response.

        Raises:
          requests.HTTPError: On a non-retryable status or exhausted retries.
          requests.ConnectionError: When the network stays unreachable.
        """
        kwargs.setdefault("timeout", self._timeout)
        attempt = 0
        while True:
            self._wait_turn()
            try:
                response = self._session.request(method, url, **kwargs)
                if response.status_code not in _RETRY_STATUS:
                    response.raise_for_status()
                    return response
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit() and int(retry_after) > 600:
                    # Quota exhausted (e.g. a daily credit limit): fail fast
                    # instead of sleeping for hours.
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} for {url}: quota "
                        f"exhausted, retry after {int(retry_after)}s",
                        response=response,
                    )
                error = requests.HTTPError(
                    f"HTTP {response.status_code} for {url}", response=response
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                error = exc
            attempt += 1
            if attempt > self._max_retries:
                raise error
            sleep = self._backoff**attempt
            logging.warning(
                "Retry %d/%d in %.0fs after: %s",
                attempt,
                self._max_retries,
                sleep,
                error,
            )
            time.sleep(sleep)

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET with rate limiting and retries."""
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        """POST with rate limiting and retries."""
        return self.request("POST", url, **kwargs)


def download_file(
    session: RateLimitedSession, url: str, dest: str, chunk_bytes: int = 1 << 20
) -> str:
    """Streams `url` to `dest` atomically (via a .part file).

    Args:
      session: Session to use.
      url: Source URL.
      dest: Destination path.
      chunk_bytes: Streaming chunk size.

    Returns:
      `dest`.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    response = session.get(url, stream=True)
    with open(tmp, "wb") as fh:
        for chunk in response.iter_content(chunk_size=chunk_bytes):
            fh.write(chunk)
    os.replace(tmp, dest)
    return dest
