"""Source-adapter interface and registry.

Each open dataset is wrapped in a `SourceAdapter` with two steps:

  download()  Network. Fetches the original files into datasets/raw/<name>/.
              Idempotent: existing, checksum-verified files are reused.
  ingest()    Offline. Parses the originals into the canonical schema
              (see opensrbench.schema). Records may omit title/abstract when
              they have a PMID; text is attached later from the PubMed store.

Adding a new dataset = one subclass decorated with `@register`.
"""

import abc
import logging
import os
import subprocess
from typing import Any

import pandas as pd

from opensrbench import config
from opensrbench.utils import http
from opensrbench.utils import io

_REGISTRY: dict[str, type["SourceAdapter"]] = {}


def register(cls: type["SourceAdapter"]) -> type["SourceAdapter"]:
    """Class decorator adding an adapter to the registry."""
    _REGISTRY[cls.name] = cls
    return cls


def registry() -> dict[str, type["SourceAdapter"]]:
    """Returns name -> adapter class for every registered source."""
    # Importing the package populates the registry.
    from opensrbench import \
        sources  # pylint: disable=import-outside-toplevel,unused-import

    return dict(_REGISTRY)


class SourceAdapter(abc.ABC):
    """Base class for dataset adapters.

    Attributes:
      name: Registry key; also the review_id prefix.
      cfg: The adapter's section of configs/datasets.yaml.
      paths: Data layout.
    """

    name: str = ""

    def __init__(
        self,
        cfg: dict[str, Any],
        paths: config.Paths,
        context: config.Config | None = None,
        session: http.RateLimitedSession | None = None,
    ):
        """Initializes the adapter.

        Args:
          cfg: The source's section of configs/datasets.yaml.
          paths: Data layout.
          context: The benchmark config (network settings, harvest window).
          session: Optional HTTP session (tests inject fakes).
        """
        self.cfg = cfg
        self.paths = paths
        self.context = context or config.Config({})
        self.session = session or http.RateLimitedSession(4.0)

    @property
    def raw_dir(self) -> str:
        return self.paths.raw(self.name)

    @abc.abstractmethod
    def download(self) -> None:
        """Downloads originals into `raw_dir`."""

    @abc.abstractmethod
    def ingest(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parses originals into canonical (reviews, records) frames."""

    def run_ingest(self) -> tuple[str, str]:
        """Runs `ingest` and writes interim parquet files."""
        reviews, records = self.ingest()
        out = self.paths.interim(self.name)
        logging.info(
            "[%s] %d reviews, %s records (%s positives)",
            self.name,
            len(reviews),
            f"{len(records):,}",
            f"{int(records['label_final'].sum()):,}",
        )
        return (
            io.write_parquet(reviews, os.path.join(out, "reviews.parquet")),
            io.write_parquet(records, os.path.join(out, "records.parquet")),
        )

    # Shared download helpers ------------------------------------------------

    def git_clone(self, url: str, dest_name: str = "repo") -> str:
        """Shallow-clones a git repository (or reuses an existing clone)."""
        dest = os.path.join(self.raw_dir, dest_name)
        if os.path.isdir(os.path.join(dest, ".git")):
            logging.info("[%s] reusing clone %s", self.name, dest)
            return dest
        io.ensure_dir(self.raw_dir)
        logging.info("[%s] cloning %s", self.name, url)
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", url, dest], check=True
        )
        return dest

    def fetch(self, url: str, filename: str, sha256: str | None = None) -> str:
        """Downloads a file once, verifying its checksum when given.

        Raises:
          ValueError: If the checksum does not match.
        """
        dest = os.path.join(self.raw_dir, filename)
        if not os.path.exists(dest):
            logging.info("[%s] downloading %s", self.name, url)
            http.download_file(self.session, url, dest)
        if sha256:
            actual = io.file_sha256(dest)
            if actual != sha256:
                raise ValueError(
                    f"{dest}: sha256 {actual} != expected {sha256}"
                    ". Delete the file and re-run download."
                )
        return dest
