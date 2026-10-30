"""YAML configuration loading with environment-variable expansion.

Configs are plain nested dicts wrapped in `Config` for attribute-free,
explicit access (`cfg.get("cohort.k_min")`). Values like `${VAR}` or
`${VAR:-default}` are expanded from the environment at load time.
"""

import copy
import dataclasses
import os
import re
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_MISSING = object()

DEFAULT_BENCHMARK_CONFIG = "configs/benchmark_v1.yaml"
DEFAULT_DATASETS_CONFIG = "configs/datasets.yaml"
DEFAULT_ENSEMBLE_CONFIG = "configs/titan_ensemble.yaml"


def _expand(value: Any) -> Any:
    """Recursively expands ${VAR:-default} in strings."""
    if isinstance(value, str):

        def repl(match: re.Match) -> str:
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(match.group(1), default)

        expanded = _ENV_PATTERN.sub(repl, value)
        return expanded
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class Config:
    """Read-only view over a nested configuration dictionary."""

    def __init__(self, data: dict[str, Any], source: str = "<dict>"):
        self._data = data
        self.source = source

    @classmethod
    def load(cls, path: str) -> "Config":
        """Loads and env-expands a YAML file."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(_expand(data), source=path)

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Returns a value by dotted path, e.g. "cohort.dense.enabled".

        Args:
          dotted_key: Path of keys joined by ".".
          default: Returned when the key is missing. If omitted, a KeyError
            is raised instead.

        Returns:
          The (deep-copied) value.

        Raises:
          KeyError: If the key is missing and no default is given.
        """
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise KeyError(f"{dotted_key!r} missing from {self.source}")
                return default
            node = node[part]
        return copy.deepcopy(node)

    def section(self, dotted_key: str) -> "Config":
        """Returns a sub-config."""
        return Config(self.get(dotted_key, {}), source=self.source)

    def as_dict(self) -> dict[str, Any]:
        """Returns a deep copy of the underlying data."""
        return copy.deepcopy(self._data)

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Returns a copy with dotted-key overrides applied."""
        data = self.as_dict()
        for dotted_key, value in overrides.items():
            node = data
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data, source=self.source)


@dataclasses.dataclass(frozen=True)
class Paths:
    """Canonical on-disk layout under the data root (see datasets/README.md).

    Attributes:
      root: Data root (OSB_DATA_ROOT).
      runs: Runs root for checkpoints, predictions and reports.
    """

    root: str
    runs: str

    @classmethod
    def from_config(cls, cfg: Config) -> "Paths":
        root = cfg.get("paths.data_root", "datasets") or "datasets"
        runs = cfg.get("paths.runs_root", "runs") or "runs"
        return cls(root=os.path.abspath(root), runs=os.path.abspath(runs))

    def raw(self, source: str) -> str:
        """Directory holding downloaded originals of a source."""
        return os.path.join(self.root, "raw", source)

    def interim(self, source: str) -> str:
        """Directory with a source's canonical (pre-merge) parquet files."""
        return os.path.join(self.root, "interim", source)

    @property
    def pubmed_db(self) -> str:
        return os.path.join(self.root, "cache", "pubmed.sqlite")

    @property
    def europepmc_cache(self) -> str:
        return os.path.join(self.root, "cache", "europepmc")

    @property
    def pools(self) -> str:
        return os.path.join(self.root, "interim", "pools")

    @property
    def processed(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def splits(self) -> str:
        return os.path.join(self.root, "splits")

    @property
    def frozen(self) -> str:
        return os.path.join(self.root, "frozen")

    def processed_file(self, name: str) -> str:
        return os.path.join(self.processed, f"{name}.parquet")
