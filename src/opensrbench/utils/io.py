"""Small I/O helpers: parquet, JSON, logging and file checksums."""

import hashlib
import json
import logging
import os
from typing import Any

import pandas as pd


def setup_logging(level: str = "INFO") -> None:
    """Configures root logging once with a compact format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(path: str) -> str:
    """Creates `path` (and parents) if needed and returns it."""
    os.makedirs(path, exist_ok=True)
    return path


def write_parquet(df: pd.DataFrame, path: str) -> str:
    """Writes a dataframe atomically to parquet."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    logging.info("Wrote %s rows -> %s", f"{len(df):,}", path)
    return path


def read_parquet(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Reads a parquet file, raising a clear error when missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run the upstream pipeline stage first "
            "(see docs/PIPELINE.md)."
        )
    return pd.read_parquet(path, columns=columns)


def write_json(obj: Any, path: str) -> str:
    """Writes pretty, key-sorted JSON atomically."""
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def read_json(path: str) -> Any:
    """Reads JSON from `path`."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path: str, chunk_bytes: int = 1 << 20) -> str:
    """Returns the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(df: pd.DataFrame, columns: list[str]) -> str:
    """Order-independent SHA-256 of selected dataframe columns."""
    rows = df[columns].astype(str).agg("\x1f".join, axis=1).sort_values()
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
