"""Pre-registration and the unblinding ledger.

Held-out benchmarks (internal_test, open_sr_bench[_extended], titan_test,
prospective) may only be scored with an explicit `--unblind`. Every
unblinding appends an entry to <runs>/unblinding_log.jsonl recording when,
from which git commit, which systems, and the SHA-256 of every locked
threshold file, plus whether those files were pre-registered (an identical
copy committed under datasets/manifests/ via `osb preregister`). Repeated
looks at a held-out set are therefore visible in the record — the main
source of "leakage by iteration".
"""

import datetime
import glob
import json
import os
import shutil
import subprocess

from opensrbench.utils import io

PROTECTED_BENCHMARKS = frozenset(
    {
        "internal_test",
        "open_sr_bench",
        "open_sr_bench_extended",
        "titan_test",
        "prospective",
    }
)
DEV_BENCHMARKS = frozenset({"val"})
MANIFEST_DIR = os.path.join("datasets", "manifests")


class BlindedError(PermissionError):
    """Raised when a held-out benchmark is scored without --unblind."""


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def locked_files(runs_root: str) -> dict[str, str]:
    """Locked-threshold file -> SHA-256."""
    return {
        path: io.file_sha256(path)
        for path in sorted(
            glob.glob(
                os.path.join(
                    runs_root,
                    "calibration",
                    "*",
                    "locked_operating_points.json",
                )
            )
        )
    }


def is_preregistered(sha: str, manifest_dir: str = MANIFEST_DIR) -> bool:
    for path in glob.glob(os.path.join(manifest_dir, "*.json")):
        if io.file_sha256(path) == sha:
            return True
    return False


def require_unblind(benchmark: str, unblind: bool) -> None:
    """Raises BlindedError for protected benchmarks without --unblind."""
    if benchmark in PROTECTED_BENCHMARKS and not unblind:
        raise BlindedError(
            f"{benchmark} is a held-out benchmark. Pre-register first "
            "(`osb preregister`), then re-run with --unblind; every unblinding "
            "is logged. For development use --benchmark val."
        )


def record_unblinding(
    runs_root: str, benchmark: str, tier: int, systems: list[str]
) -> dict:
    """Appends one ledger entry and returns it."""
    files = locked_files(runs_root)
    entry = {
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "benchmark": benchmark,
        "tier": tier,
        "git_commit": _git_commit(),
        "systems": sorted(systems),
        "locked_threshold_files": {
            path: {"sha256": sha, "preregistered": is_preregistered(sha)}
            for path, sha in files.items()
        },
    }
    path = os.path.join(runs_root, "unblinding_log.jsonl")
    io.ensure_dir(runs_root)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def preregister(
    runs_root: str, frozen_root: str, manifest_dir: str = MANIFEST_DIR
) -> list[str]:
    """Copies frozen manifests + locked thresholds into datasets/manifests/.

    Commit the copied files before running `osb evaluate --unblind`.

    Returns:
      Paths written.
    """
    io.ensure_dir(manifest_dir)
    written = []
    for path in sorted(
        glob.glob(os.path.join(frozen_root, "*", "manifest.json"))
    ):
        name = os.path.basename(os.path.dirname(path))
        dest = os.path.join(manifest_dir, f"{name}.manifest.json")
        shutil.copyfile(path, dest)
        written.append(dest)
    for path in locked_files(runs_root):
        name = os.path.basename(os.path.dirname(path))
        dest = os.path.join(
            manifest_dir, f"locked_operating_points.{name}.json"
        )
        shutil.copyfile(path, dest)
        written.append(dest)
    return written
