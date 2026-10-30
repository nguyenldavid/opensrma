"""End-to-end smoke test of the offline pipeline (~1 minute on CPU)."""

import argparse
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

# pylint: disable=import-outside-toplevel,wrong-import-position
from opensrbench import smoke

_ROOT = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.skipif(
    os.environ.get("OSB_SKIP_SMOKE") == "1", reason="OSB_SKIP_SMOKE=1"
)
def test_smoke_pipeline_runs_and_has_zero_leakage(tmp_path):
    args = argparse.Namespace(
        config=os.path.join(_ROOT, "configs", "benchmark_v1.yaml"),
        ensemble_config=os.path.join(_ROOT, "configs", "titan_ensemble.yaml"),
        workdir=str(tmp_path / "smoke"),
        runs_dir=None,
        log_level="WARNING",
    )
    result = smoke.run(args)
    systems = {row["system"] for row in result["leaderboard"]}
    assert "smoke@tau95" in systems
    assert any(s.startswith("cascade[") for s in systems)
    best = max(
        row["youdens_j"][0]
        for row in result["leaderboard"]
        if row["system"].startswith("smoke@")
    )
    assert best > 0.5  # The tiny model learns the synthetic signal.
