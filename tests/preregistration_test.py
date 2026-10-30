"""Tests for the pre-registration gate and the A19 threshold sets."""

import argparse
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from opensrbench import cli
from opensrbench import config
from opensrbench.calibration import run as calibrate
from opensrbench.evaluation import report
from opensrbench.utils import io

_REPO = os.path.join(os.path.dirname(__file__), "..")


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _init_repo(path):
    """A throw-away git repo with a first commit and datasets/manifests/."""
    os.makedirs(os.path.join(path, "datasets", "manifests"))
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Test")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "commit.gpgsign", "false")
    _git(path, "config", "core.hooksPath", os.path.join(path, ".no-hooks"))
    readme = os.path.join(path, "datasets", "manifests", "README.md")
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write("manifests\n")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")
    return path


def _write_locked(runs, ensemble, content):
    path = os.path.join(
        runs, "calibration", ensemble, "locked_operating_points.json"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


@pytest.fixture(name="setup")
def _setup(tmp_path):
    repo = _init_repo(str(tmp_path / "repo"))
    runs = str(tmp_path / "runs")
    locked = _write_locked(runs, "ens", '{"thresholds": {"tau99": 0.1}}\n')
    manifests = os.path.join(repo, "datasets", "manifests")
    copy = os.path.join(manifests, "locked_operating_points.ens.t.json")
    return repo, runs, locked, manifests, copy


def test_calibrated_ensembles_parses_system_names():
    names = [
        "titan_open@tau99",
        "titan_open@pooled_conformal.tau99",
        "titan_sr_baseline@tauJ",
        "cascade[titan_open->gpt-oss-20b/condition_b]",
        "gpt-oss-20b/condition_b",
    ]
    assert report.calibrated_ensembles(names) == [
        "titan_open",
        "titan_sr_baseline",
    ]


def test_gate_passes_on_committed_identical_copy(setup):
    repo, runs, locked, manifests, copy = setup
    shutil.copy(locked, copy)
    _git(repo, "add", copy)
    _git(repo, "commit", "-q", "-m", "pre-register")
    out = report.check_preregistration(runs, ["ens"], manifests)
    assert out["verified"]
    assert out["commit"] == _git(repo, "rev-parse", "HEAD").strip()
    assert out["ensembles"]["ens"]["committed_as"] == [
        "datasets/manifests/locked_operating_points.ens.t.json"
    ]


def test_gate_fails_without_copy(setup):
    _, runs, _, manifests, _ = setup
    with pytest.raises(report.PreregistrationError, match="matches no file"):
        report.check_preregistration(runs, ["ens"], manifests)


def test_gate_fails_on_uncommitted_copy(setup):
    repo, runs, locked, manifests, copy = setup
    shutil.copy(locked, copy)  # Untracked.
    with pytest.raises(report.PreregistrationError):
        report.check_preregistration(runs, ["ens"], manifests)
    _git(repo, "add", copy)  # Staged but not committed.
    with pytest.raises(report.PreregistrationError):
        report.check_preregistration(runs, ["ens"], manifests)


def test_gate_fails_when_thresholds_change_after_commit(setup):
    repo, runs, locked, manifests, copy = setup
    shutil.copy(locked, copy)
    _git(repo, "add", copy)
    _git(repo, "commit", "-q", "-m", "pre-register")
    _write_locked(runs, "ens", '{"thresholds": {"tau99": 0.2}}\n')
    with pytest.raises(report.PreregistrationError, match="ens"):
        report.check_preregistration(runs, ["ens"], manifests)
    # A committed copy of different content does not pass either, even if
    # the working-tree copy is edited to match.
    shutil.copy(locked, copy)
    with pytest.raises(report.PreregistrationError):
        report.check_preregistration(runs, ["ens"], manifests)


def test_gate_fails_on_missing_locked_file_or_repo(setup, tmp_path):
    _, runs, _, manifests, _ = setup
    with pytest.raises(report.PreregistrationError, match="does not exist"):
        report.check_preregistration(runs, ["other"], manifests)
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(report.PreregistrationError, match="git"):
        report.check_preregistration(runs, ["ens"], str(plain))


def test_gate_without_calibrated_systems_needs_no_git(tmp_path):
    out = report.check_preregistration(str(tmp_path), [], str(tmp_path))
    assert out["verified"] and not out["ensembles"]


def test_gate_can_be_disabled_by_config_or_flag(tmp_path):
    names = ["ens@tau99"]
    args = argparse.Namespace(skip_preregistration_check=False)
    with pytest.raises(report.PreregistrationError):
        report._preregistration_gate(  # pylint: disable=protected-access
            args,
            {"preregistration_dir": str(tmp_path)},
            str(tmp_path),
            names,
        )
    out = report._preregistration_gate(  # pylint: disable=protected-access
        args, {"require_preregistration": False}, str(tmp_path), names
    )
    assert not out["verified"] and out["ensembles"] == ["ens"]
    args.skip_preregistration_check = True
    out = report._preregistration_gate(  # pylint: disable=protected-access
        args, {"require_preregistration": True}, str(tmp_path), names
    )
    assert out["skipped_by"] == "--skip-preregistration-check"
    table = report.markdown_table(
        {
            "tier": 20,
            "reference": None,
            "leaderboard": [],
            "comparisons": {},
            "preregistration": out,
        }
    )
    assert "NOT verified" in table


def test_cli_flag_parses():
    parser = cli.build_parser()
    assert parser.parse_args(
        ["evaluate", "--skip-preregistration-check"]
    ).skip_preregistration_check
    assert not parser.parse_args(["evaluate"]).skip_preregistration_check


def test_preregister_script_commits_copies(tmp_path):
    repo = _init_repo(str(tmp_path / "repo"))
    os.makedirs(os.path.join(repo, "slurm"))
    for name in ("env.sh", "preregister.sh"):
        shutil.copy(
            os.path.join(_REPO, "slurm", name), os.path.join(repo, "slurm")
        )
    _git(repo, "add", "slurm")
    _git(repo, "commit", "-q", "-m", "scripts")
    storage = tmp_path / "storage"
    env = dict(
        os.environ,
        OSB_REPO=repo,
        OSB_STORAGE=str(storage),
        OSB_VENV=str(tmp_path / "no-venv"),
        HF_HOME=str(tmp_path / "hf"),
        RUN_TAG="t",
    )
    env.pop("OSB_DATA_ROOT", None)
    env.pop("OSB_RUNS_ROOT", None)
    env.pop("SLURM_JOB_ID", None)
    runs = str(storage / "runs")
    _write_locked(runs, "ens", '{"thresholds": {"tau99": 0.1}}\n')
    frozen = storage / "datasets" / "frozen" / "bench"
    frozen.mkdir(parents=True)
    (frozen / "manifest.json").write_text('{"hash": "x"}\n')

    def preregister():
        return subprocess.run(
            ["bash", "slurm/preregister.sh", "ens"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    proc = preregister()
    assert proc.returncode == 0, proc.stderr
    assert "push" in proc.stdout.lower()
    log = _git(repo, "log", "-1", "--format=%B")
    assert log.startswith("Pre-register locked operating points (t)")
    assert "Co-Authored-By:" in log and "Claude-Session:" in log
    committed = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert sorted(committed) == [
        "datasets/manifests/bench.t.manifest.json",
        "datasets/manifests/locked_operating_points.ens.t.json",
    ]
    manifests = os.path.join(repo, "datasets", "manifests")
    assert report.check_preregistration(runs, ["ens"], manifests)["verified"]
    head = _git(repo, "rev-parse", "HEAD")
    proc = preregister()  # Identical re-run: no new commit.
    assert proc.returncode == 0, proc.stderr
    assert _git(repo, "rev-parse", "HEAD") == head

    def registered(ensemble):
        return subprocess.run(
            ["bash", "slurm/preregister.sh", "--registered", ensemble],
            cwd=repo,
            env=env,
            capture_output=True,
            check=False,
        ).returncode

    # Jobs skip `osb calibrate` only for an identical committed lock.
    assert registered("ens") == 0
    assert registered("other") != 0
    _write_locked(runs, "ens", '{"thresholds": {"tau99": 0.2}}\n')
    assert registered("ens") != 0
    proc = preregister()  # Different thresholds: refuse to re-lock.
    assert proc.returncode != 0
    assert "re-lock" in proc.stderr
    assert _git(repo, "rev-parse", "HEAD") == head


def _val(seed=0, n_reviews=40):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_reviews):
        n_pos = int(rng.integers(2, 15))
        for i in range(n_pos + 30):
            y = int(i < n_pos)
            rows.append(
                {
                    "review_id": f"R{r}",
                    "family_id": f"F{r // 2}",
                    "record_id": f"R{r}:{i}",
                    "label_final": y,
                    "score": float(rng.normal(1.5 * y, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def test_locked_points_hold_primary_and_secondary_sets():
    targets = [0.9, 0.99]
    locked = calibrate.lock_operating_points(
        _val(), targets, "equal_review_lcb", ["pooled_conformal"]
    )
    assert locked["threshold_method"] == "equal_review_lcb"
    assert set(locked["thresholds"]) == {"tau90", "tau99", "tauJ"}
    secondary = locked["secondary"]["pooled_conformal"]
    assert set(secondary["thresholds"]) == {"tau90", "tau99"}
    for entry in (locked, secondary):
        for op, target in (("tau90", 0.9), ("tau99", 0.99)):
            val = entry["validation"][op]
            assert val["target_sensitivity"] == target
            assert 0.0 <= val["equal_review_sensitivity"] <= 1.0
    # The LCB method certifies the equal-review target on validation.
    for op, target in (("tau90", 0.9), ("tau99", 0.99)):
        achieved = locked["validation"][op]["equal_review_sensitivity"]
        assert achieved >= target - 1e-12
    # The primary method is never duplicated as a secondary set.
    same = calibrate.lock_operating_points(
        _val(), targets, "pooled_conformal", ["pooled_conformal"]
    )
    assert not same["secondary"]


def test_benchmark_config_records_amendment_a19():
    cfg = config.Config.load(
        os.path.join(_REPO, "configs", "benchmark_v1.yaml")
    )
    assert cfg.get("calibration.threshold_method") == "equal_review_lcb"
    assert "pooled_conformal" in cfg.get(
        "calibration.secondary_threshold_methods"
    )
    assert cfg.get("evaluation.require_preregistration") is True


def test_write_systems_keeps_secondary_names_distinct(tmp_path):
    scored = _val().assign(set="open_sr_bench")
    paths = calibrate.write_systems(
        scored, {"pooled_conformal.tau99": 0.0}, "ens", str(tmp_path)
    )
    frame = io.read_parquet(paths[0])
    assert frame["system"].unique().tolist() == ["ens@pooled_conformal.tau99"]
    assert report.calibrated_ensembles(frame["system"]) == ["ens"]


def test_report_dir_tags_secondary_analysis(tmp_path):
    runs = str(tmp_path)
    assert report.report_dir(runs, "titan_test", 20) == os.path.join(
        runs, "reports", "titan_test_T20"
    )
    assert report.report_dir(runs, "titan_test", 20, "secondary") == (
        os.path.join(runs, "reports", "titan_test_T20_secondary")
    )
    with pytest.raises(ValueError, match="invalid report tag"):
        report.report_dir(runs, "titan_test", 20, "../primary")
    args = cli.build_parser().parse_args(["evaluate", "--tag", "secondary"])
    assert args.tag == "secondary"
    assert cli.build_parser().parse_args(["evaluate"]).tag is None


def test_secondary_evaluation_does_not_overwrite_primary(tmp_path, monkeypatch):
    """Regression: a secondary evaluate overwrote the primary leaderboard."""
    monkeypatch.setenv("OSB_DATA_ROOT", str(tmp_path / "data"))
    runs = str(tmp_path / "runs")
    frozen = tmp_path / "data" / "frozen" / "bench"
    frozen.mkdir(parents=True)
    tiers = _val(n_reviews=6).assign(in_tier_t20=True)
    io.write_parquet(tiers, str(frozen / "tiers.parquet"))
    totals = (
        tiers.groupby("review_id")["label_final"]
        .agg(n_total_pos="sum", n_total_neg=lambda y: int((y == 0).sum()))
        .reset_index()
        .assign(review_type="intervention")
    )
    io.write_parquet(totals, str(frozen / "review_totals.parquet"))
    for subdir, op in (("systems", "tau99"), ("systems_secondary", "p.tau99")):
        decisions = tiers[["review_id", "record_id", "score"]].assign(
            system=f"ens@{op}",
            set="bench",
            decision_keep=(tiers["score"] > 0).astype(int),
            is_valid_answer=True,
        )
        io.write_parquet(decisions, os.path.join(runs, subdir, "ens.parquet"))

    def evaluate(systems_dir, tag):
        return report.run(
            argparse.Namespace(
                config=config.DEFAULT_BENCHMARK_CONFIG,
                runs_dir=runs,
                benchmark="bench",
                tier=20,
                systems=[os.path.join(runs, systems_dir, "*.parquet")],
                reference=None,
                replicates=20,
                skip_preregistration_check=True,
                tag=tag,
            )
        )

    evaluate("systems", None)
    primary = os.path.join(runs, "reports", "bench_T20", "leaderboard.json")
    with open(primary, encoding="utf-8") as fh:
        before = fh.read()
    evaluate("systems_secondary", "secondary")
    with open(primary, encoding="utf-8") as fh:
        assert fh.read() == before
    secondary = io.read_json(
        os.path.join(runs, "reports", "bench_T20_secondary", "leaderboard.json")
    )
    assert secondary["tag"] == "secondary"
    assert [r["system"] for r in secondary["leaderboard"]] == ["ens@p.tau99"]
