"""Tests for config inheritance and hardware profiles."""

import os

import pytest

from opensrbench import config
from opensrbench.modeling import spec

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_ENSEMBLE = os.path.join(_ROOT, "configs", "titan_ensemble.yaml")


@pytest.fixture(autouse=True)
def _cwd_repo(monkeypatch):
    monkeypatch.chdir(_ROOT)


def test_config_extends_deep_merges_parent(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {b: 1, c: 2}\nd: 3\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: {c: 9}\n")
    cfg = config.Config.load(str(tmp_path / "child.yaml"))
    assert cfg.get("a") == {"b": 1, "c": 9} and cfg.get("d") == 3


def test_local_benchmark_profile_only_shrinks_data_knobs():
    base = config.Config.load("configs/benchmark_v1.yaml")
    local = config.Config.load("configs/benchmark_local.yaml")
    assert local.get("cohort.augment_k") < base.get("cohort.augment_k")
    for key in ("benchmark.salt", "temporal", "splits", "calibration"):
        assert local.get(key) == base.get(key)


@pytest.mark.parametrize(
    "profile",
    ["rtx3080ti", "gpu_24gb", "killarney_l40s", "killarney_h100", "cpu"],
)
def test_effective_batch_is_preserved_on_every_profile(profile):
    ens = config.Config.load(_ENSEMBLE)
    world = (
        8
        if profile == "killarney_h100"
        else (4 if profile == "killarney_l40s" else 1)
    )
    for backbone in (
        "modernbert_large",
        "deberta_v3_large",
        "pubmedbert_base",
        "titan_sr_pubmedbert",
    ):
        resolved = spec.backbone_spec(ens, backbone, profile)
        optim = resolved["optim"]
        accum = spec.grad_accum_steps(optim, world)
        effective = optim["batch_size_per_gpu"] * world * accum
        assert effective >= optim["effective_batch_size"]
        assert effective < optim["effective_batch_size"] + (
            optim["batch_size_per_gpu"] * world
        )
        assert resolved["hardware"] == profile


def test_rtx3080ti_uses_checkpointing_and_refuses_4b_decoder():
    ens = config.Config.load(_ENSEMBLE)
    resolved = spec.backbone_spec(ens, "modernbert_large", "rtx3080ti")
    assert resolved["gradient_checkpointing"] is True
    assert resolved["optim"]["batch_size_per_gpu"] <= 4
    with pytest.raises(ValueError, match="24 GB"):
        spec.backbone_spec(ens, "qwen3_reranker_4b", "rtx3080ti")
    # The recipe itself is untouched by the hardware layer.
    base = spec.backbone_spec(ens, "modernbert_large")
    for key in ("loss", "sampler", "budgets", "max_seq_len", "architecture"):
        assert resolved[key] == base[key]


def test_lite_ensemble_plan():
    ens = config.Config.load(_ENSEMBLE)
    runs = spec.plan_runs(ens, "titan_open_lite")
    assert len(runs) == 9 and runs[0] == ("modernbert_base", 0)
