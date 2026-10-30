"""Tests for `extends:` in the config loader and the pilot guards."""

import os

import pytest
import yaml

from opensrbench import config
from opensrbench import pilot
from opensrbench.modeling import spec

_CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")
_PILOT = os.path.join(_CONFIGS, "killarney")


def _write(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_extends_deep_merges_relative_to_the_overlay(tmp_path):
    (tmp_path / "sub").mkdir()
    _write(
        tmp_path / "base.yaml",
        {"a": {"x": 1, "y": [1, 2]}, "b": "${OSB_TEST_VAR:-dflt}"},
    )
    overlay = _write(
        tmp_path / "sub" / "overlay.yaml",
        {"extends": "../base.yaml", "a": {"y": [3]}, "c": True},
    )
    assert config.load_yaml(overlay) == {
        "a": {"x": 1, "y": [3]},
        "b": "${OSB_TEST_VAR:-dflt}",
        "c": True,
    }
    cfg = config.Config.load(overlay)
    assert cfg.get("a.x") == 1 and cfg.get("b") == "dflt"
    assert "extends" not in cfg.as_dict()


def test_extends_cycle_is_an_error(tmp_path):
    _write(tmp_path / "a.yaml", {"extends": "b.yaml"})
    _write(tmp_path / "b.yaml", {"extends": "a.yaml"})
    with pytest.raises(ValueError, match="cycle"):
        config.load_yaml(str(tmp_path / "a.yaml"))


def test_benchmark_pilot_loads_directly_with_protocol_calibration():
    """Regression: the overlay loaded without `calibration` (no extends)."""
    protocol = config.Config.load(os.path.join(_CONFIGS, "benchmark_v1.yaml"))
    pilot_cfg = config.Config.load(os.path.join(_PILOT, "benchmark_pilot.yaml"))
    for key in ("calibration", "evaluation", "splits", "paths"):
        assert pilot_cfg.get(key) == protocol.get(key)


def test_pilot_overlays_stay_within_the_protocol():
    load = config.load_yaml
    bench = load(os.path.join(_PILOT, "benchmark_pilot.yaml"))
    protocol = load(os.path.join(_CONFIGS, "benchmark_v1.yaml"))
    pilot.check_benchmark(bench, protocol)
    assert bench["calibration"] == protocol["calibration"]
    assert bench["evaluation"] == protocol["evaluation"]
    assert bench["cohort"]["augment_incomplete_pools"] is False
    # Apart from cohort augmentation, only network throughput differs.
    changed = {k for k in protocol if bench.get(k) != protocol[k]}
    assert changed == {"cohort", "network"}
    net = {
        k
        for k in protocol["network"]
        if bench["network"][k] != protocol["network"][k]
    }
    assert net == {"efetch_batch_size", "efetch_workers"}

    datasets = load(os.path.join(_PILOT, "datasets_pilot.yaml"))
    pilot.check_datasets(datasets)
    enabled = {k for k, v in datasets["sources"].items() if v.get("enabled")}
    assert enabled == {"titan_sr", "chan2024"}

    ens = config.Config.load(os.path.join(_PILOT, "titan_ensemble_pilot.yaml"))
    backbones = pilot.check_ensembles(ens)
    protocol_ens = config.Config.load(
        os.path.join(_CONFIGS, "titan_ensemble.yaml")
    )
    assert backbones == spec.ensemble_members(
        protocol_ens, "titan_open"
    ) + spec.ensemble_members(protocol_ens, "titan_sr_baseline")
    for backbone in backbones:
        got = spec.backbone_spec(ens, backbone)
        want = spec.backbone_spec(protocol_ens, backbone)
        assert got["n_folds"] == 1
        got.pop("n_folds"), want.pop("n_folds")
        assert got == want, backbone


def test_pilot_checks_reject_protocol_drift():
    protocol = {"calibration": {"m": "lcb"}, "evaluation": {"gate": True}}
    with pytest.raises(pilot.PilotConfigError, match="calibration"):
        pilot.check_benchmark({**protocol, "calibration": {"m": "x"}}, protocol)
    with pytest.raises(pilot.PilotConfigError, match="evaluation"):
        pilot.check_benchmark({"calibration": {"m": "lcb"}}, protocol)
    with pytest.raises(pilot.PilotConfigError, match="augment"):
        pilot.check_benchmark(
            {**protocol, "cohort": {"augment_incomplete_pools": True}},
            protocol,
        )
    sources = {"titan_sr": {"enabled": True}, "chan2024": {"enabled": True}}
    pilot.check_datasets({"sources": sources})
    with pytest.raises(pilot.PilotConfigError, match="enabled sources"):
        pilot.check_datasets(
            {"sources": {**sources, "synergy": {"enabled": True}}}
        )
    ens = config.Config(
        {
            "n_folds": 5,
            "defaults": {},
            "backbones": {"a": {"n_folds": 1}, "b": {}},
            "ensembles": {"one": ["a"], "two": ["a", "b"]},
        }
    )
    assert pilot.check_ensembles(ens, ["one"]) == ["a"]
    with pytest.raises(pilot.PilotConfigError, match="b: n_folds 5"):
        pilot.check_ensembles(ens, ["one", "two"])
