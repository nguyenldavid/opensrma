"""Resolution of per-backbone training specs from titan_ensemble.yaml."""

import copy
from typing import Any

from opensrbench import config


def deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merges `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def backbone_spec(ensemble_cfg: config.Config, backbone: str) -> dict[str, Any]:
    """Returns defaults merged with the backbone's overrides.

    Raises:
      KeyError: For an unknown backbone.
    """
    backbones = ensemble_cfg.get("backbones")
    if backbone not in backbones:
        raise KeyError(
            f"unknown backbone {backbone!r}; known: {sorted(backbones)}"
        )
    spec = deep_merge(ensemble_cfg.get("defaults"), backbones[backbone])
    spec["name"] = backbone
    spec.setdefault("n_folds", int(ensemble_cfg.get("n_folds", 5)))
    spec.setdefault("patience", 0)
    return spec


def ensemble_members(ensemble_cfg: config.Config, ensemble: str) -> list[str]:
    """Backbone names of an ensemble."""
    ensembles = ensemble_cfg.get("ensembles")
    if ensemble not in ensembles:
        raise KeyError(
            f"unknown ensemble {ensemble!r}; known: {sorted(ensembles)}"
        )
    return list(ensembles[ensemble])


def plan_runs(
    ensemble_cfg: config.Config, ensemble: str
) -> list[tuple[str, int]]:
    """(backbone, fold) pairs to train for an ensemble (Slurm array order)."""
    runs = []
    for backbone in ensemble_members(ensemble_cfg, ensemble):
        n_folds = int(backbone_spec(ensemble_cfg, backbone)["n_folds"])
        runs.extend((backbone, fold) for fold in range(n_folds))
    return runs


def run_name(backbone: str, fold: int) -> str:
    return f"{backbone}/fold{fold}"
