"""Resolution of per-backbone training specs.

A spec is resolved in four layers (later wins):

  1. `defaults` in configs/titan_ensemble.yaml
  2. `backbones.<name>` in configs/titan_ensemble.yaml
  3. `all` in the hardware profile (configs/hardware/<profile>.yaml)
  4. `backbones.<name>` in the hardware profile

Hardware profiles only change *how* a model is trained (batch size,
gradient checkpointing, sequence length caps, workers), never the recipe,
and the effective batch size is preserved through gradient accumulation,
so results are comparable across Killarney H100/L40S nodes and a single
12 GB RTX 3080 Ti.
"""

import math
import os
from typing import Any

from opensrbench import config

DEFAULT_HARDWARE = "auto"


def _hardware_dir() -> str:
    """configs/hardware under the CWD, else under the source checkout."""
    local = os.path.join("configs", "hardware")
    if os.path.isdir(local):
        return local
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "..", "configs", "hardware")
    )


def deep_merge(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """Recursively merges `override` into a copy of `base`."""
    return config.deep_merge(base, override)


def hardware_profile(name: str | None) -> dict[str, Any]:
    """Loads configs/hardware/<name>.yaml ("auto" picks by detected GPU)."""
    name = name or os.environ.get("OSB_HARDWARE") or DEFAULT_HARDWARE
    if name == "auto":
        name = detect_hardware()
    folder = _hardware_dir()
    path = (
        name if name.endswith(".yaml") else os.path.join(folder, f"{name}.yaml")
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"hardware profile {path} not found; choose from "
            f"{sorted(f[:-5] for f in os.listdir(folder))}"
        )
    profile = config.Config.load(path).as_dict()
    profile.setdefault("name", os.path.basename(path)[:-5])
    return profile


def detect_hardware() -> str:
    """Maps the visible GPU to a profile name (falls back to "cpu")."""
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    gpu = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu or "a100" in gpu or gib >= 70:
        return "killarney_h100"
    if gib >= 40:
        return "killarney_l40s"
    if gib >= 20:
        return "gpu_24gb"
    return "rtx3080ti"


def backbone_spec(
    ensemble_cfg: config.Config,
    backbone: str,
    hardware: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Returns the fully resolved spec for one backbone.

    Args:
      ensemble_cfg: configs/titan_ensemble.yaml.
      backbone: Backbone key.
      hardware: Profile name, loaded profile dict, or None to skip the
        hardware layers (used by planning code that only needs n_folds).

    Raises:
      KeyError: For an unknown backbone.
      ValueError: If the hardware profile marks the backbone unsupported.
    """
    backbones = ensemble_cfg.get("backbones")
    if backbone not in backbones:
        raise KeyError(
            f"unknown backbone {backbone!r}; known: " f"{sorted(backbones)}"
        )
    spec = deep_merge(ensemble_cfg.get("defaults"), backbones[backbone])
    if hardware is not None:
        profile = (
            hardware
            if isinstance(hardware, dict)
            else hardware_profile(hardware)
        )
        spec = deep_merge(spec, profile.get("all", {}))
        override = profile.get("backbones", {}).get(backbone, {})
        if override.get("unsupported"):
            raise ValueError(
                f"{backbone} is not supported on hardware "
                f"profile {profile['name']}: "
                f"{override['unsupported']}"
            )
        spec = deep_merge(spec, override)
        spec["hardware"] = profile["name"]
    spec["name"] = backbone
    spec.setdefault("n_folds", int(ensemble_cfg.get("n_folds", 5)))
    spec.setdefault("patience", 0)
    return spec


def grad_accum_steps(optim: dict[str, Any], world_size: int) -> int:
    """Accumulation steps that preserve `effective_batch_size`.

    Falls back to an explicit `grad_accum` when no effective batch is set.
    """
    per_step = int(optim["batch_size_per_gpu"]) * max(1, world_size)
    target = optim.get("effective_batch_size")
    if target:
        return max(1, int(math.ceil(int(target) / per_step)))
    return max(1, int(optim.get("grad_accum", 1)))


def ensemble_members(ensemble_cfg: config.Config, ensemble: str) -> list[str]:
    """Backbone names of an ensemble."""
    ensembles = ensemble_cfg.get("ensembles")
    if ensemble not in ensembles:
        raise KeyError(
            f"unknown ensemble {ensemble!r}; known: " f"{sorted(ensembles)}"
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
