"""Guards that keep the Killarney L40S pilot within the protocol.

configs/killarney/*_pilot.yaml `extends:` the protocol configs. The pilot
may narrow the data to TITAN-SR + Chan 2024, train one model per backbone
and skip pool augmentation, but it must keep the protocol's calibration and
evaluation settings (threshold methods, pre-registration gate) unchanged.
slurm/pilot_l40s.sh runs these checks on the resolved configs before it
submits anything. They raise instead of `assert`ing, so they also hold
under `python -O`.
"""

from collections.abc import Sequence
from typing import Any

from opensrbench import config
from opensrbench.modeling import spec

PILOT_SOURCES = frozenset({"titan_sr", "chan2024"})
PILOT_ENSEMBLES = ("titan_open", "titan_sr_baseline")


class PilotConfigError(ValueError):
    """A pilot config deviates from the protocol where it must not."""


def check_benchmark(pilot: dict[str, Any], protocol: dict[str, Any]) -> None:
    """Checks the resolved pilot benchmark config against the protocol's.

    Args:
      pilot: Resolved pilot benchmark config (`config.load_yaml`).
      protocol: The protocol benchmark config it extends.

    Raises:
      PilotConfigError: If `calibration` or `evaluation` differ from the
        protocol or pool augmentation is enabled.
    """
    for key in ("calibration", "evaluation"):
        if pilot.get(key) != protocol.get(key):
            raise PilotConfigError(f"the pilot overrides `{key}`")
    if (pilot.get("cohort") or {}).get("augment_incomplete_pools"):
        raise PilotConfigError("cohort.augment_incomplete_pools must be off")


def check_datasets(pilot: dict[str, Any]) -> None:
    """Checks that exactly the pilot's sources are enabled.

    Raises:
      PilotConfigError: If the enabled sources are not `PILOT_SOURCES`.
    """
    enabled = {
        name
        for name, block in (pilot.get("sources") or {}).items()
        if (block or {}).get("enabled")
    }
    if enabled != PILOT_SOURCES:
        raise PilotConfigError(
            f"enabled sources {sorted(enabled)} != {sorted(PILOT_SOURCES)}"
        )


def check_ensembles(
    pilot: config.Config, ensembles: Sequence[str] = PILOT_ENSEMBLES
) -> list[str]:
    """Checks that every member of `ensembles` trains a single model.

    Args:
      pilot: Resolved pilot ensemble config.
      ensembles: Ensembles the pilot trains.

    Returns:
      Their backbones, de-duplicated, in order (the Slurm array order).

    Raises:
      PilotConfigError: If a member has `n_folds` != 1.
      KeyError: For an unknown ensemble or backbone.
    """
    backbones = []
    for ensemble in ensembles:
        for backbone in spec.ensemble_members(pilot, ensemble):
            n_folds = int(spec.backbone_spec(pilot, backbone)["n_folds"])
            if n_folds != 1:
                raise PilotConfigError(f"{backbone}: n_folds {n_folds} != 1")
            if backbone not in backbones:
                backbones.append(backbone)
    return backbones
