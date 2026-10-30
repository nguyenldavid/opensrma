"""`osb calibrate`: combine ensemble members, lock thresholds, emit systems.

Outputs under <runs>/calibration/<ensemble>/:
  scored.parquet                 set, review_id, record_id, label_final, score
  locked_operating_points.json   primary thresholds + validation achievement
                                 + hash, and every pre-specified secondary
                                 threshold set under "secondary" (A19)
and one decision file per primary operating point under <runs>/systems/:
  <ensemble>@tau93.parquet ... <ensemble>@tauJ.parquet
Secondary sets go to <runs>/systems_secondary/ (e.g.
<ensemble>@pooled_conformal.tau99.parquet) so that the default
`osb evaluate` (and its Holm family) only sees the primary set.
"""

from collections.abc import Sequence
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from opensrbench import config
from opensrbench.calibration import stacking
from opensrbench.calibration import thresholds as thresholds_lib
from opensrbench.modeling import spec as spec_lib
from opensrbench.utils import io


def _val_summary(
    val: pd.DataFrame,
    taus: dict[str, float],
    targets: dict[str, float] | None = None,
) -> dict:
    """Validation achievement of each threshold.

    Args:
      val: Validation rows with review_id, label_final and score.
      taus: Operating-point name -> threshold.
      targets: Operating-point name -> target sensitivity, recorded next to
        the achieved equal-review sensitivity when given.

    Returns:
      Operating-point name -> {tau, [target_sensitivity],
      equal_review_sensitivity, equal_review_specificity,
      pooled_sensitivity}.
    """
    targets = targets or {}
    out = {}
    for name, tau in taus.items():
        keep = val["score"] >= tau
        per = (
            val.assign(keep=keep)
            .groupby("review_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "sens": (
                            g.loc[g["label_final"] == 1, "keep"].mean()
                            if (g["label_final"] == 1).any()
                            else np.nan
                        ),
                        "spec": (
                            1 - g.loc[g["label_final"] == 0, "keep"].mean()
                            if (g["label_final"] == 0).any()
                            else np.nan
                        ),
                    }
                ),
                include_groups=False,
            )
        )
        out[name] = {"tau": tau}
        if name in targets:
            out[name]["target_sensitivity"] = targets[name]
        out[name] |= {
            "equal_review_sensitivity": float(per["sens"].mean()),
            "equal_review_specificity": float(per["spec"].mean()),
            "pooled_sensitivity": float(keep[val["label_final"] == 1].mean()),
        }
    return out


def lock_operating_points(
    val: pd.DataFrame,
    targets: list[float],
    method: str,
    secondary_methods: Sequence[str] = (),
) -> dict[str, Any]:
    """Locks the primary and every pre-specified secondary threshold set.

    Only the primary set ("thresholds") is used by `osb cascade` and the
    default `osb evaluate`. Secondary sets (protocol amendment A19) are
    locked on the same validation scores at the same time, so they are
    pre-registered together with the primary set and cannot be chosen post
    hoc. tauJ does not depend on the method and is only kept in the primary
    set.

    Args:
      val: Validation rows with score, label_final, review_id, family_id.
      targets: Target sensitivities (e.g. [0.93, ..., 0.995]).
      method: Primary threshold method (see calibration/thresholds.py).
      secondary_methods: Further methods to lock; the primary is skipped.

    Returns:
      {threshold_method, target_sensitivities, thresholds, validation,
      secondary: {method: {thresholds, validation}}}. "validation" holds the
      achieved equal-review sensitivity per target.
    """
    args = (
        val["score"].to_numpy(),
        val["label_final"].to_numpy(),
        val["review_id"].to_numpy(),
        val["family_id"].to_numpy(),
        targets,
    )
    op_targets = {thresholds_lib.op_name(t): t for t in targets}
    taus = thresholds_lib.lock_thresholds(*args, method)
    locked = {
        "threshold_method": method,
        "target_sensitivities": targets,
        "thresholds": taus,
        "validation": _val_summary(val, taus, op_targets),
        "secondary": {},
    }
    for other in secondary_methods:
        if other == method:
            continue
        other_taus = thresholds_lib.lock_thresholds(*args, other)
        other_taus.pop("tauJ", None)
        locked["secondary"][other] = {
            "thresholds": other_taus,
            "validation": _val_summary(val, other_taus, op_targets),
        }
    return locked


def write_systems(
    scored: pd.DataFrame, taus: dict[str, float], name: str, systems_dir: str
) -> list[str]:
    """Writes one decision file per operating point (non-val sets)."""
    out_paths = []
    test = scored[scored["set"] != "val"]
    for op, tau in taus.items():
        system = f"{name}@{op}"
        frame = test[["set", "review_id", "record_id", "score"]].assign(
            system=system,
            decision_keep=(test["score"] >= tau).astype(int),
            is_valid_answer=True,
            cost_usd=0.0,
        )
        out_paths.append(
            io.write_parquet(
                frame, os.path.join(systems_dir, f"{system}.parquet")
            )
        )
    return out_paths


def run(args: Any) -> dict:
    bench_cfg = config.Config.load(args.config)
    ens_cfg = config.Config.load(args.ensemble_config)
    paths = config.Paths.from_config(bench_cfg)
    runs_root = args.runs_dir or paths.runs
    members = spec_lib.ensemble_members(ens_cfg, args.ensemble)
    combiner = ens_cfg.get(f"combiners.{args.ensemble}", "stacking")
    cal = bench_cfg.get("calibration")
    method = args.method or cal["threshold_method"]
    df = stacking.collect_predictions(runs_root, members)
    splits = io.read_parquet(os.path.join(paths.splits, "splits.parquet"))
    df["family_id"] = df["review_id"].map(
        splits.set_index("review_id")["family_id"]
    )
    is_val = (df["set"] == "val").to_numpy()
    if not is_val.any():
        raise RuntimeError(
            "no validation predictions; predict with --sets val,... first"
        )
    if combiner == "stacking":
        df, features = stacking.build_features(df, members)
        oof, model = stacking.fit_cross_fitted(
            df[is_val],
            features,
            df.loc[is_val, "family_id"].to_numpy(),
            int(cal["cross_fit_folds"]),
            float(cal["meta_c"]),
            float(cal["positive_class_weight"]),
        )
        score = np.zeros(len(df))
        score[is_val] = oof
        if (~is_val).any():
            score[~is_val] = model.predict_proba(
                df.loc[~is_val, features].to_numpy(np.float64)
            )[:, 1]
    elif combiner == "mean_sigmoid":
        score = stacking.mean_sigmoid(df, members)
    else:
        raise ValueError(f"unknown combiner {combiner!r}")
    df["score"] = score
    val = df[is_val]
    operating_points = lock_operating_points(
        val,
        [float(t) for t in cal["target_sensitivities"]],
        method,
        list(cal.get("secondary_threshold_methods") or []),
    )
    taus = operating_points["thresholds"]
    out_dir = os.path.join(runs_root, "calibration", args.ensemble)
    scored = df[
        [
            "set",
            "review_id",
            "record_id",
            "label_final",
            "label_abs",
            "family_id",
            "score",
        ]
    ]
    io.write_parquet(scored, os.path.join(out_dir, "scored.parquet"))
    locked = {
        "ensemble": args.ensemble,
        "members": members,
        "combiner": combiner,
        **operating_points,
        "validation_predictions_sha256": io.dataframe_sha256(
            val, ["review_id", "record_id", "score"]
        ),
        "n_val_reviews": int(val["review_id"].nunique()),
        "n_val_positives": int(val["label_final"].sum()),
    }
    io.write_json(locked, os.path.join(out_dir, "locked_operating_points.json"))
    write_systems(
        scored, taus, args.ensemble, os.path.join(runs_root, "systems")
    )
    for other, entry in operating_points["secondary"].items():
        write_systems(
            scored,
            {f"{other}.{op}": tau for op, tau in entry["thresholds"].items()},
            args.ensemble,
            os.path.join(runs_root, "systems_secondary"),
        )
        logging.info(
            "secondary thresholds (%s): %s", other, entry["thresholds"]
        )
    logging.info("locked thresholds (%s): %s", method, taus)
    return locked
