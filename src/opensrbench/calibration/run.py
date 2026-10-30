"""`osb calibrate`: combine ensemble members, lock thresholds, emit systems.

Outputs under <runs>/calibration/<ensemble>/:
  scored.parquet                 set, review_id, record_id, label_final, score
  locked_operating_points.json   thresholds + validation achievement + hash
and one decision file per operating point under <runs>/systems/:
  <ensemble>@tau93.parquet ... <ensemble>@tauJ.parquet
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from opensrbench import config
from opensrbench.calibration import stacking
from opensrbench.calibration import thresholds as thresholds_lib
from opensrbench.modeling import spec as spec_lib
from opensrbench.rct import run as rct_run
from opensrbench.utils import io


def _val_summary(val: pd.DataFrame, taus: dict[str, float]) -> dict:
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
        out[name] = {
            "tau": tau,
            "equal_review_sensitivity": float(per["sens"].mean()),
            "equal_review_specificity": float(per["spec"].mean()),
            "pooled_sensitivity": float(keep[val["label_final"] == 1].mean()),
        }
    return out


def write_systems(
    scored: pd.DataFrame, taus: dict[str, float], name: str, systems_dir: str
) -> list[str]:
    """Writes one decision file per operating point.

    Validation rows carry out-of-fold stacked scores and are kept for
    development reports (`osb evaluate --benchmark val`); they are flagged
    as a dev set there because thresholds were locked on them.
    """
    out_paths = []
    test = scored
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


def _use_rct(mode: str, combiner: str, paths: config.Paths) -> bool:
    """Whether to add RCT-classifier features (needs `osb rct score`)."""
    available = os.path.exists(rct_run.scores_path(paths))
    if mode == "on" and not available:
        raise FileNotFoundError(
            "--rct on needs splits/rct_scores.parquet; "
            "run `osb rct train` and `osb rct score`."
        )
    return combiner == "stacking" and mode != "off" and available


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
    name = args.ensemble
    use_rct = _use_rct(getattr(args, "rct", "auto"), combiner, paths)
    if use_rct:
        rct = rct_run.rct_features(paths)
        df = df.merge(rct, on=["review_id", "record_id"], how="left")
        for col in rct_run.RCT_FEATURES:
            df[col] = df[col].fillna(1.0 if col == "rct_missing" else 0.0)
        name = f"{args.ensemble}+rct"
    if combiner == "stacking":
        df, features = stacking.build_features(df, members)
        if use_rct:
            features += list(rct_run.RCT_FEATURES)
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
    taus = thresholds_lib.lock_thresholds(
        val["score"].to_numpy(),
        val["label_final"].to_numpy(),
        val["review_id"].to_numpy(),
        val["family_id"].to_numpy(),
        [float(t) for t in cal["target_sensitivities"]],
        method,
    )
    out_dir = os.path.join(runs_root, "calibration", name)
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
        "ensemble": name,
        "rct_features": bool(use_rct),
        "members": members,
        "combiner": combiner,
        "threshold_method": method,
        "thresholds": taus,
        "validation": _val_summary(val, taus),
        "validation_predictions_sha256": io.dataframe_sha256(
            val, ["review_id", "record_id", "score"]
        ),
        "n_val_reviews": int(val["review_id"].nunique()),
        "n_val_positives": int(val["label_final"].sum()),
    }
    io.write_json(locked, os.path.join(out_dir, "locked_operating_points.json"))
    write_systems(scored, taus, name, os.path.join(runs_root, "systems"))
    logging.info("locked thresholds (%s): %s", method, taus)
    return locked
