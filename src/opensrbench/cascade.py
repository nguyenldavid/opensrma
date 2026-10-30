"""Two-stage hybrid cascade (protocol Section 4, last row).

Stage 1: the calibrated TITAN-Open score with validation-locked thresholds.
  score <  tau_low  (tau99.5)  -> auto-exclude
  score >= tau_high (tau93)    -> auto-include
  otherwise (gray zone)        -> Stage 2 verifier (Condition B LLM)
Stage 2 decisions come from an LLM decision file. A gray-zone record with no
valid verifier answer is kept (abstention rule). Only gray-zone calls are
charged to the cascade's cost.
"""

import os
from typing import Any

import pandas as pd

from opensrbench import config
from opensrbench.utils import io


def cascade_decisions(
    scored: pd.DataFrame,
    tau_low: float,
    tau_high: float,
    verifier: pd.DataFrame,
) -> pd.DataFrame:
    """Combines stage-1 scores and stage-2 decisions.

    Args:
      scored: review_id, record_id, score (and set).
      tau_low: Auto-exclude below this score.
      tau_high: Auto-include at or above this score.
      verifier: review_id, record_id, decision_keep, is_valid_answer
        [, cost_usd].

    Returns:
      Decision frame with a `stage` column ("auto_exclude", "auto_include",
      "verifier").
    """
    if tau_low > tau_high:
        raise ValueError("tau_low must not exceed tau_high")
    out = scored.copy()
    out["stage"] = "verifier"
    out.loc[out["score"] < tau_low, "stage"] = "auto_exclude"
    out.loc[out["score"] >= tau_high, "stage"] = "auto_include"
    cols = ["review_id", "record_id", "decision_keep", "is_valid_answer"]
    if "cost_usd" in verifier.columns:
        cols.append("cost_usd")
    merged = out.merge(
        verifier[cols].rename(
            columns={
                "decision_keep": "v_keep",
                "is_valid_answer": "v_valid",
                "cost_usd": "v_cost",
            }
        ),
        on=["review_id", "record_id"],
        how="left",
    )
    gray = merged["stage"] == "verifier"
    v_ok = merged["v_valid"].fillna(False).astype(bool)
    merged["decision_keep"] = 1
    merged.loc[merged["stage"] == "auto_exclude", "decision_keep"] = 0
    merged.loc[gray & v_ok, "decision_keep"] = merged.loc[
        gray & v_ok, "v_keep"
    ].astype(int)
    merged["is_valid_answer"] = True
    merged.loc[gray & ~v_ok, "is_valid_answer"] = False
    merged["cost_usd"] = 0.0
    if "v_cost" in merged.columns:
        merged.loc[gray, "cost_usd"] = merged.loc[gray, "v_cost"].fillna(0.0)
    return merged.drop(
        columns=[
            c for c in ("v_keep", "v_valid", "v_cost") if c in merged.columns
        ]
    )


def run(args: Any) -> str:
    bench_cfg = config.Config.load(args.config)
    paths = config.Paths.from_config(bench_cfg)
    runs_root = args.runs_dir or paths.runs
    cascade_cfg = bench_cfg.get("cascade")
    cal_dir = os.path.join(runs_root, "calibration", args.ensemble)
    locked = io.read_json(os.path.join(cal_dir, "locked_operating_points.json"))
    taus = locked["thresholds"]
    low = taus[args.low or cascade_cfg["low_threshold"]]
    high = taus[args.high or cascade_cfg["high_threshold"]]
    scored = io.read_parquet(os.path.join(cal_dir, "scored.parquet"))
    scored = scored[scored["set"] == args.benchmark]
    verifier_name = args.verifier or cascade_cfg["verifier_system"]
    verifier_file = os.path.join(
        runs_root, "systems", verifier_name.replace("/", "__") + ".parquet"
    )
    verifier = io.read_parquet(verifier_file)
    decisions = cascade_decisions(scored, low, high, verifier)
    system = f"cascade[{args.ensemble}->{verifier_name}]"
    decisions = decisions.assign(system=system)
    path = os.path.join(
        runs_root,
        "systems",
        f"cascade__{args.ensemble}__"
        f"{verifier_name.replace('/', '__')}.parquet",
    )
    io.write_parquet(
        decisions[
            [
                "system",
                "set",
                "review_id",
                "record_id",
                "decision_keep",
                "is_valid_answer",
                "cost_usd",
                "stage",
            ]
        ],
        path,
    )
    return path
