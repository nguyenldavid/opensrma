"""`osb evaluate`: leaderboard, paired tests, strata and Pareto data.

Inputs
  frozen/<benchmark>/tiers.parquet, review_totals.parquet  (from freeze)
  systems/*.parquet  decision files with columns
      system, review_id, record_id, decision_keep, is_valid_answer
      [, set, score, cost_usd]

Outputs (under <runs>/reports/<benchmark>_T<k>/)
  leaderboard.json / leaderboard.md   Table 1 (equal-review J, Sens, Spec,
                                      IPW workload, 95% family-bootstrap CIs,
                                      paired deltas vs the reference, Holm).
  by_review_type.csv                  Table 2 strata.
  ranking.csv                         Threshold-free per-review metrics on
                                      the whole cohort (TITAN-SR comparable:
                                      AUC, spec@95/99/99.5 recall, WSS@95).
  pareto.csv                          Figure 2: equal-review sensitivity vs
                                      work saved over a threshold sweep.
"""

import glob
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from opensrbench import config
from opensrbench.evaluation import bootstrap
from opensrbench.evaluation import ledger
from opensrbench.evaluation import metrics
from opensrbench.evaluation import ranking
from opensrbench.utils import io


def _load_systems(
    patterns: list[str], benchmark: str
) -> dict[str, pd.DataFrame]:
    """Loads decision files, keeping rows of `benchmark` when sets exist."""
    systems: dict[str, pd.DataFrame] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            df = pd.read_parquet(path)
            if "set" in df.columns:
                df = df[df["set"] == benchmark]
            for name, group in df.groupby("system"):
                systems[name] = group
    if not systems:
        raise FileNotFoundError(
            f"no system decisions for {benchmark} in {patterns}"
        )
    return systems


def pareto_curve(scored: pd.DataFrame, n_points: int = 200) -> pd.DataFrame:
    """Equal-review sensitivity vs work saved on whole cohorts."""
    pos = (scored["label_final"] == 1).to_numpy()
    review = scored["review_id"].to_numpy()
    rows = []
    for tau in np.unique(
        np.quantile(scored["score"], np.linspace(0, 1, n_points))
    ):
        keep = (scored["score"] >= tau).to_numpy()
        frame = pd.DataFrame(
            {"r": review, "keep": keep, "kp": keep & pos, "pos": pos}
        )
        per = frame.groupby("r").agg(
            kp=("kp", "sum"), pos=("pos", "sum"), work=("keep", "mean")
        )
        sens = (per["kp"] / per["pos"].where(per["pos"] > 0)).mean()
        rows.append(
            {
                "tau": float(tau),
                "sensitivity": float(sens),
                "work_saved": float(1 - per["work"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _fmt(triple: tuple[float, float, float]) -> str:
    point, lo, hi = triple
    return f"{point:.3f} ({lo:.3f}–{hi:.3f})"


def evaluate(
    benchmark_dir: str,
    tier: int,
    systems: dict[str, pd.DataFrame],
    reference: str | None,
    replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Scores every system on tier T_k of one frozen benchmark."""
    tiers = io.read_parquet(os.path.join(benchmark_dir, "tiers.parquet"))
    totals = io.read_parquet(
        os.path.join(benchmark_dir, "review_totals.parquet")
    )
    tier_df = tiers[tiers[f"in_tier_t{tier}"]][
        ["review_id", "record_id", "family_id", "label_final"]
    ]
    boot = bootstrap.FamilyBootstrap(
        tier_df["family_id"].unique(), replicates, seed
    )
    rows, dists, per_review_all = [], {}, []
    for name, decisions in systems.items():
        merged = metrics.apply_abstention_rule(tier_df, decisions)
        per_review = metrics.score_review_metrics(merged, totals)
        points = metrics.point_estimates(per_review)
        dist = boot.distributions(per_review)
        dists[name] = dist
        row = {
            "system": name,
            "reviews": int(per_review["review_id"].nunique()),
            "families": int(per_review["family_id"].nunique()),
            "invalid_rate": float(per_review["invalid_rate"].mean()),
            "cost_usd": float(
                merged.get("cost_usd", pd.Series([0.0])).fillna(0).sum()
            ),
        }
        for metric in bootstrap.METRICS:
            row[metric] = bootstrap.percentile_ci(
                points[metric], dist[metric], alpha
            )
        rows.append(row)
        per_review_all.append(per_review.assign(system=name))
    comparisons = {}
    if reference and reference in dists:
        ref_row = next(r for r in rows if r["system"] == reference)
        raw_p = {}
        for row in rows:
            if row["system"] == reference:
                continue
            comp = bootstrap.paired_difference(
                dists[row["system"]]["youdens_j"],
                dists[reference]["youdens_j"],
                row["youdens_j"][0] - ref_row["youdens_j"][0],
                alpha,
            )
            comparisons[row["system"]] = comp
            raw_p[row["system"]] = comp["p_value"]
        for name, p_adj in bootstrap.holm(raw_p).items():
            comparisons[name]["p_holm"] = p_adj
    elif reference:
        logging.warning("reference system %s not found", reference)
    rows.sort(key=lambda r: -r["youdens_j"][0])
    return {
        "tier": tier,
        "reference": reference,
        "leaderboard": rows,
        "comparisons": comparisons,
        "per_review": pd.concat(per_review_all, ignore_index=True),
    }


def markdown_table(result: dict[str, Any]) -> str:
    lines = [
        f"### Leaderboard on T{result['tier']} "
        f"(reference: {result['reference']})",
        "",
        "| System | Youden's J | Sensitivity | Specificity | IPW workload "
        "| ΔJ vs ref (95% CI) | p (Holm) | Invalid | Cost (USD) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["leaderboard"]:
        comp = result["comparisons"].get(row["system"])
        delta = (
            f"{comp['delta']:+.3f} ({comp['ci_low']:+.3f}, "
            f"{comp['ci_high']:+.3f})"
            if comp
            else "—"
        )
        p_holm = f"{comp['p_holm']:.4f}" if comp and "p_holm" in comp else "—"
        lines.append(
            f"| {row['system']} | {_fmt(row['youdens_j'])} | "
            f"{_fmt(row['sensitivity'])} | {_fmt(row['specificity'])} | "
            f"{_fmt(row['workload'])} | {delta} | {p_holm} | "
            f"{row['invalid_rate']:.3f} | {row['cost_usd']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def run(args: Any) -> dict[str, Any]:
    bench_cfg = config.Config.load(args.config)
    paths = config.Paths.from_config(bench_cfg)
    runs_root = args.runs_dir or paths.runs
    eval_cfg = bench_cfg.get("evaluation")
    tier = int(args.tier or bench_cfg.get("benchmark.primary_tier"))
    ledger.require_unblind(
        args.benchmark, bool(getattr(args, "unblind", False))
    )
    benchmark_dir = os.path.join(paths.frozen, args.benchmark)
    patterns = args.systems or [os.path.join(runs_root, "systems", "*.parquet")]
    systems = _load_systems(patterns, args.benchmark)
    if args.benchmark in ledger.PROTECTED_BENCHMARKS:
        entry = ledger.record_unblinding(
            runs_root, args.benchmark, tier, list(systems)
        )
        if not all(
            v["preregistered"] for v in entry["locked_threshold_files"].values()
        ):
            logging.warning(
                "UNBLINDING WITHOUT PRE-REGISTRATION: some locked threshold "
                "files have no identical copy in datasets/manifests/. Run "
                "`osb preregister` and commit before reporting results."
            )
    reference = args.reference or eval_cfg.get("reference_system")
    result = evaluate(
        benchmark_dir,
        tier,
        systems,
        reference,
        int(args.replicates or eval_cfg["bootstrap_replicates"]),
        int(eval_cfg["bootstrap_seed"]),
        float(eval_cfg.get("alpha", 0.05)),
    )
    out_dir = os.path.join(runs_root, "reports", f"{args.benchmark}_T{tier}")
    per_review = result.pop("per_review")
    io.write_json(result, os.path.join(out_dir, "leaderboard.json"))
    with open(
        os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8"
    ) as fh:
        if args.benchmark in ledger.DEV_BENCHMARKS:
            fh.write(
                "> DEV SET: thresholds were locked on these reviews; "
                "this is not a held-out estimate.\n\n"
            )
        fh.write(markdown_table(result))
    reviews = io.read_parquet(
        os.path.join(benchmark_dir, "review_totals.parquet")
    )
    strata = (
        per_review.merge(reviews[["review_id", "review_type"]], on="review_id")
        .groupby(["system", "review_type"])
        .agg(
            reviews=("review_id", "nunique"),
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
            workload=("workload", "mean"),
        )
    )
    strata.reset_index().to_csv(
        os.path.join(out_dir, "by_review_type.csv"), index=False
    )
    # Threshold-free metrics on whole cohorts for score-bearing systems.
    ranking_rows, pareto_rows = [], []
    seen_scores = set()
    for name, decisions in systems.items():
        if "score" not in decisions.columns or decisions["score"].isna().all():
            continue
        base = name.split("@")[0]
        if base in seen_scores:
            continue
        seen_scores.add(base)
        cohort = decisions.merge(
            io.read_parquet(os.path.join(benchmark_dir, "tiers.parquet"))[
                ["review_id", "record_id", "label_final"]
            ],
            on=["review_id", "record_id"],
        )
        summary = ranking.summarize_ranking(cohort, "score")
        ranking_rows.append({"system": base, **summary})
        pareto_rows.append(pareto_curve(cohort).assign(system=base))
    if ranking_rows:
        pd.DataFrame(ranking_rows).to_csv(
            os.path.join(out_dir, "ranking.csv"), index=False
        )
        pd.concat(pareto_rows).to_csv(
            os.path.join(out_dir, "pareto.csv"), index=False
        )
    logging.info("\n%s", markdown_table(result))
    return result
