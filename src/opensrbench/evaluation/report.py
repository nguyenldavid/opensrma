"""`osb evaluate`: leaderboard, paired tests, strata and Pareto data.

Inputs
  frozen/<benchmark>/tiers.parquet, review_totals.parquet  (from freeze)
  systems/*.parquet  decision files with columns
      system, review_id, record_id, decision_keep, is_valid_answer
      [, set, score, cost_usd]

Outputs (under <runs>/reports/<benchmark>_T<k>/, or <benchmark>_T<k>_<tag>/
with --tag, e.g. "secondary" for the A19 secondary threshold set, so a
secondary analysis never overwrites the primary report)
  leaderboard.json / leaderboard.md   Table 1 (equal-review J, Sens, Spec,
                                      IPW workload, 95% family-bootstrap CIs,
                                      paired deltas vs the reference, Holm).
  by_review_type.csv                  Table 2 strata.
  ranking.csv                         Threshold-free per-review metrics on
                                      the whole cohort (TITAN-SR comparable:
                                      AUC, spec@95/99/99.5 recall, WSS@95).
  pareto.csv                          Figure 2: equal-review sensitivity vs
                                      work saved over a threshold sweep.

Pre-registration gate (protocol §7): before any metric is computed, every
calibrated ensemble behind the evaluated systems (`<ensemble>@<op>` and
`cascade[<ensemble>->...]`) must have its
<runs>/calibration/<ensemble>/locked_operating_points.json committed,
byte-identical, in git HEAD under evaluation.preregistration_dir
(slurm/preregister.sh does this). The outcome is recorded in
leaderboard.json under "preregistration".
"""

from collections.abc import Iterable
import glob
import hashlib
import logging
import os
import re
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from opensrbench import config
from opensrbench.evaluation import bootstrap
from opensrbench.evaluation import metrics
from opensrbench.evaluation import ranking
from opensrbench.utils import io

_CASCADE_SYSTEM = re.compile(r"cascade\[(?P<ensemble>.+?)->")
_REPORT_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class PreregistrationError(RuntimeError):
    """Locked operating points were not committed before evaluation."""


def calibrated_ensembles(system_names: Iterable[str]) -> list[str]:
    """Returns the calibrated ensembles whose thresholds the systems use.

    `osb calibrate` names systems `<ensemble>@<op>` and `osb cascade` names
    them `cascade[<ensemble>-><verifier>]`; LLM systems (`<model>/<cond>`)
    use no locked thresholds.

    Args:
      system_names: Names of the systems being evaluated.

    Returns:
      Sorted unique ensemble names.
    """
    out = set()
    for name in system_names:
        match = _CASCADE_SYSTEM.match(name)
        if match:
            out.add(match.group("ensemble"))
        elif "@" in name:
            out.add(name.split("@", 1)[0])
    return sorted(out)


def _git(args: list[str], cwd: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, check=True
        ).stdout
    except FileNotFoundError as err:
        raise PreregistrationError("git is not installed") from err
    except subprocess.CalledProcessError as err:
        raise PreregistrationError(
            f"git {' '.join(args)} failed in {cwd}: "
            f"{err.stderr.decode(errors='replace').strip()}"
        ) from err


def committed_file_hashes(directory: str) -> tuple[str, dict[str, str]]:
    """SHA-256 of every file committed in git HEAD under `directory`.

    Content comes from the HEAD blobs, not the working tree, so a copy that
    is untracked, only staged, or modified after the commit does not count.

    Args:
      directory: A directory inside a git work tree.

    Returns:
      (HEAD commit id, {repo-relative path: sha256 of the committed bytes}).

    Raises:
      PreregistrationError: If `directory` is not in a git work tree, the
        repository has no commits, or git fails.
    """
    if not os.path.isdir(directory):
        raise PreregistrationError(f"{directory} does not exist")
    toplevel = _git(["rev-parse", "--show-toplevel"], directory).decode()
    toplevel = toplevel.strip()
    head = _git(["rev-parse", "--verify", "HEAD^{commit}"], toplevel)
    rel = os.path.relpath(os.path.realpath(directory), toplevel)
    listing = _git(["ls-tree", "-r", "-z", "HEAD", "--", rel], toplevel)
    hashes = {}
    for entry in filter(None, listing.split(b"\0")):
        meta, path = entry.split(b"\t", 1)
        _, kind, oid = meta.split()
        if kind != b"blob":
            continue
        blob = _git(["cat-file", "blob", oid.decode()], toplevel)
        hashes[path.decode()] = hashlib.sha256(blob).hexdigest()
    return head.decode().strip(), hashes


def check_preregistration(
    runs_root: str, ensembles: Iterable[str], manifests_dir: str
) -> dict[str, Any]:
    """Verifies that locked operating points were committed before use.

    Args:
      runs_root: Runs root holding calibration/<ensemble>/.
      ensembles: Calibrated ensembles being evaluated.
      manifests_dir: Pre-registration directory (datasets/manifests).

    Returns:
      {"verified": True, "commit": HEAD id, "ensembles": {ensemble:
      {"locked_sha256", "committed_as"}}}.

    Raises:
      PreregistrationError: If any ensemble's locked_operating_points.json is
        missing or matches no file committed in HEAD under `manifests_dir`.
    """
    ensembles = list(ensembles)
    problems, verified = [], {}
    commit, committed = (None, {})
    if ensembles:
        commit, committed = committed_file_hashes(manifests_dir)
    for ensemble in ensembles:
        locked = os.path.join(
            runs_root, "calibration", ensemble, "locked_operating_points.json"
        )
        if not os.path.isfile(locked):
            problems.append(f"{ensemble}: {locked} does not exist")
            continue
        with open(locked, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        matches = sorted(p for p, h in committed.items() if h == digest)
        if not matches:
            problems.append(
                f"{ensemble}: {locked} (sha256 {digest[:16]}...) matches no "
                f"file committed in HEAD under {manifests_dir}"
            )
            continue
        verified[ensemble] = {"locked_sha256": digest, "committed_as": matches}
    if problems:
        raise PreregistrationError(
            "refusing to evaluate: locked operating points are not "
            "pre-registered (protocol §7).\n  "
            + "\n  ".join(problems)
            + "\nCommit them first: bash slurm/preregister.sh "
            + " ".join(ensembles)
        )
    return {"verified": True, "commit": commit, "ensembles": verified}


def _preregistration_gate(
    args: Any, eval_cfg: dict[str, Any], runs_root: str, names: Iterable[str]
) -> dict[str, Any]:
    """Runs the pre-registration check unless explicitly disabled."""
    ensembles = calibrated_ensembles(names)
    if getattr(args, "skip_preregistration_check", False):
        reason = "--skip-preregistration-check"
    elif not eval_cfg.get("require_preregistration", True):
        reason = "evaluation.require_preregistration: false"
    else:
        return check_preregistration(
            runs_root,
            ensembles,
            os.path.abspath(
                eval_cfg.get("preregistration_dir", "datasets/manifests")
            ),
        )
    logging.warning(
        "PRE-REGISTRATION NOT VERIFIED (%s) for %s. These results are not "
        "protocol results.",
        reason,
        ", ".join(ensembles) or "no calibrated ensembles",
    )
    return {"verified": False, "skipped_by": reason, "ensembles": ensembles}


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
    lines = []
    prereg = result.get("preregistration")
    if prereg and not prereg.get("verified"):
        lines += [
            "> **Pre-registration NOT verified** "
            f"({prereg.get('skipped_by')}): not a protocol result.",
            "",
        ]
    lines += [
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


def report_dir(
    runs_root: str, benchmark: str, tier: int, tag: str | None = None
) -> str:
    """Returns <runs>/reports/<benchmark>_T<tier>[_<tag>].

    Args:
      runs_root: Runs root.
      benchmark: Frozen benchmark name.
      tier: Tier k of T_k.
      tag: Optional analysis label (e.g. "secondary"); None or "" for the
        primary analysis.

    Raises:
      ValueError: If `tag` is not a plain name (letters, digits, `_.-`).
    """
    name = f"{benchmark}_T{tier}"
    if tag:
        if not _REPORT_TAG.fullmatch(tag):
            raise ValueError(f"invalid report tag {tag!r}")
        name += f"_{tag}"
    return os.path.join(runs_root, "reports", name)


def run(args: Any) -> dict[str, Any]:
    bench_cfg = config.Config.load(args.config)
    paths = config.Paths.from_config(bench_cfg)
    runs_root = args.runs_dir or paths.runs
    eval_cfg = bench_cfg.get("evaluation")
    tier = int(args.tier or bench_cfg.get("benchmark.primary_tier"))
    benchmark_dir = os.path.join(paths.frozen, args.benchmark)
    tag = getattr(args, "tag", None)
    out_dir = report_dir(runs_root, args.benchmark, tier, tag)
    patterns = args.systems or [os.path.join(runs_root, "systems", "*.parquet")]
    systems = _load_systems(patterns, args.benchmark)
    preregistration = _preregistration_gate(args, eval_cfg, runs_root, systems)
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
    result["preregistration"] = preregistration
    if tag:
        result["tag"] = tag
    per_review = result.pop("per_review")
    io.write_json(result, os.path.join(out_dir, "leaderboard.json"))
    with open(
        os.path.join(out_dir, "leaderboard.md"), "w", encoding="utf-8"
    ) as fh:
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
