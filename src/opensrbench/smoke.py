"""`osb smoke`: synthetic end-to-end run of the offline pipeline on CPU.

Generates a small synthetic corpus (legacy + post-cutoff harvested reviews
with lexical signal), then runs build -> split -> freeze -> train (tiny
model, 2 folds) -> predict -> calibrate -> fake verifier -> cascade ->
evaluate. It exercises every offline code path in about a minute and is
the recommended first command on a new machine or cluster account.
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import yaml

from opensrbench import config
from opensrbench import pipeline
from opensrbench import schema
from opensrbench.rct import classifier as rct_classifier
from opensrbench.rct import run as rct_run
from opensrbench.utils import io

_TOPICS = [
    "asthma",
    "diabetes",
    "stroke",
    "sepsis",
    "malaria",
    "glaucoma",
    "psoriasis",
    "migraine",
    "anemia",
    "hepatitis",
    "lupus",
    "gout",
]
_DESIGNS = [
    "randomized trial",
    "cohort study",
    "case report",
    "diagnostic accuracy study",
    "qualitative interviews",
]
_TYPES = ["intervention_ma", "dta", "prognostic", "qualitative", "prevalence"]
_FILLER = (
    "patients outcome measured results method data group analysis "
    "follow baseline sample hospital clinical score level rate"
).split()


def _words(rng, n):
    return " ".join(rng.choice(_FILLER, size=n))


def synthetic_corpus(
    seed: int = 0, n_legacy: int = 48, n_harvest: int = 36
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Returns {source: (reviews, records)} with learnable lexical signal."""
    rng = np.random.default_rng(seed)
    out = {}
    for source, n, years in (
        ("chan2024", n_legacy, (2015, 2023)),
        ("pmc_harvest", n_harvest, (2025, 2026)),
    ):
        reviews, records = [], []
        for i in range(n):
            topic = _TOPICS[i % len(_TOPICS)]
            design = _DESIGNS[i % len(_DESIGNS)]
            rtype = _TYPES[i % len(_TYPES)]
            year = int(rng.integers(years[0], years[1] + 1))
            month = int(rng.integers(1, 10 if year == 2026 else 13))
            review_id = f"{source}:R{i:03d}"
            reviews.append(
                {
                    "review_id": review_id,
                    "source": source,
                    "review_key": f"{source}:{i}",
                    "title": f"{topic} treatment {i}: a systematic review",
                    "objective": f"To assess {design}s of {topic} therapy.",
                    "selection_criteria": (
                        f"Inclusion criteria: (1) adults with {topic}; (2) "
                        f"{design}. Exclusion criteria: (1) animal studies; (2) "
                        f"children."
                    ),
                    "review_type": rtype,
                    "pub_date": f"{year}-{month:02d}-15",
                    "pool_complete": True,
                    "pool_tier": (
                        "tier2_hybrid"
                        if source == "pmc_harvest"
                        else "provided"
                    ),
                    "has_abs_labels": True,
                }
            )
            n_pos = int(rng.integers(6, 12))
            n_neg = int(rng.integers(110, 140))
            for j in range(n_pos + n_neg):
                pos = j < n_pos
                ft_excl = (not pos) and j < n_pos + 3
                if pos:
                    text = f"{design} in adults with {topic}"
                elif ft_excl:
                    text = f"{design} in children with {topic}"
                elif rng.random() < 0.3:
                    text = f"animal model of {topic}"
                else:
                    text = f"{rng.choice(_DESIGNS)} of {rng.choice(_TOPICS)}"
                title = f"{text} {j} {i}"
                records.append(
                    {
                        "review_id": review_id,
                        "record_id": schema.text_record_id(source, title, ""),
                        "title": title,
                        "abstract": f"{text}. {_words(rng, 70)}",
                        "pub_date": f"{year - 1}-01-01",
                        "label_final": int(pos),
                        "label_abs": int(pos or ft_excl),
                        "retrieval_rank": j if not pos else None,
                        "retrieval_source": "hybrid",
                    }
                )
        out[source] = (
            schema.make_reviews_frame(reviews),
            schema.make_records_frame(records),
        )
    return out


def _write_configs(workdir: str, base_config: str) -> tuple[str, str]:
    cfg = config.Config.load(base_config).with_overrides(
        {
            "paths.data_root": os.path.join(workdir, "data"),
            "paths.runs_root": os.path.join(workdir, "runs"),
            "cohort.dense.enabled": False,
            "splits.embedder": "tfidf",
            "splits.min_release_families": 5,
            "splits.min_release_appearances": 20,
            "splits.titan_sr_test_as_benchmark": False,
            "evaluation.bootstrap_replicates": 300,
            "evaluation.reference_system": "smoke@tau99",
        }
    )
    io.ensure_dir(workdir)
    bench_path = os.path.join(workdir, "benchmark_smoke.yaml")
    with open(bench_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.as_dict(), fh)
    datasets_path = os.path.join(workdir, "datasets_smoke.yaml")
    with open(datasets_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "sources": {
                    "pmc_harvest": {"enabled": True, "priority": 5},
                    "chan2024": {"enabled": True, "priority": 20},
                }
            },
            fh,
        )
    return bench_path, datasets_path


def _train_synthetic_rct(ctx: pipeline.Context) -> None:
    """Fits the Cochrane-recipe RCT classifier on synthetic text."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(600):
        rct = i % 3 == 0
        design = (
            "participants were randomly assigned in a randomized "
            "controlled trial"
            if rct
            else rng.choice(
                [
                    "retrospective cohort study",
                    "case control analysis",
                    "cross sectional survey",
                ]
            )
        )
        rows.append(
            {
                "title": (
                    f"Study {i} of {rng.choice(_TOPICS)} therapy " f"outcomes"
                ),
                "abstract": f"{design}. {_words(rng, 80)}",
                "label": int(rct),
            }
        )
    corpus = pd.DataFrame(rows)
    train_part, calib_part = corpus.iloc[:400], corpus.iloc[400:]
    model = rct_classifier.CochraneRctClassifier(min_df=1)
    # 90% target: 200 calibration records cannot certify 99% recall.
    model.fit(
        train_part["title"],
        train_part["abstract"],
        train_part["label"],
        calib_part["title"],
        calib_part["abstract"],
        calib_part["label"],
        target_recall=0.90,
    )
    model.save(os.path.join(ctx.paths.root, "rct", "model"))


def run(args: argparse.Namespace) -> dict:
    # pylint: disable=import-outside-toplevel
    from opensrbench import cascade
    from opensrbench.calibration import run as calibrate
    from opensrbench.evaluation import report
    from opensrbench.modeling import predict
    from opensrbench.modeling import train

    # pylint: enable=import-outside-toplevel
    workdir = os.path.abspath(args.workdir)
    bench_path, datasets_path = _write_configs(workdir, args.config)
    ctx = pipeline.Context.create(bench_path, datasets_path, device="none")
    for source, (reviews, records) in synthetic_corpus().items():
        folder = ctx.paths.interim(source)
        io.write_parquet(reviews, os.path.join(folder, "reviews.parquet"))
        io.write_parquet(records, os.path.join(folder, "records.parquet"))
    pipeline.run(ctx, ["build", "split", "freeze"])

    common = dict(
        config=bench_path,
        ensemble_config=args.ensemble_config,
        runs_dir=ctx.paths.runs,
        output_dir=ctx.paths.runs,
        hardware="cpu",
    )
    for fold in (0, 1):
        train.run(
            argparse.Namespace(
                backbone="tiny",
                fold=fold,
                max_steps=0,
                limit_reviews=0,
                override=None,
                **common,
            )
        )
        predict.run(
            argparse.Namespace(
                backbone="tiny",
                fold=fold,
                batch_size=None,
                sets=["val", "internal_test", "open_sr_bench", "prospective"],
                **common,
            )
        )
    calibrate.run(
        argparse.Namespace(ensemble="smoke", method=None, rct="off", **common)
    )
    # RCT classifier: Cochrane recipe on a synthetic, disjoint corpus, then
    # score + filter baseline + stacking features ("smoke+rct").
    _train_synthetic_rct(ctx)
    rct_run.cmd_score(ctx)
    rct_run.cmd_systems(ctx)
    calibrate.run(
        argparse.Namespace(ensemble="smoke", method=None, rct="on", **common)
    )

    # A fake "verifier LLM" so the cascade and LLM scoring paths run.
    tiers = io.read_parquet(
        os.path.join(ctx.paths.frozen, "open_sr_bench", "tiers.parquet")
    )
    rng = np.random.default_rng(1)
    t20 = tiers[tiers["in_tier_t20"]]
    keep = np.where(
        t20["label_final"] == 1,
        rng.random(len(t20)) < 0.97,
        rng.random(len(t20)) < 0.25,
    ).astype(int)
    fake = t20[["review_id", "record_id"]].assign(
        system="fake-llm/condition_b",
        set="open_sr_bench",
        decision_keep=keep,
        is_valid_answer=rng.random(len(t20)) > 0.02,
        cost_usd=0.001,
    )
    io.write_parquet(
        fake,
        os.path.join(
            ctx.paths.runs, "systems", "fake-llm__condition_b.parquet"
        ),
    )
    cascade.run(
        argparse.Namespace(
            ensemble="smoke",
            benchmark="open_sr_bench",
            verifier="fake-llm/condition_b",
            low=None,
            high=None,
            **common,
        )
    )
    result = report.run(
        argparse.Namespace(
            benchmark="open_sr_bench",
            tier=20,
            systems=None,
            reference="smoke@tau99",
            replicates=300,
            unblind=True,
            **common,
        )
    )
    report.run(
        argparse.Namespace(
            benchmark="internal_test",
            tier=20,
            systems=None,
            reference="smoke@tau99",
            replicates=300,
            unblind=True,
            **common,
        )
    )
    audit = io.read_json(os.path.join(ctx.paths.splits, "audit.json"))
    logging.info(
        "decontamination audit: %s", json.dumps(audit["decontamination"])
    )
    print(report.markdown_table(result))
    print(f"Smoke run complete. Outputs in {workdir}")
    return result
