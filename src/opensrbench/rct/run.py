"""`osb rct {corpus,train,score,systems}`.

corpus   [network]  Sample + cache the MEDLINE publication-type corpus.
train    [offline]  Decontaminate against val + protected splits, fit the
                    Cochrane-recipe classifier, write a model card.
score    [offline]  p_rct for every record in splits/records.parquet plus
                    the review's `requires_rct` flag ->
                    splits/rct_scores.parquet (used by `osb calibrate`).
systems  [offline]  Standalone baseline "cochrane_rct_filter": in reviews
                    that only admit RCTs, exclude records the classifier
                    places below its 99%-recall cut-off; keep the rest.
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from opensrbench import config
from opensrbench import pipeline
from opensrbench.rct import classifier as clf_lib
from opensrbench.rct import corpus as corpus_lib
from opensrbench.splits import assign
from opensrbench.text import design
from opensrbench.utils import io


def _dirs(paths: config.Paths) -> tuple[str, str]:
    root = os.path.join(paths.root, "rct")
    return root, os.path.join(root, "model")


def scores_path(paths: config.Paths) -> str:
    return os.path.join(paths.splits, "rct_scores.parquet")


def cmd_corpus(ctx: pipeline.Context) -> str:
    rct_cfg = ctx.cfg.get("rct")
    root, _ = _dirs(ctx.paths)
    with ctx.pubmed_store() as store:
        frame = corpus_lib.fetch_corpus(
            ctx.eutils(),
            store,
            tuple(int(y) for y in rct_cfg["years"]),
            int(rct_cfg["per_year_rct"]),
            int(rct_cfg["per_year_per_negative_stratum"]),
            int(ctx.cfg.get("network.efetch_batch_size", 200)),
        )
    return io.write_parquet(frame, os.path.join(root, "corpus_pmids.parquet"))


def cmd_train(ctx: pipeline.Context) -> dict:
    rct_cfg = ctx.cfg.get("rct")
    root, model_dir = _dirs(ctx.paths)
    pmids = io.read_parquet(os.path.join(root, "corpus_pmids.parquet"))
    with ctx.pubmed_store() as store:
        corpus = corpus_lib.attach_labels(pmids, store)
    if rct_cfg.get("crowd_csv"):
        crowd = pd.read_csv(rct_cfg["crowd_csv"], dtype={"pmid": str})
        crowd["pmid"] = crowd.get("pmid", pd.Series(dtype=str)).fillna(
            "crowd:" + crowd.index.astype(str)
        )
        crowd["stratum"] = "crowd"
        corpus = pd.concat(
            [corpus, crowd[["pmid", "title", "abstract", "label", "stratum"]]],
            ignore_index=True,
        )
    _, records, splits = pipeline.load_split_data(
        ctx.paths, ["review_id", "pmid", "title_fp", "abstract_fp"]
    )
    corpus, audit = corpus_lib.decontaminate(
        corpus,
        records,
        splits.set_index("review_id")["split"],
        tuple(ctx.cfg.get("splits.purge_against")),
    )
    parts = corpus_lib.split_corpus(corpus)
    model = clf_lib.CochraneRctClassifier(
        c1=float(rct_cfg.get("svm_c", 0.1)), c2=float(rct_cfg.get("svm_c", 0.1))
    )
    model.fit(
        parts["train"]["title"],
        parts["train"]["abstract"],
        parts["train"]["label"].to_numpy(),
        parts["calib"]["title"],
        parts["calib"]["abstract"],
        parts["calib"]["label"].to_numpy(),
        float(rct_cfg.get("target_recall", 0.99)),
    )
    model.card["test"] = model.evaluate(
        parts["test"]["title"],
        parts["test"]["abstract"],
        parts["test"]["label"].to_numpy(),
    )
    model.card["decontamination"] = audit
    model.card["label_source"] = "MEDLINE publication types" + (
        " + Cochrane Crowd" if rct_cfg.get("crowd_csv") else ""
    )
    model.save(model_dir)
    logging.info("RCT classifier test metrics: %s", model.card["test"])
    return model.card


def cmd_score(ctx: pipeline.Context, backend: str | None = None) -> str:
    rct_cfg = ctx.cfg.get("rct")
    backend = backend or rct_cfg.get("backend", "cochrane")
    _, model_dir = _dirs(ctx.paths)
    reviews, records, _ = pipeline.load_split_data(
        ctx.paths,
        ["review_id", "record_id", "title", "abstract", "publication_types"],
    )
    unique = records.drop_duplicates("record_id")
    if backend == "cochrane":
        probs = clf_lib.CochraneRctClassifier.load(model_dir).predict_proba(
            unique["title"].tolist(), unique["abstract"].tolist()
        )
    elif backend == "hf":
        probs = clf_lib.HfRctScorer(rct_cfg["hf_model"]).predict_proba(
            unique["title"].tolist(), unique["abstract"].tolist()
        )
    elif backend == "pubmed_pt":
        probs = clf_lib.PublicationTypeScorer().predict_proba(
            unique["publication_types"].fillna("").tolist()
        )
    else:
        raise ValueError(f"unknown RCT backend {backend!r}")
    p_of = dict(zip(unique["record_id"], probs))
    out = records[["review_id", "record_id"]].copy()
    out["p_rct"] = out["record_id"].map(p_of)
    out["pt_rct"] = (
        records["publication_types"]
        .fillna("")
        .map(design.has_rct_publication_type)
        .astype(int)
    )
    needs = {
        r: design.requires_rct(c, t)
        for r, c, t in zip(
            reviews["review_id"],
            reviews["criteria_list"],
            reviews["review_type"],
        )
    }
    out["requires_rct"] = out["review_id"].map(needs).fillna(False).astype(int)
    out["rct_backend"] = backend
    return io.write_parquet(out, scores_path(ctx.paths))


def cmd_systems(ctx: pipeline.Context) -> str:
    _, model_dir = _dirs(ctx.paths)
    cutoff = clf_lib.CochraneRctClassifier.load(model_dir).cutoff
    scores = io.read_parquet(scores_path(ctx.paths))
    splits = io.read_parquet(os.path.join(ctx.paths.splits, "splits.parquet"))
    frames = []
    for name, ids in assign.review_sets(splits).items():
        sub = scores[scores["review_id"].isin(ids)]
        if sub.empty:
            continue
        exclude = (
            (sub["requires_rct"] == 1)
            & sub["p_rct"].notna()
            & (sub["p_rct"] < cutoff)
        )
        frames.append(
            sub[["review_id", "record_id"]].assign(
                set=name,
                decision_keep=(~exclude).astype(int),
                is_valid_answer=True,
                cost_usd=0.0,
                system="cochrane_rct_filter",
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return io.write_parquet(
        out,
        os.path.join(ctx.paths.runs, "systems", "cochrane_rct_filter.parquet"),
    )


def rct_features(paths: config.Paths) -> pd.DataFrame | None:
    """Stacking features from splits/rct_scores.parquet (None if absent)."""
    path = scores_path(paths)
    if not os.path.exists(path):
        return None
    df = io.read_parquet(path)
    p = df["p_rct"].astype(float)
    df["rct_missing"] = p.isna().astype(int)
    df["rct_logit"] = np.log(
        (p.fillna(0.5).clip(1e-4, 1 - 1e-4))
        / (1 - p.fillna(0.5).clip(1e-4, 1 - 1e-4))
    )
    df["rct_x_requires"] = df["rct_logit"] * df["requires_rct"]
    df["pt_x_requires"] = df["pt_rct"] * df["requires_rct"]
    return df[
        [
            "review_id",
            "record_id",
            "rct_logit",
            "rct_missing",
            "requires_rct",
            "rct_x_requires",
            "pt_rct",
            "pt_x_requires",
        ]
    ]


RCT_FEATURES = (
    "rct_logit",
    "rct_missing",
    "requires_rct",
    "rct_x_requires",
    "pt_rct",
    "pt_x_requires",
)


def run(args: Any) -> Any:
    ctx = pipeline.Context.create(
        args.config, args.datasets_config, device=getattr(args, "device", None)
    )
    if args.rct_command == "corpus":
        return cmd_corpus(ctx)
    if args.rct_command == "train":
        return cmd_train(ctx)
    if args.rct_command == "score":
        return cmd_score(ctx, args.backend)
    if args.rct_command == "systems":
        return cmd_systems(ctx)
    raise ValueError(args.rct_command)
