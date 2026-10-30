"""Whole-cohort inference for a trained (backbone, fold) checkpoint.

Scores every record of the requested review sets so the calibrator can
compute within-review statistics over full candidate pools:

  val            the validation/calibration split (thresholds are locked here)
  open_sr_bench  the frozen post-cutoff benchmark reviews
  titan_test     TITAN-SR's own held-out test reviews
  prospective    reviews published after the benchmark freeze
  eval           every eval-split review (extended benchmark)

Writes <run>/<backbone>/fold<k>/preds.parquet with review_id, record_id,
set, logit_final, logit_abs, logit_softmin.
"""

import logging
import os
from typing import Any

import pandas as pd

from opensrbench import config
from opensrbench import pipeline
from opensrbench.modeling import data as data_lib
from opensrbench.modeling import distributed
from opensrbench.modeling import inputs
from opensrbench.modeling import model as model_lib
from opensrbench.modeling import spec as spec_lib
from opensrbench.modeling import train as train_lib
from opensrbench.splits import assign
from opensrbench.utils import io

DEFAULT_SETS = (
    "val",
    "internal_test",
    "open_sr_bench",
    "titan_test",
    "prospective",
)


review_sets = assign.review_sets


def run(args: Any) -> str:
    """Scores the requested sets with one checkpoint; returns output path."""
    rank, _, world, device = distributed.setup()
    paths = config.Paths.from_config(config.Config.load(args.config))
    run_dir = os.path.join(
        args.output_dir or paths.runs, args.backbone, f"fold{args.fold}"
    )
    model, tokenizer, meta = model_lib.load_checkpoint(
        os.path.join(run_dir, "best"), str(device)
    )
    spec = meta["spec"]
    if getattr(args, "hardware", None):
        # Inference batch size follows the *current* machine, not the one
        # the checkpoint was trained on.
        ens_cfg = config.Config.load(args.ensemble_config)
        spec["eval_batch_size"] = spec_lib.backbone_spec(
            ens_cfg, args.backbone, args.hardware
        ).get("eval_batch_size")
    batch_size = args.batch_size or train_lib.eval_batch_size(spec)
    builder = inputs.InputBuilder(tokenizer, spec)
    reviews, records, splits = pipeline.load_split_data(
        paths, pipeline.MODEL_RECORD_COLUMNS
    )
    sets = review_sets(splits)
    frames = []
    for name in args.sets:
        ids = sets.get(name, set())
        if not ids:
            logging.info("set %s is empty; skipping", name)
            continue
        frame = data_lib.attach_review_text(
            records[records["review_id"].isin(ids)], reviews
        )
        frame["set"] = name
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"none of {args.sets} has reviews")
    frame = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["set", "review_id", "record_id"])
        .reset_index(drop=True)
    )
    scores = train_lib.score_frame(
        model,
        frame,
        builder,
        batch_size,
        device,
        rank,
        world,
        spec.get("precision", "bf16"),
        int(spec.get("num_workers", 2)),
    )
    out = frame[
        ["set", "review_id", "record_id", "label_final", "label_abs"]
    ].copy()
    for col in scores.columns:
        out[col] = scores[col].to_numpy()
    out["backbone"], out["fold"] = args.backbone, args.fold
    path = os.path.join(run_dir, "preds.parquet")
    if distributed.is_main(rank):
        io.write_parquet(out, path)
    distributed.teardown()
    return path
