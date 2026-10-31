"""One arm of the backbone screen (docs/BACKBONE_SCREEN.md).

Trains one backbone on fold 0 with a step budget, skips checkpoint writes
(disk space), and appends the per-review holdout AUC to
runs/screen/results.jsonl.

  python scripts/screen_arm.py --backbone biomedbert_large --seed 1234
"""

import argparse
import json
import os

import pandas as pd

from opensrbench.evaluation import ranking
from opensrbench.modeling import model as model_lib
from opensrbench.modeling import train

OUT = "runs/screen"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--config", default="configs/local/benchmark_3080ti.yaml"
    )
    parser.add_argument(
        "--ensemble-config", default="configs/local/titan_ensemble_3080ti.yaml"
    )
    args = parser.parse_args()

    name = f"{args.backbone}_s{args.seed}"
    model_lib.save_checkpoint = lambda *a, **k: None  # Keep disk free.
    train_args = argparse.Namespace(
        config=args.config,
        ensemble_config=args.ensemble_config,
        backbone=args.backbone,
        fold=0,
        max_steps=args.max_steps,
        limit_reviews=None,
        override=[
            "optim.ema_decay=0",
            # Same global batch (32) and memory setup for every candidate.
            "optim.batch_size_per_gpu=16",
            "optim.grad_accum=2",
            "gradient_checkpointing=true",
            f"seed={args.seed}",
        ],
        output_dir=os.path.join(OUT, name),
    )
    summary = train.run(train_args)
    run_dir = os.path.join(OUT, name, args.backbone, "fold0")
    preds = pd.read_parquet(os.path.join(run_dir, "holdout_preds.parquet"))
    stats = ranking.summarize_ranking(
        preds[["review_id", "label_final"]].assign(score=preds["logit_final"])
    )
    result = {
        "backbone": args.backbone,
        "seed": args.seed,
        "steps": summary["steps"],
        "minutes": round(summary["minutes"], 1),
        "auc_per_review": round(stats["auc"], 4),
        "spec_at_99_recall": round(stats["spec_at_99_recall"], 4),
        "holdout_reviews": int(preds["review_id"].nunique()),
    }
    print("RESULT", json.dumps(result))
    with open(os.path.join(OUT, "results.jsonl"), "a") as fh:
        fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
