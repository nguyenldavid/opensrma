"""Streams finished and running training logs to Weights & Biases.

A sidecar: it never touches training. Every run directory below the given
roots that holds a `train_log.jsonl` becomes one W&B run (stable id, so a
restart resumes it); new step lines are logged as they appear, holdout
metrics at each epoch end, and `summary.json` / screen results as run
summary once the run has finished. Only training metrics and run settings
are sent (no text, labels or predictions).

  source ~/.config/opensrma/wandb.env      # WANDB_API_KEY, never committed
  python scripts/wandb_sync.py --root runs_v2=/mnt/FNX/.../runs_v2 \
      --root screen=runs/screen --project opensrma
"""

import argparse
import glob
import hashlib
import json
import os
import time

import wandb


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}/{key}" if prefix else key, inner, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value


def _sync_run(group: str, run_dir: str, project: str, state: dict) -> None:
    log_path = os.path.join(run_dir, "train_log.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")
    key = os.path.abspath(run_dir)
    done = os.path.exists(summary_path)
    entry = state.setdefault(key, {"lines": 0, "finished": False})
    with open(log_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    if entry["finished"] or (len(lines) <= entry["lines"] and not done):
        return
    parts = os.path.normpath(run_dir).split(os.sep)
    backbone, fold = parts[-2], parts[-1]
    name = (
        f"{group}/{parts[-3]}" if group == "screen" else f"{group}/{backbone}"
    )
    run = wandb.init(
        project=project,
        id=hashlib.sha1(key.encode()).hexdigest()[:16],
        name=name,
        group=group,
        job_type="train",
        config={"backbone": backbone, "fold": fold, "run_dir": key},
        resume="allow",
        reinit="finish_previous",
    )
    last_step = entry.get("last_step", 0)
    for line in lines[entry["lines"] :]:
        row = json.loads(line)
        metrics: dict = {}
        if "step" in row:
            last_step = int(row["step"])
            _flatten(
                "train", {k: v for k, v in row.items() if k != "step"}, metrics
            )
        else:
            _flatten("", row, metrics)
        run.log(metrics, step=last_step)
    entry["lines"], entry["last_step"] = len(lines), last_step
    if done:
        with open(summary_path, encoding="utf-8") as fh:
            summary: dict = {}
            _flatten("final", json.load(fh), summary)
        run.summary.update(summary)
        entry["finished"] = True
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        action="append",
        required=True,
        help="group=directory (repeatable)",
    )
    parser.add_argument("--project", default="opensrma")
    parser.add_argument("--state", default="logs/wandb_sync_state.json")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    state = {}
    if os.path.exists(args.state):
        with open(args.state, encoding="utf-8") as fh:
            state = json.load(fh)
    while True:
        for spec in args.root:
            group, root = spec.split("=", 1)
            for log in sorted(
                glob.glob(f"{root}/**/train_log.jsonl", recursive=True)
            ):
                try:
                    _sync_run(group, os.path.dirname(log), args.project, state)
                except Exception as error:  # pylint: disable=broad-except
                    print(f"!!! wandb sync {log}: {error}", flush=True)
        with open(args.state, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
