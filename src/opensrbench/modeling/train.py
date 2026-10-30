"""DDP training of one TITAN-Open backbone on one family fold.

Launch (single node, 8 GPUs):
  torchrun --nproc_per_node=8 -m opensrbench.cli train \
      --backbone modernbert_large --fold 0

Outputs (under runs/<ensemble-agnostic>/<backbone>/fold<k>/):
  best/            Best checkpoint by the early-stopping metric on the
                   held-out fold (safetensors + tokenizer + screener.json).
  holdout_preds.parquet  Out-of-fold scores of the best checkpoint.
  train_log.jsonl  Step/epoch metrics.  summary.json  Final summary.
"""

import contextlib
import json
import logging
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import parallel as torch_parallel
from torch.utils import data as torch_data
import yaml

from opensrbench import config
from opensrbench import pipeline
from opensrbench.evaluation import ranking
from opensrbench.modeling import data as data_lib
from opensrbench.modeling import distributed
from opensrbench.modeling import inputs
from opensrbench.modeling import losses
from opensrbench.modeling import model as model_lib
from opensrbench.modeling import sampler as sampler_lib
from opensrbench.modeling import spec as spec_lib
from opensrbench.utils import io


class Ema:
    """Exponential moving average of trainable parameters."""

    def __init__(self, module: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in module.named_parameters()
            if p.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for name, param in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(param.detach(), 1.0 - self.decay)

    @contextlib.contextmanager
    def applied(self, module: torch.nn.Module):
        """Temporarily swaps EMA weights into `module`."""
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in self.shadow:
                    self._backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in module.named_parameters():
                    if name in self._backup:
                        param.copy_(self._backup[name])
            self._backup.clear()


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def score_frame(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    builder: inputs.InputBuilder,
    batch_size: int,
    device: torch.device,
    rank: int = 0,
    world: int = 1,
    precision: str = "bf16",
    num_workers: int = 2,
) -> pd.DataFrame:
    """Scores every row of `frame`, sharded across ranks.

    Returns (on every rank) a frame aligned with `frame` holding
    logit_final, logit_abs and logit_softmin.
    """
    core = model.module if hasattr(model, "module") else model
    core.eval()
    dataset = data_lib.ScreeningDataset(frame, builder)
    # Sort by length for efficient padding, then shard round-robin.
    lengths = (
        frame["title"].str.len() + frame["abstract"].str.len()
    ).to_numpy()
    order = np.argsort(-lengths, kind="stable")[rank::world]
    loader = torch_data.DataLoader(
        torch_data.Subset(dataset, order.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_lib.Collator(builder.pad_id, core.use_token_types),
    )
    rows, outs = [], {
        k: [] for k in ("logit_final", "logit_abs", "logit_softmin")
    }
    for batch in loader:
        rows.append(batch["row_idx"].numpy())
        token_types = batch.get("token_type_ids")
        with _autocast(device, precision):
            out = core(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["crit_positions"].to(device),
                token_types.to(device) if token_types is not None else None,
            )
        for key in outs:
            outs[key].append(out[key].float().cpu().numpy())
    local = {"row": np.concatenate(rows) if rows else np.zeros(0, np.int64)}
    for key, parts in outs.items():
        local[key] = np.concatenate(parts) if parts else np.zeros(0)
    gathered = distributed.gather_objects(local, world)
    result = (
        pd.DataFrame(
            {k: np.concatenate([g[k] for g in gathered]) for k in local}
        )
        .set_index("row")
        .sort_index()
    )
    return result.reset_index(drop=True)


def _evaluate(
    model, frame, builder, spec, device, rank, world
) -> dict[str, float]:
    scores = score_frame(
        model,
        frame,
        builder,
        int(spec["optim"]["batch_size_per_gpu"]) * 2,
        device,
        rank,
        world,
        spec.get("precision", "bf16"),
        int(spec.get("num_workers", 2)),
    )
    scored = frame[["review_id", "label_final"]].assign(
        score=scores["logit_final"].to_numpy()
    )
    return ranking.summarize_ranking(scored), scores


def apply_overrides(
    spec: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    """Applies "optim.lr=1e-5"-style overrides (values parsed as YAML)."""
    for override in overrides:
        key, value = override.split("=", 1)
        node = spec
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return spec


def run(args: Any) -> dict[str, Any]:
    """Trains one (backbone, fold). `args` comes from the CLI parser."""
    rank, _, world, device = distributed.setup()
    main = distributed.is_main(rank)
    bench_cfg = config.Config.load(args.config)
    ens_cfg = config.Config.load(args.ensemble_config)
    paths = config.Paths.from_config(bench_cfg)
    spec = spec_lib.backbone_spec(ens_cfg, args.backbone)
    spec = apply_overrides(spec, args.override or [])
    seed = int(spec["seed"]) + args.fold
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = os.path.join(
        args.output_dir or paths.runs, args.backbone, f"fold{args.fold}"
    )
    if main:
        io.ensure_dir(out_dir)
        logging.info("spec: %s", json.dumps(spec, indent=1))

    reviews, records, splits = pipeline.load_split_data(paths)
    train_df, holdout_df = data_lib.training_frames(
        reviews,
        records,
        splits,
        spec,
        args.fold,
        ens_cfg.get("fold_salt"),
        float(ens_cfg.get("holdout_fraction", 0.05)),
    )
    if args.limit_reviews:
        keep = (
            train_df["review_id"].drop_duplicates().iloc[: args.limit_reviews]
        )
        train_df = train_df[train_df["review_id"].isin(keep)].reset_index(
            drop=True
        )
    holdout_eval = data_lib.subsample_negatives(
        holdout_df, int(spec.get("eval_negatives_per_review", 1000)), "holdout"
    )
    if main:
        logging.info(
            "train %s rows / %d reviews; holdout %s rows / %d reviews",
            f"{len(train_df):,}",
            train_df["review_id"].nunique(),
            f"{len(holdout_eval):,}",
            holdout_eval["review_id"].nunique(),
        )

    tokenizer = model_lib.load_tokenizer(spec)
    builder = inputs.InputBuilder(tokenizer, spec)
    model = model_lib.TitanScreener(spec, tokenizer).to(device)
    total, trainable = model_lib.count_parameters(model)
    if main:
        logging.info(
            "parameters: %s total, %s trainable", f"{total:,}", f"{trainable:,}"
        )
    dataset = data_lib.ScreeningDataset(train_df, builder)
    optim_cfg = spec["optim"]
    batch_size = int(optim_cfg["batch_size_per_gpu"])
    if spec["sampler"].get("type") == "class_balanced":
        batch_sampler = sampler_lib.ClassBalancedBatchSampler(
            dataset.label_final, batch_size, seed, rank, world
        )
    else:
        batch_sampler = sampler_lib.ReviewGroupedBatchSampler(
            dataset.review_idx,
            dataset.label_final,
            dataset.label_abs,
            train_df["retrieval_rank"].to_numpy(np.float64),
            batch_size,
            spec["sampler"],
            seed,
            rank,
            world,
        )
    loader = torch_data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(spec.get("num_workers", 4)),
        collate_fn=data_lib.Collator(builder.pad_id, model.use_token_types),
        persistent_workers=int(spec.get("num_workers", 4)) > 0,
    )

    optimizer = torch.optim.AdamW(
        model_lib.layerwise_param_groups(
            model,
            float(optim_cfg["lr"]),
            float(optim_cfg["head_lr"]),
            float(optim_cfg["weight_decay"]),
            float(optim_cfg["layer_decay"]),
        ),
        betas=(0.9, 0.98),
        eps=1e-6,
        fused=device.type == "cuda",
    )
    accum = int(optim_cfg.get("grad_accum", 1))
    epochs = int(optim_cfg["epochs"])
    steps_per_epoch = max(1, len(batch_sampler) // accum)
    total_steps = steps_per_epoch * epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(float(optim_cfg["warmup_ratio"]) * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = losses.ScreeningLoss(spec["loss"])
    ema = (
        Ema(model, float(optim_cfg["ema_decay"]))
        if float(optim_cfg.get("ema_decay", 0)) > 0
        else None
    )
    ddp = model
    if world > 1:
        ddp = torch_parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=bool(spec.get("lora")),
        )
    metric_name = spec.get("early_stopping_metric", "spec_at_99_recall")
    best, best_epoch, bad_epochs, step = -1.0, -1, 0, 0
    log_path = os.path.join(out_dir, "train_log.jsonl")
    started = time.time()
    for epoch in range(epochs):
        batch_sampler.set_epoch(epoch)
        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            batch = {
                k: v.to(device, non_blocking=True) for k, v in batch.items()
            }
            sync = (i + 1) % accum == 0
            ctx = (
                ddp.no_sync()
                if world > 1 and not sync
                else contextlib.nullcontext()
            )
            with ctx, _autocast(device, spec.get("precision", "bf16")):
                out = ddp(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["crit_positions"],
                    batch.get("token_type_ids"),
                )
                loss, parts = loss_fn(
                    out,
                    batch["label_final"],
                    batch["label_abs"],
                    batch["review_idx"],
                )
                (loss / accum).backward()
            if not sync:
                continue
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optim_cfg["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)
            step += 1
            if main and step % 50 == 0:
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_min": (time.time() - started) / 60,
                    **parts,
                }
                logging.info("%s", entry)
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
            if args.max_steps and step >= args.max_steps:
                break
        with ema.applied(model) if ema else contextlib.nullcontext():
            metrics, scores = _evaluate(
                model, holdout_eval, builder, spec, device, rank, world
            )
            value = metrics.get(metric_name, -1.0)
            improved = value > best
            if improved and main:
                model_lib.save_checkpoint(
                    model,
                    tokenizer,
                    os.path.join(out_dir, "best"),
                    {"fold": args.fold, "epoch": epoch, "metrics": metrics},
                )
                holdout_eval[["review_id", "record_id", "label_final"]].assign(
                    **{k: scores[k].to_numpy() for k in scores}
                ).to_parquet(os.path.join(out_dir, "holdout_preds.parquet"))
        if improved:
            best, best_epoch, bad_epochs = value, epoch, 0
        else:
            bad_epochs += 1
        if main:
            entry = {"epoch": epoch, "holdout": metrics, "best": best}
            logging.info("epoch %d holdout %s", epoch, metrics)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        patience = int(spec.get("patience", 0))
        if (patience and bad_epochs >= patience) or (
            args.max_steps and step >= args.max_steps
        ):
            break
    summary = {
        "backbone": args.backbone,
        "fold": args.fold,
        "best_metric": best,
        "metric": metric_name,
        "best_epoch": best_epoch,
        "steps": step,
        "minutes": (time.time() - started) / 60,
    }
    if main:
        io.write_json(summary, os.path.join(out_dir, "summary.json"))
    distributed.teardown()
    return summary
