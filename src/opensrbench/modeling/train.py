"""DDP training of one TITAN-Open backbone on one family fold.

Launch (single node, 8 GPUs):
  torchrun --nproc_per_node=8 -m opensrbench.cli train \
      --backbone modernbert_large --fold 0

Outputs (under runs/<ensemble-agnostic>/<backbone>/fold<k>/):
  best/            Best checkpoint by the early-stopping metric on the
                   held-out fold (safetensors + tokenizer + screener.json).
  holdout_preds.parquet  Out-of-fold scores of the best checkpoint.
  train_log.jsonl  Step/epoch metrics.  summary.json  Final summary.

The global (effective) batch is `optim.batch_size_per_gpu x optim.grad_accum`
as written in the YAML, whatever the number of GPUs, so a recipe means the
same thing on 1 GPU, 4 x L40S or 8 x H100 (see `resolve_batching`).
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


def resolve_batching(
    optim_cfg: dict[str, Any], world: int, min_micro_batch: int = 1
) -> tuple[int, int, int]:
    """Splits the configured global batch over `world` ranks.

    The global batch is `batch_size_per_gpu * grad_accum` (the YAML values,
    i.e. the single-GPU recipe). Each rank processes `global / world`
    examples per optimizer step as `accum` micro-batches of `micro` rows,
    with the fewest micro-batches whose size stays <= `batch_size_per_gpu`
    (a memory ceiling). When `grad_accum` is divisible by `world` this keeps
    `micro == batch_size_per_gpu` and `accum == grad_accum / world`; with
    one rank nothing changes.

    Args:
      optim_cfg: The spec's `optim` block.
      world: Number of DDP ranks.
      min_micro_batch: Review-group size of the review-grouped sampler (1
        for samplers that draw single rows). That sampler packs whole groups
        of up to `min_micro_batch` rows into a micro-batch, so the number of
        groups per optimizer step, not the row count, must be unchanged.

    Returns:
      (micro_batch_per_rank, grad_accum_per_rank, global_batch).

    Raises:
      ValueError: If the global batch cannot be split evenly over `world`
        ranks, or (grouped sampler) only with micro-batches below
        `min_micro_batch` or with a different number of review groups per
        optimizer step than on one GPU.
    """
    per_gpu = int(optim_cfg["batch_size_per_gpu"])
    global_batch = per_gpu * int(optim_cfg.get("grad_accum", 1))
    if global_batch % world:
        raise ValueError(
            f"global batch {global_batch} (batch_size_per_gpu {per_gpu} x "
            f"grad_accum {optim_cfg.get('grad_accum', 1)}) is not divisible "
            f"by {world} GPUs; change optim.grad_accum or the GPU count."
        )
    per_rank = global_batch // world
    accum = math.ceil(per_rank / per_gpu)
    while per_rank % accum:
        accum += 1
    micro = per_rank // accum

    # A micro-batch of `rows` rows holds max(1, rows // group) whole review
    # groups, so compare groups per optimizer step with the 1-GPU recipe.
    def groups(rows: int) -> int:
        return max(1, rows // min_micro_batch)

    recipe_groups = groups(per_gpu) * int(optim_cfg.get("grad_accum", 1))
    if micro < min(per_gpu, min_micro_batch) or (
        groups(micro) * accum * world != recipe_groups
    ):
        raise ValueError(
            f"global batch {global_batch} over {world} GPUs leaves "
            f"{accum} micro-batch(es) of {micro} rows per GPU, i.e. "
            f"{groups(micro) * accum * world} review groups of "
            f"{min_micro_batch} per optimizer step instead of the recipe's "
            f"{recipe_groups} (whole review groups); change optim.grad_accum "
            "(this changes the recipe) or use fewer GPUs."
        )
    return micro, accum, global_batch


def wrap_ddp(
    model: torch.nn.Module,
    spec: dict[str, Any],
    device: torch.device,
    world: int,
) -> torch.nn.Module:
    """Wraps `model` in DDP when `world > 1` (identity otherwise).

    `TitanScreener` builds no module without a gradient path (no backbone
    pooler; no abstract head under `weighted_bce`), so DDP runs with
    `find_unused_parameters=False` (no extra graph traversal per step).
    """
    if world == 1:
        return model
    return torch_parallel.DistributedDataParallel(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        # PEFT-wrapped decoders are not yet verified under DDP; keep the
        # safe (slower) unused-parameter search for LoRA specs only.
        find_unused_parameters=bool(spec.get("lora")),
    )


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
    # Resolved before any data/model loading so a bad split fails fast.
    optim_cfg = spec["optim"]
    class_balanced = spec["sampler"].get("type") == "class_balanced"
    batch_size, accum, global_batch = resolve_batching(
        optim_cfg,
        world,
        1 if class_balanced else int(spec["sampler"].get("group_size", 16)),
    )
    if main:
        logging.info(
            "global batch %d = %d GPU(s) x micro-batch %d x grad_accum %d",
            global_batch,
            world,
            batch_size,
            accum,
        )

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
    if class_balanced:
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
    ddp = wrap_ddp(model, spec, device, world)
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
