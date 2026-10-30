"""Thin torch.distributed helpers (torchrun / Slurm friendly)."""

import datetime
import os

import torch
from torch import distributed as dist


def setup() -> tuple[int, int, int, torch.device]:
    """Initialises NCCL/Gloo from torchrun env vars.

    Returns:
      (rank, local_rank, world_size, device).
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=datetime.timedelta(minutes=60),
        )
    return rank, local_rank, world, device


def teardown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def gather_objects(obj, world: int) -> list:
    """all_gather_object that degrades to [obj] without a process group."""
    if world == 1 or not dist.is_initialized():
        return [obj]
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out
