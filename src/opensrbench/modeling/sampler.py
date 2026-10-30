"""Distributed batch samplers.

ReviewGroupedBatchSampler (TITAN-Open)
  Each batch is a concatenation of review *groups*: `pos_per_group`
  positives of one review plus negatives of the same review, half of them
  hard (records excluded at full text first, then the top retrieval ranks
  of the reconstructed pool). This makes the in-review ranking loss
  meaningful and shows the model the hard negatives that decide specificity
  at 99%+ sensitivity. An epoch is one pass over (capped) positives.

ClassBalancedBatchSampler (TITAN-SR baseline)
  Equivalent of `WeightedRandomSampler` with 1/class-frequency weights.

Both shard deterministically across DDP ranks (identical length on every
rank) and reshuffle per epoch via `set_epoch`.
"""

import math

import numpy as np


def _shard(batches: list[list[int]], rank: int, world: int) -> list[list[int]]:
    usable = (len(batches) // world) * world
    return batches[rank:usable:world]


class ReviewGroupedBatchSampler:
    """Yields lists of row indices grouped by review (see module doc)."""

    def __init__(
        self,
        review_idx: np.ndarray,
        label_final: np.ndarray,
        label_abs: np.ndarray,
        retrieval_rank: np.ndarray,
        batch_size: int,
        sampler_cfg: dict,
        seed: int,
        rank: int = 0,
        world: int = 1,
    ):
        self.batch_size = batch_size
        self.cfg = sampler_cfg
        self.seed = seed
        self.rank, self.world = rank, world
        self.epoch = 0
        order = np.argsort(review_idx, kind="stable")
        bounds = np.flatnonzero(np.diff(review_idx[order])) + 1
        self._reviews = []
        for rows in np.split(order, bounds):
            pos = rows[label_final[rows] == 1]
            neg = rows[label_final[rows] == 0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            ft_exc = neg[label_abs[neg] == 1]
            ranked = neg[np.isfinite(retrieval_rank[neg])]
            ranked = ranked[np.argsort(retrieval_rank[ranked], kind="stable")]
            hard = np.concatenate([ft_exc, ranked[~np.isin(ranked, ft_exc)]])
            self._reviews.append((pos, neg, hard))
        self._plan = self._make_plan()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._plan = self._make_plan()

    def _make_plan(self) -> list[list[int]]:
        cfg = self.cfg
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        pos_per_group = int(cfg.get("pos_per_group", 3))
        group_size = int(cfg.get("group_size", 16))
        neg_per_pos = float(cfg.get("negatives_per_positive", 6))
        hard_frac = float(cfg.get("hard_negative_fraction", 0.5))
        max_pos = int(cfg.get("max_pos_per_review", 64))
        groups: list[list[int]] = []
        for pos, neg, hard in self._reviews:
            pos = rng.permutation(pos)[:max_pos]
            n_neg = min(len(neg), int(math.ceil(len(pos) * neg_per_pos)))
            n_hard = min(len(hard), int(round(n_neg * hard_frac)))
            pool = hard[: max(3 * n_hard, n_hard)]
            chosen_hard = (
                rng.choice(pool, size=n_hard, replace=False)
                if (n_hard)
                else np.array([], dtype=np.int64)
            )
            rest = np.setdiff1d(neg, chosen_hard, assume_unique=True)
            chosen_rand = rng.choice(
                rest, size=min(len(rest), n_neg - n_hard), replace=False
            )
            negatives = rng.permutation(
                np.concatenate([chosen_hard, chosen_rand])
            )
            n_groups = int(math.ceil(len(pos) / pos_per_group))
            neg_chunks = np.array_split(negatives, n_groups)
            for g in range(n_groups):
                members = list(pos[g * pos_per_group : (g + 1) * pos_per_group])
                members += list(
                    neg_chunks[g][: max(0, group_size - len(members))]
                )
                groups.append([int(x) for x in members])
        rng.shuffle(groups)
        batches, current = [], []
        for group in groups:
            if current and len(current) + len(group) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(group)
        if current:
            batches.append(current)
        return _shard(batches, self.rank, self.world)

    def __iter__(self):
        return iter(self._plan)

    def __len__(self) -> int:
        return len(self._plan)


class ClassBalancedBatchSampler:
    """1/class-frequency sampling with replacement (TITAN-SR recipe)."""

    def __init__(
        self,
        label_final: np.ndarray,
        batch_size: int,
        seed: int,
        rank: int = 0,
        world: int = 1,
        num_samples: int | None = None,
    ):
        self.labels = label_final
        self.batch_size = batch_size
        self.seed = seed
        self.rank, self.world = rank, world
        self.num_samples = num_samples or len(label_final)
        self.epoch = 0
        counts = np.bincount(label_final, minlength=2).astype(np.float64)
        weights = 1.0 / np.maximum(counts[label_final], 1.0)
        self._p = weights / weights.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000003 * self.epoch)
        draws = rng.choice(
            len(self.labels), size=self.num_samples, replace=True, p=self._p
        )
        batches = [
            draws[i : i + self.batch_size].tolist()
            for i in range(0, len(draws), self.batch_size)
        ]
        return iter(_shard(batches, self.rank, self.world))

    def __len__(self) -> int:
        n_batches = int(math.ceil(self.num_samples / self.batch_size))
        return n_batches // self.world
