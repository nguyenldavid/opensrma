"""Dataset construction for training and inference."""

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data

from opensrbench.modeling import inputs
from opensrbench.utils import hashing

_RECORD_COLS = [
    "review_id",
    "record_id",
    "title",
    "abstract",
    "label_final",
    "label_abs",
    "retrieval_rank",
    # Kept so the training-time leakage guard can audit the exact frames.
    "pmid",
    "title_fp",
    "abstract_fp",
]


def fold_of_family(family_id: str, n_folds: int, salt: str) -> int:
    return hashing.stable_int(f"{salt}:{family_id}", n_folds)


def training_frames(
    reviews: pd.DataFrame,
    records: pd.DataFrame,
    splits: pd.DataFrame,
    spec: dict[str, Any],
    fold: int,
    fold_salt: str,
    holdout_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train, holdout) record frames for one backbone fold.

    Only the `train` split is used. Families are assigned to folds by hash;
    with n_folds == 1 a `holdout_fraction` of families is held out for
    early stopping instead. Review text columns are joined in.
    """
    train_ids = splits.loc[
        splits["split"] == "train", ["review_id", "family_id"]
    ]
    n_folds = int(spec["n_folds"])
    if n_folds > 1:
        train_ids["fold"] = train_ids["family_id"].map(
            lambda f: fold_of_family(f, n_folds, fold_salt)
        )
        is_holdout = train_ids["fold"] == fold
    else:
        is_holdout = train_ids["family_id"].map(
            lambda f: hashing.stable_unit(f"{fold_salt}:{f}") < holdout_fraction
        )
    joined = attach_review_text(
        records[records["review_id"].isin(train_ids["review_id"])], reviews
    )
    holdout_reviews = set(train_ids.loc[is_holdout, "review_id"])
    in_holdout = joined["review_id"].isin(holdout_reviews)
    return joined[~in_holdout].reset_index(drop=True), joined[
        in_holdout
    ].reset_index(drop=True)


def attach_review_text(
    records: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Joins objective / criteria text onto records."""
    cols = ["review_id", "objective", "selection_criteria", "criteria_list"]
    out = records[[c for c in _RECORD_COLS if c in records.columns]].merge(
        reviews[cols], on="review_id", how="left"
    )
    for col in ("title", "abstract", "objective", "selection_criteria"):
        out[col] = out[col].fillna("").astype(str)
    out["criteria_list"] = out["criteria_list"].fillna("[]")
    out["retrieval_rank"] = pd.to_numeric(
        out.get("retrieval_rank"), errors="coerce"
    )
    return out


def subsample_negatives(
    df: pd.DataFrame, per_review: int, salt: str
) -> pd.DataFrame:
    """Keeps all positives and a hash-ordered negative subset per review."""
    if per_review <= 0:
        return df
    key = [
        hashing.sha256_hex(f"{salt}:{r}:{i}")
        for r, i in zip(df["review_id"], df["record_id"])
    ]
    ranked = df.assign(_k=key).sort_values("_k")
    neg_rank = ranked.groupby(["review_id", "label_final"]).cumcount()
    keep = (ranked["label_final"] == 1) | (neg_rank < per_review)
    return ranked[keep].drop(columns="_k").reset_index(drop=True)


class ScreeningDataset(torch_data.Dataset):
    """Map-style dataset that encodes rows lazily."""

    def __init__(self, frame: pd.DataFrame, builder: inputs.InputBuilder):
        self.frame = frame.reset_index(drop=True)
        self.builder = builder
        codes, _ = pd.factorize(self.frame["review_id"])
        self.review_idx = codes.astype(np.int64)
        self._cols = {
            c: self.frame[c].tolist()
            for c in (
                "title",
                "abstract",
                "objective",
                "criteria_list",
                "selection_criteria",
            )
        }
        self.label_final = self.frame["label_final"].to_numpy(np.int64)
        self.label_abs = self.frame["label_abs"].to_numpy(np.int64)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> dict[str, Any]:
        c = self._cols
        enc = self.builder.encode(
            c["title"][i],
            c["abstract"][i],
            c["objective"][i],
            c["criteria_list"][i],
            c["selection_criteria"][i],
        )
        enc.update(
            label_final=int(self.label_final[i]),
            label_abs=int(self.label_abs[i]),
            review_idx=int(self.review_idx[i]),
            row_idx=i,
        )
        return enc


class Collator:
    """Picklable collate function returning torch tensors."""

    def __init__(self, pad_id: int, use_token_types: bool):
        self.pad_id = pad_id
        self.use_token_types = use_token_types

    def __call__(self, features):
        batch = inputs.collate(features, self.pad_id, self.use_token_types)
        return {k: torch.as_tensor(v) for k, v in batch.items()}
