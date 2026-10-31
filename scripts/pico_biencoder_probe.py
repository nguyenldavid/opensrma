"""CPU probe: can a bi-encoder rule records out per criterion? (idea B1).

docs/EXPERIMENT_IDEAS.md, B1. MedCPT (query encoder for criteria, article
encoder for documents and abstract sentences) scores every record of a
review in four ways:

  obj_doc     objective vs. whole title + abstract
  min_doc     min over inclusion criteria of criterion vs. whole document
  min_sent    min over inclusion criteria of the best-matching sentence
  mean_sent   mean over inclusion criteria of the best-matching sentence

Scores are z-normalised within each review's candidate pool. A single
global threshold per variant is fitted on a sample of TRAIN reviews to keep
99.5% of train positives (pooled) and then applied to VAL, reporting
equal-review sensitivity and the share of negatives ruled out. Per-review
AUC is reported for context. Only train and val are read.

  python scripts/pico_biencoder_probe.py --data-root <pilot_v2 root>
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import transformers

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_EXCLUDE = ("Exclude:",)


def _sentences(abstract: str, limit: int = 30) -> list[str]:
    parts = [s.strip() for s in _SENT.split(abstract or "")]
    return [s for s in parts if len(s) >= 20][:limit] or [abstract or ""]


class _Encoder:
    """CLS (MedCPT) or mean-pooled, normalised (sentence-transformers)."""

    def __init__(self, name: str, max_len: int, pooling: str = "cls"):
        self.tok = transformers.AutoTokenizer.from_pretrained(name)
        self.model = transformers.AutoModel.from_pretrained(name).eval()
        self.max_len = max_len
        self.pooling = pooling

    @torch.inference_mode()
    def __call__(self, texts, pairs=None, batch: int = 64) -> np.ndarray:
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for start in range(0, len(texts), batch):
            idx = order[start : start + batch]
            first = [texts[i] for i in idx]
            enc = (
                self.tok(
                    first,
                    [pairs[i] for i in idx],
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                if pairs is not None
                else self.tok(
                    first,
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
            )
            hidden = self.model(**enc).last_hidden_state
            if self.pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vec = torch.nn.functional.normalize(vec, dim=-1)
            else:
                vec = hidden[:, 0]
            out[idx] = vec.numpy()
        return out


class _NeumlDocs:
    """Adapts a symmetric encoder to the (title, abstract) pair interface."""

    def __init__(self, encoder: "_Encoder"):
        self.enc = encoder

    def __call__(self, titles, abstracts=None, batch: int = 64):
        texts = [f"{t}. {a}".strip(". ") for t, a in zip(titles, abstracts)]
        return self.enc(texts, batch=batch)


def _per_review_auc(df: pd.DataFrame, col: str) -> float:
    aucs = []
    for _, g in df.groupby("review_id"):
        y = g["label_final"].to_numpy()
        if 0 < y.sum() < len(y):
            s = g[col].to_numpy()
            pos, neg = s[y == 1], s[y == 0]
            aucs.append(
                (
                    (pos[:, None] > neg[None]).mean()
                    + 0.5 * (pos[:, None] == neg[None]).mean()
                )
            )
    return float(np.mean(aucs))


def score_reviews(reviews, records, q_enc, a_enc, cache):
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    records = records.copy()
    docs = a_enc(
        records["title"].fillna("").tolist(),
        records["abstract"].fillna("").tolist(),
    )
    sent_lists = [_sentences(a) for a in records["abstract"].fillna("")]
    flat = [s for ss in sent_lists for s in ss]
    owner = np.repeat(
        np.arange(len(sent_lists)), [len(ss) for ss in sent_lists]
    )
    sents = a_enc([""] * len(flat), flat)
    crit = {}
    for rid, row in reviews.iterrows():
        items = [
            c
            for c in json.loads(row["criteria_list"] or "[]")
            if not c.startswith(_EXCLUDE)
        ]
        crit[rid] = (
            q_enc([row["objective"] or ""])[0],
            q_enc(items) if items else None,
        )
    cols = {
        k: np.zeros(len(records))
        for k in ("obj_doc", "min_doc", "min_sent", "mean_sent")
    }
    rid_arr = records["review_id"].to_numpy()
    for i in range(len(records)):
        obj, items = crit[rid_arr[i]]
        cols["obj_doc"][i] = docs[i] @ obj
        if items is None:
            for k in ("min_doc", "min_sent", "mean_sent"):
                cols[k][i] = cols["obj_doc"][i]
            continue
        sv = sents[owner == i] @ items.T  # [sentences, criteria]
        best = sv.max(axis=0)
        cols["min_doc"][i] = (items @ docs[i]).min()
        cols["min_sent"][i] = best.min()
        cols["mean_sent"][i] = best.mean()
    for k, v in cols.items():
        records[k] = v
        records[k + "_z"] = records.groupby("review_id")[k].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-6)
        )
    out = records[
        ["review_id", "label_final"]
        + [c for c in records if c.endswith(("_doc", "_sent", "_z"))]
    ]
    out.to_parquet(cache)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-reviews", type=int, default=250)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--encoder", choices=("medcpt", "neuml"), default="medcpt"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    splits = pd.read_parquet(f"{args.data_root}/splits/splits.parquet")
    reviews = pd.read_parquet(
        f"{args.data_root}/processed/reviews.parquet",
        columns=["review_id", "objective", "criteria_list"],
    ).set_index("review_id")
    val_ids = set(splits.loc[splits.split == "val", "review_id"])
    train_ids = (
        splits.loc[splits.split == "train", "review_id"]
        .drop_duplicates()
        .sample(args.train_reviews, random_state=0)
    )
    keep = val_ids | set(train_ids)
    records = pd.read_parquet(
        f"{args.data_root}/processed/records.parquet",
        columns=["review_id", "title", "abstract", "label_final"],
        filters=[("review_id", "in", list(keep))],
    )
    if args.encoder == "medcpt":
        q_enc = _Encoder("ncbi/MedCPT-Query-Encoder", 64)
        a_enc = _Encoder("ncbi/MedCPT-Article-Encoder", 512)
    else:
        # One symmetric encoder; documents as "title. abstract" text.
        q_enc = _Encoder("NeuML/pubmedbert-base-embeddings", 512, "mean")
        a_enc = _NeumlDocs(q_enc)
    results = {}
    frames = {}
    for name, ids in (("train", set(train_ids)), ("val", val_ids)):
        recs = records[records.review_id.isin(ids)].reset_index(drop=True)
        frames[name] = score_reviews(
            reviews.loc[sorted(ids & set(recs.review_id))],
            recs,
            q_enc,
            a_enc,
            f"{args.out}/{name}_scores.parquet",
        )
        print(name, len(recs), "records scored", flush=True)
    for col in ("obj_doc_z", "min_doc_z", "min_sent_z", "mean_sent_z"):
        tr, va = frames["train"], frames["val"]
        pos = np.sort(tr.loc[tr.label_final == 1, col].to_numpy())
        tau = pos[int(np.floor(0.005 * len(pos)))]
        kept = va[col] >= tau
        va = va.assign(kept=kept)
        sens = va[va.label_final == 1].groupby("review_id")["kept"].mean()
        neg_out = 1 - va.loc[va.label_final == 0, "kept"].mean()
        results[col] = {
            "val_auc_per_review": round(_per_review_auc(va, col), 4),
            "tau_train_pooled_99.5": round(float(tau), 4),
            "val_equal_review_sensitivity": round(float(sens.mean()), 4),
            "val_reviews_below_99": int((sens < 0.99).sum()),
            "val_negatives_ruled_out": round(float(neg_out), 4),
        }
        print(col, results[col], flush=True)
    with open(f"{args.out}/results.json", "w") as fh:
        json.dump(results, fh, indent=1)


if __name__ == "__main__":
    main()
