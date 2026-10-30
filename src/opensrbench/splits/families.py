"""Connected-family graph over reviews (protocol Section 2.3.1-2).

Two reviews are linked when their included studies overlap
(|Pu ∩ Pv| / min(|Pu|, |Pv|) >= 0.10) or their objectives are near-duplicates
(cosine >= 0.88). Families are the connected components, and they — not
reviews — are the unit of splitting and of the cluster bootstrap.

Included studies are matched by their normalized-title fingerprint (falling
back to PMID), so overlaps are detected even between sources that have no
PMIDs (e.g. Chan 2024 vs CLEF TAR).
"""

import collections
import itertools
import logging
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from opensrbench.utils import hashing


class UnionFind:
    """Disjoint-set forest with path compression and union by size."""

    def __init__(self, items: Sequence[str]):
        self._parent = {x: x for x in items}
        self._size = {x: 1 for x in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]

    def components(self) -> dict[str, list[str]]:
        out = collections.defaultdict(list)
        for x in self._parent:
            out[self.find(x)].append(x)
        return out


def positive_study_keys(records: pd.DataFrame) -> dict[str, set[str]]:
    """review_id -> set of study keys (title fingerprint or PMID)."""
    pos = records[records["label_final"] == 1]
    keys = np.where(
        pos["title_fp"].fillna("") != "",
        "t:" + pos["title_fp"].fillna(""),
        "p:" + pos["pmid"].fillna("").astype(str),
    )
    out: dict[str, set[str]] = collections.defaultdict(set)
    for review_id, key in zip(pos["review_id"], keys):
        if key not in ("p:", "t:"):
            out[review_id].add(key)
    return out


def overlap_edges(
    study_keys: dict[str, set[str]], threshold: float
) -> list[tuple[str, str]]:
    """Edges whose positive-set overlap coefficient reaches `threshold`."""
    index = collections.defaultdict(list)
    for review_id, keys in study_keys.items():
        for key in keys:
            index[key].append(review_id)
    shared = collections.Counter()
    for reviews in index.values():
        if len(reviews) > 1:
            for a, b in itertools.combinations(sorted(reviews), 2):
                shared[(a, b)] += 1
    edges = []
    for (a, b), n in shared.items():
        denom = min(len(study_keys[a]), len(study_keys[b]))
        if denom and n / denom >= threshold:
            edges.append((a, b))
    return edges


def embedding_edges(
    review_ids: Sequence[str],
    embeddings: np.ndarray,
    threshold: float,
    block: int = 2048,
) -> list[tuple[str, str]]:
    """Edges between reviews whose embedding cosine reaches `threshold`."""
    if len(review_ids) < 2:
        return []
    emb = embeddings.astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    edges = []
    for start in range(0, len(emb), block):
        sims = emb[start : start + block] @ emb.T
        rows, cols = np.nonzero(sims >= threshold)
        for r, c in zip(rows, cols):
            i = start + r
            if i < c:
                edges.append((review_ids[i], review_ids[c]))
    return edges


def tfidf_embed(texts: Sequence[str]) -> np.ndarray:
    """CPU fallback embedder (TF-IDF, dense; fine up to ~50k reviews)."""
    from sklearn.feature_extraction import \
        text as sk_text  # pylint: disable=import-outside-toplevel

    vectorizer = sk_text.TfidfVectorizer(
        min_df=1, max_features=50_000, sublinear_tf=True, stop_words="english"
    )
    matrix = vectorizer.fit_transform(texts)
    from sklearn.decomposition import \
        TruncatedSVD  # pylint: disable=import-outside-toplevel

    k = max(2, min(256, matrix.shape[1] - 1, matrix.shape[0] - 1))
    return TruncatedSVD(n_components=k, random_state=0).fit_transform(matrix)


def build_families(
    reviews: pd.DataFrame,
    records: pd.DataFrame,
    overlap_threshold: float,
    cosine_threshold: float,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None,
) -> tuple[pd.DataFrame, dict]:
    """Clusters reviews into connected families.

    Args:
      reviews: Processed reviews.
      records: Processed records (needs title_fp, pmid, label_final).
      overlap_threshold: Positive-overlap coefficient threshold.
      cosine_threshold: Objective-embedding cosine threshold.
      embed_fn: texts -> embeddings; None disables the cosine criterion.

    Returns:
      (frame with review_id, family_id, family_size; statistics dict).
    """
    ids = reviews["review_id"].tolist()
    forest = UnionFind(ids)
    edges = overlap_edges(positive_study_keys(records), overlap_threshold)
    for a, b in edges:
        forest.union(a, b)
    n_cos = 0
    if embed_fn is not None:
        texts = (
            reviews["title"].fillna("") + ". " + reviews["objective"].fillna("")
        ).tolist()
        cos_edges = embedding_edges(ids, embed_fn(texts), cosine_threshold)
        n_cos = len(cos_edges)
        for a, b in cos_edges:
            forest.union(a, b)
    rows = []
    for members in forest.components().values():
        family_id = "fam_" + hashing.short_id("|".join(sorted(members)), 12)
        rows.extend(
            {
                "review_id": m,
                "family_id": family_id,
                "family_size": len(members),
            }
            for m in members
        )
    frame = pd.DataFrame(rows)
    stats = {
        "reviews": len(ids),
        "overlap_edges": len(edges),
        "cosine_edges": n_cos,
        "families": int(frame["family_id"].nunique()),
        "largest_family": int(frame["family_size"].max()),
    }
    logging.info("family graph: %s", stats)
    return frame, stats
