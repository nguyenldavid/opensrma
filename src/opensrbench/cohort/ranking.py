"""Lexical (BM25), dense (MedCPT) and fused ranking of candidate citations."""

import collections
import math
import re
from typing import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an the of in on for with to and or by at from as is are was were be "
    "been this that these those it its we our their there which who whom "
    "than then into over under not no".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens without stop words."""
    return [
        t
        for t in _TOKEN.findall((text or "").lower())
        if t not in _STOP and len(t) > 1
    ]


class Bm25:
    """Okapi BM25 over a small in-memory corpus (one review's candidates)."""

    def __init__(
        self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75
    ):
        self._k1, self._b = k1, b
        self._docs = [collections.Counter(tokenize(d)) for d in documents]
        self._lengths = np.array(
            [sum(d.values()) for d in self._docs], dtype=np.float64
        )
        self._avg = float(self._lengths.mean()) if len(self._docs) else 0.0
        df = collections.Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(self._docs)
        self._idf = {
            t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for `query`."""
        out = np.zeros(len(self._docs), dtype=np.float64)
        terms = set(tokenize(query))
        norm = self._k1 * (
            1 - self._b + self._b * self._lengths / max(self._avg, 1e-9)
        )
        for i, doc in enumerate(self._docs):
            score = 0.0
            for term in terms:
                tf = doc.get(term)
                if tf:
                    score += (
                        self._idf[term] * tf * (self._k1 + 1) / (tf + norm[i])
                    )
            out[i] = score
        return out


def reciprocal_rank_fusion(
    score_lists: Sequence[np.ndarray], k: int = 60
) -> np.ndarray:
    """Fuses several score vectors with RRF (Cormack et al., 2009)."""
    fused = np.zeros(len(score_lists[0]), dtype=np.float64)
    for scores in score_lists:
        ranks = np.empty(len(scores), dtype=np.int64)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(len(scores))
        fused += 1.0 / (k + ranks + 1)
    return fused


class MedCptRanker:
    """MedCPT bi-encoder (Jin et al., 2023) query/article similarity.

    Loads lazily so CPU-only data stages never import torch unless dense
    ranking is enabled.
    """

    def __init__(
        self,
        query_encoder: str,
        article_encoder: str,
        device: str | None = None,
        batch_size: int = 64,
    ):
        import torch  # pylint: disable=import-outside-toplevel
        import transformers  # pylint: disable=import-outside-toplevel

        self._torch = torch
        self._device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._batch = batch_size
        self._q_tok = transformers.AutoTokenizer.from_pretrained(query_encoder)
        self._q_enc = (
            transformers.AutoModel.from_pretrained(query_encoder)
            .to(self._device)
            .eval()
        )
        self._a_tok = transformers.AutoTokenizer.from_pretrained(
            article_encoder
        )
        self._a_enc = (
            transformers.AutoModel.from_pretrained(article_encoder)
            .to(self._device)
            .eval()
        )
        self._cache: dict[str, np.ndarray] = {}
        self._max_cache = 400_000

    def _encode(self, tokenizer, encoder, inputs, max_length: int):
        torch = self._torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self._batch):
                batch = inputs[start : start + self._batch]
                enc = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                with torch.autocast(
                    self._device.split(":")[0],
                    dtype=torch.bfloat16,
                    enabled=self._device.startswith("cuda"),
                ):
                    hidden = encoder(**enc).last_hidden_state[:, 0, :]
                out.append(hidden.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode(self._q_tok, self._q_enc, list(queries), 64)

    def scores(
        self,
        query: str,
        keys: Sequence[str],
        titles: Sequence[str],
        abstracts: Sequence[str],
    ) -> np.ndarray:
        """Dot-product similarity, caching article embeddings by key."""
        if len(self._cache) > self._max_cache:
            self._cache.clear()  # Bounded memory (~3 KB per article).
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            pairs = [[titles[i] or "", abstracts[i] or ""] for i in missing]
            emb = self._encode(self._a_tok, self._a_enc, pairs, 512)
            for i, vec in zip(missing, emb):
                self._cache[keys[i]] = vec
        q = self.encode_queries([query])[0]
        docs = (
            np.stack([self._cache[k] for k in keys])
            if keys
            else np.zeros((0, q.shape[0]))
        )
        return docs @ q
