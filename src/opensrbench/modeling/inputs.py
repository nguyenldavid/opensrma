"""Tokenisation, segment budgeting and batching for the screeners.

Decomposed layout (TITAN-Open):

    [CLS] title [SEP] abstract [SEP] objective [SEP]
          crit_1 [CRIT] crit_2 [CRIT] ... crit_K [CRIT] [SEP]

The document comes first and each criterion is *followed* by its marker so
the same layout works for bidirectional encoders and for causal decoders
(a marker after the criterion has seen the document and the criterion).
Marker positions are returned explicitly (`crit_positions`), so the model
never depends on a particular token id.

TITAN-SR layout (baseline): tokenizer pair `criteria` / "title. abstract"
with longest-first truncation to 512 tokens, exactly as published.
"""

import json
from typing import Any, Sequence

import numpy as np

CRIT_TOKEN = "[CRIT]"


class InputBuilder:
    """Builds model inputs for (review, record) pairs."""

    def __init__(self, tokenizer, spec: dict[str, Any]):
        """Initializes the builder.

        Args:
          tokenizer: A Hugging Face fast tokenizer (with [CRIT] added when
            `spec["add_crit_token"]`).
          spec: Resolved backbone spec (see modeling.spec).
        """
        self.tok = tokenizer
        self.spec = spec
        self.max_len = int(spec["max_seq_len"])
        self.budgets = spec["budgets"]
        self.layout = spec["input_format"]
        cls_id = tokenizer.cls_token_id
        if cls_id is None:
            cls_id = tokenizer.bos_token_id
        self.prefix = [cls_id] if cls_id is not None else []
        sep = tokenizer.sep_token_id
        if sep is None:
            sep = tokenizer.eos_token_id
        self.sep = sep
        if spec.get("add_crit_token", True):
            self.crit_id = tokenizer.convert_tokens_to_ids(CRIT_TOKEN)
        else:
            self.crit_id = sep
        self.suffix_eos = (
            spec.get("pooling") == "last" and tokenizer.eos_token_id is not None
        )
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )
        self._review_cache: dict[tuple[str, str], tuple] = {}

    def _ids(self, text: str, limit: int) -> list[int]:
        if not text or limit <= 0:
            return []
        return self.tok(
            text, add_special_tokens=False, truncation=True, max_length=limit
        )["input_ids"]

    def review_part(
        self, objective: str, criteria_json: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Tokenises the review side once per review (memoised).

        Returns:
          (token ids, relative marker offsets).
        """
        key = (objective, criteria_json)
        if key not in self._review_cache:
            if len(self._review_cache) > 50_000:
                self._review_cache.clear()
            self._review_cache[key] = self._review_part(
                objective, criteria_json
            )
        return self._review_cache[key]

    def _review_part(
        self, objective: str, criteria_json: str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        b = self.budgets
        ids = list(self._ids(objective, int(b["objective"])))
        if ids:
            ids.append(self.sep)
        offsets = []
        criteria = json.loads(criteria_json) if criteria_json else []
        criteria = criteria[: int(b["max_criteria"])]
        total = int(b["criteria_total"])
        if criteria:
            per = max(8, min(int(b["criterion_max"]), total // len(criteria)))
            for criterion in criteria:
                crit_ids = self._ids(criterion, per)
                if not crit_ids:
                    continue
                ids.extend(crit_ids)
                offsets.append(len(ids))
                ids.append(self.crit_id)
        return tuple(ids), tuple(offsets)

    def encode(
        self,
        title: str,
        abstract: str,
        objective: str,
        criteria_json: str,
        criteria_text: str = "",
    ) -> dict[str, Any]:
        """Encodes one pair into ids, mask, type ids and marker positions."""
        if self.layout == "titan_sr":
            enc = self.tok(
                criteria_text or "",
                f"{title or ''}. {abstract or ''}",
                truncation="longest_first",
                max_length=self.max_len,
            )
            return {
                "input_ids": enc["input_ids"],
                "token_type_ids": enc.get("token_type_ids"),
                "crit_positions": [],
            }
        review_ids, offsets = self.review_part(
            objective or "", criteria_json or "[]"
        )
        reserve = len(self.prefix) + 2 + int(self.suffix_eos) + 1
        doc_budget = self.max_len - reserve - len(review_ids)
        if doc_budget < 64:  # Long criteria: trim review side, keep document.
            keep = max(0, self.max_len - reserve - 64)
            review_ids = review_ids[:keep]
            offsets = tuple(o for o in offsets if o < keep)
            doc_budget = self.max_len - reserve - len(review_ids)
        title_ids = self._ids(
            title, min(int(self.budgets["title"]), doc_budget)
        )
        abstract_ids = self._ids(abstract, doc_budget - len(title_ids))
        ids = self.prefix + title_ids + [self.sep] + abstract_ids + [self.sep]
        start = len(ids)
        ids.extend(review_ids)
        ids.append(self.sep)
        if self.suffix_eos:
            ids.append(self.tok.eos_token_id)
        ids = ids[: self.max_len]
        positions = [start + o for o in offsets if start + o < len(ids)]
        type_ids = [0] * start + [1] * (len(ids) - start)
        return {
            "input_ids": ids,
            "token_type_ids": type_ids,
            "crit_positions": positions,
        }


def collate(
    features: Sequence[dict[str, Any]], pad_id: int, use_token_types: bool
) -> dict[str, Any]:
    """Right-pads a list of encoded examples into numpy arrays."""
    n = len(features)
    length = max(len(f["input_ids"]) for f in features)
    k = max([len(f["crit_positions"]) for f in features] + [1])
    input_ids = np.full((n, length), pad_id, dtype=np.int64)
    mask = np.zeros((n, length), dtype=np.int64)
    types = np.zeros((n, length), dtype=np.int64)
    crit = np.full((n, k), -1, dtype=np.int64)
    for i, f in enumerate(features):
        m = len(f["input_ids"])
        input_ids[i, :m] = f["input_ids"]
        mask[i, :m] = 1
        if use_token_types and f.get("token_type_ids") is not None:
            types[i, :m] = f["token_type_ids"]
        if f["crit_positions"]:
            crit[i, : len(f["crit_positions"])] = f["crit_positions"]
    batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
        "crit_positions": crit,
    }
    if use_token_types:
        batch["token_type_ids"] = types
    for key in ("label_final", "label_abs", "review_idx", "row_idx"):
        if key in features[0]:
            batch[key] = np.array([f[key] for f in features])
    return batch
