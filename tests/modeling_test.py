"""Tests for the screener inputs, model heads, losses and samplers."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

# pylint: disable=import-outside-toplevel,wrong-import-position
from opensrbench.modeling import inputs
from opensrbench.modeling import losses
from opensrbench.modeling import model as model_lib
from opensrbench.modeling import sampler as sampler_lib

_SPEC = {
    "model_name": model_lib.TINY_MODEL,
    "architecture": "decomposed",
    "input_format": "decomposed",
    "max_seq_len": 96,
    "pooling": "cls",
    "add_crit_token": True,
    "softmin_temp": 4.0,
    "head_dropout": 0.0,
    "budgets": {
        "title": 12,
        "objective": 12,
        "criteria_total": 30,
        "criterion_max": 10,
        "max_criteria": 3,
    },
}


def _builder():
    tok = model_lib.load_tokenizer(_SPEC)
    return tok, inputs.InputBuilder(tok, _SPEC)


def test_crit_positions_point_at_marker_tokens_and_fit_length():
    tok, builder = _builder()
    crit = json.dumps(["Include: adults", "Exclude: animals", "Include: rct"])
    enc = builder.encode("a title", "an abstract " * 40, "objective text", crit)
    assert len(enc["input_ids"]) <= _SPEC["max_seq_len"]
    marker = tok.convert_tokens_to_ids(inputs.CRIT_TOKEN)
    assert enc["crit_positions"]
    assert all(enc["input_ids"][p] == marker for p in enc["crit_positions"])


def test_titan_sr_layout_has_no_markers():
    spec = dict(_SPEC, input_format="titan_sr")
    tok = model_lib.load_tokenizer(spec)
    enc = inputs.InputBuilder(tok, spec).encode(
        "t", "a", "o", "[]", "criteria text"
    )
    assert enc["crit_positions"] == []


def test_softmin_is_finite_with_and_without_criteria_and_backprops():
    tok, builder = _builder()
    feats = [
        builder.encode("t", "abs", "obj", json.dumps(["Include: x y"])),
        builder.encode("t", "abs", "", "[]"),
    ]
    batch = {
        k: torch.as_tensor(v)
        for k, v in inputs.collate(feats, builder.pad_id, False).items()
    }
    model = model_lib.TitanScreener(_SPEC, tok)
    out = model(
        batch["input_ids"], batch["attention_mask"], batch["crit_positions"]
    )
    assert torch.isfinite(out["logit_final"]).all()
    loss_fn = losses.ScreeningLoss(
        {
            "type": "hybrid",
            "gamma_pos": 0.5,
            "gamma_neg": 2.0,
            "abs_weight": 0.5,
            "rank_type": "listwise",
            "rank_weight": 0.5,
        }
    )
    loss, _ = loss_fn(
        out, torch.tensor([1, 0]), torch.tensor([1, 0]), torch.tensor([0, 0])
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_softmin_approaches_min_for_large_temperature():
    z = torch.tensor([[3.0, -2.0, 5.0]])
    t = 50.0
    softmin = -(torch.logsumexp(-t * z, 1) - np.log(3)) / t
    assert abs(float(softmin) - (-2.0)) < 0.05


def test_ranking_loss_zero_when_perfectly_separated_pairwise():
    logits = torch.tensor([5.0, -5.0, 4.0, -4.0])
    labels = torch.tensor([1, 0, 1, 0])
    reviews = torch.tensor([0, 0, 1, 1])
    valid = torch.ones(4, dtype=torch.bool)
    assert (
        float(
            losses.in_review_ranking(
                logits, labels, reviews, valid, "pairwise", 1.0
            )
        )
        == 0.0
    )
    # Cross-review pairs are ignored: no positive in review 2.
    assert (
        float(
            losses.in_review_ranking(
                logits, torch.tensor([0, 0, 0, 0]), reviews, valid, "listwise"
            )
        )
        == 0
    )


def test_ft_excluded_records_ignored_by_final_head():
    out = {"logit_final": torch.tensor([3.0]), "logit_abs": torch.tensor([3.0])}
    cfg = {
        "type": "hybrid",
        "gamma_pos": 0.0,
        "gamma_neg": 0.0,
        "abs_weight": 0.0,
        "rank_type": "none",
    }
    loss, parts = losses.ScreeningLoss(cfg)(
        out, torch.tensor([0]), torch.tensor([1]), torch.tensor([0])
    )
    assert parts["asl_final"] == 0.0 and float(loss) == 0.0


def test_review_grouped_sampler_groups_and_shards():
    review = np.repeat(np.arange(6), 30)
    label = np.tile(np.r_[np.ones(4), np.zeros(26)], 6).astype(int)
    rank = np.where(label == 0, np.tile(np.arange(30), 6), np.nan)
    cfg = {
        "group_size": 8,
        "pos_per_group": 2,
        "negatives_per_positive": 3,
        "hard_negative_fraction": 0.5,
        "max_pos_per_review": 64,
    }
    s0 = sampler_lib.ReviewGroupedBatchSampler(
        review, label, label, rank, 16, cfg, 0, rank=0, world=2
    )
    s1 = sampler_lib.ReviewGroupedBatchSampler(
        review, label, label, rank, 16, cfg, 0, rank=1, world=2
    )
    assert len(s0) == len(s1) > 0
    seen = [i for batch in s0 for i in batch] + [i for b in s1 for i in b]
    assert set(np.flatnonzero(label == 1)) >= set(
        i for i in seen if label[i] == 1
    )
    for batch in s0:
        assert len(batch) <= 16
    s0.set_epoch(1)
    assert len(s0) > 0


def test_layerwise_groups_cover_all_trainable_parameters():
    tok, _ = _builder()
    model = model_lib.TitanScreener(_SPEC, tok)
    groups = model_lib.layerwise_param_groups(model, 1e-4, 1e-3, 0.01, 0.9)
    n = sum(p.numel() for g in groups for p in g["params"])
    assert n == sum(p.numel() for p in model.parameters() if p.requires_grad)
