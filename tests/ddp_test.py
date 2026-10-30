"""Multi-process (DDP) training and global-batch tests on CPU/gloo."""

import contextlib
import json
import os
import socket

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

# pylint: disable=import-outside-toplevel,wrong-import-position
# pylint: disable=protected-access
from torch import distributed as dist
from torch import multiprocessing as mp
import transformers

from opensrbench import config
from opensrbench.modeling import inputs
from opensrbench.modeling import losses
from opensrbench.modeling import model as model_lib
from opensrbench.modeling import spec as spec_lib
from opensrbench.modeling import train

_ROOT = os.path.join(os.path.dirname(__file__), "..")
_WORLD = 2
_STEPS = 3


def _spec(backbone: str) -> dict:
    """A repo spec with its backbone swapped for the tiny smoke model."""
    cfg = config.Config.load(
        os.path.join(_ROOT, "configs", "titan_ensemble.yaml")
    )
    spec = spec_lib.backbone_spec(cfg, backbone)
    tiny = spec_lib.backbone_spec(cfg, "tiny")
    spec.update(
        model_name=model_lib.TINY_MODEL,
        max_seq_len=tiny["max_seq_len"],
        budgets=tiny["budgets"],
        gradient_checkpointing=False,
        attn_implementation=None,
    )
    # Global batch 4 x 4 = 16 -> 2 ranks x micro-batch 4 x accum 2, which
    # exercises `no_sync` accumulation as well as the synchronised step.
    spec["optim"].update(batch_size_per_gpu=4, grad_accum=4)
    return spec


def _batch(builder: inputs.InputBuilder, micro: int, seed: int) -> dict:
    criteria = json.dumps(["Include: adults", "Exclude: animals"])
    feats = []
    for i in range(micro):
        enc = builder.encode(
            f"title {seed} {i}",
            "abstract text " * (3 + (seed + i) % 4),
            "objective",
            criteria,
            "include adults exclude animals",
        )
        # One review with positives, FT-excluded and plain negatives.
        enc.update(label_final=i % 2, label_abs=int(i % 4 != 0), review_idx=0)
        feats.append(enc)
    # The tiny BERT has type_vocab_size 2, so token types are used.
    return {
        k: torch.as_tensor(v)
        for k, v in inputs.collate(feats, builder.pad_id, True).items()
    }


def _worker(rank: int, port: int, spec: dict) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD)
    try:
        torch.manual_seed(0)
        tok = model_lib.load_tokenizer(spec)
        builder = inputs.InputBuilder(tok, spec)
        model = model_lib.TitanScreener(spec, tok)
        micro, accum, _ = train.resolve_batching(spec["optim"], _WORLD)
        assert (micro, accum) == (4, 2)
        ddp = train.wrap_ddp(model, spec, torch.device("cpu"), _WORLD)
        assert not ddp.find_unused_parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = losses.ScreeningLoss(spec["loss"])
        ddp.train()
        for i in range(_STEPS * accum):
            batch = _batch(builder, micro, seed=100 * rank + i)
            sync = (i + 1) % accum == 0
            ctx = contextlib.nullcontext() if sync else ddp.no_sync()
            with ctx:
                out = ddp(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["crit_positions"],
                    batch.get("token_type_ids"),
                )
                loss, _ = loss_fn(
                    out,
                    batch["label_final"],
                    batch["label_abs"],
                    batch["review_idx"],
                )
                (loss / accum).backward()
            if not sync:
                continue
            missing = [
                n
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is None
            ]
            assert not missing, f"parameters without gradient: {missing}"
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # Synchronised gradients keep the replicas identical.
        digest = float(
            sum(p.detach().double().sum() for p in model.parameters())
        )
        digests = [None] * _WORLD
        dist.all_gather_object(digests, digest)
        assert digests[0] == pytest.approx(digests[1], rel=1e-9), digests
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch.distributed gloo backend unavailable",
)
@pytest.mark.parametrize(
    "backbone",
    [
        "biolinkbert_large",  # Decomposed soft-min + hybrid loss.
        "titan_sr_pubmedbert",  # Pooled + weighted BCE (TITAN-SR baseline).
    ],
)
def test_ddp_two_ranks_trains_and_every_parameter_gets_a_gradient(backbone):
    spec = _spec(backbone)
    mp.spawn(_worker, args=(_free_port(), spec), nprocs=_WORLD, join=True)


def test_weighted_bce_builds_no_abstract_head_and_logit_abs_is_final():
    spec = _spec("titan_sr_pubmedbert")
    tok = model_lib.load_tokenizer(spec)
    model = model_lib.TitanScreener(spec, tok)
    names = [n for n, _ in model.named_parameters()]
    assert not any("global_abs_head" in n or n == "gate_abs" for n in names)
    batch = _batch(inputs.InputBuilder(tok, spec), 2, seed=0)
    out = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["crit_positions"],
        batch["token_type_ids"],
    )
    assert torch.equal(out["logit_abs"], out["logit_final"])
    hybrid = model_lib.TitanScreener(_spec("biolinkbert_large"), tok)
    assert hybrid.has_abs_head and hasattr(hybrid, "gate_abs")


def test_drop_pooler_removes_bert_pooler_and_forward_still_works():
    cfg = transformers.BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    bert = transformers.BertModel(cfg)
    assert bert.pooler is not None
    model_lib._drop_pooler(bert)
    assert not any("pooler" in n for n, _ in bert.named_parameters())
    out = bert(input_ids=torch.tensor([[2, 7, 3]]))
    assert out.last_hidden_state.shape == (1, 3, 16)


def test_legacy_checkpoint_with_pooler_and_abs_head_still_loads(tmp_path):
    from safetensors import torch as st

    spec = _spec("titan_sr_pubmedbert")
    tok = model_lib.load_tokenizer(spec)
    model = model_lib.TitanScreener(spec, tok)
    model_lib.save_checkpoint(model, tok, str(tmp_path), {"fold": 0})
    path = os.path.join(tmp_path, "model.safetensors")
    state = st.load_file(path)
    state["encoder.pooler.dense.weight"] = torch.zeros(64, 64)
    state["encoder.pooler.dense.bias"] = torch.zeros(64)
    state["global_abs_head.weight"] = torch.zeros(1, 64)
    state["global_abs_head.bias"] = torch.zeros(1)
    st.save_file(state, path)
    loaded, _, _ = model_lib.load_checkpoint(str(tmp_path))
    weight = loaded.global_final_head.weight
    assert torch.equal(weight, model.global_final_head.weight)
    # Any other unexpected key is still an error.
    state["criterion_head.0.weight"] = torch.zeros(1)
    st.save_file(state, path)
    with pytest.raises(RuntimeError, match="unexpected keys"):
        model_lib.load_checkpoint(str(tmp_path))


@pytest.mark.parametrize(
    "per_gpu,accum,world,expected",
    [
        (64, 2, 1, (64, 2, 128)),  # Single GPU: unchanged.
        (64, 2, 2, (64, 1, 128)),
        (64, 2, 4, (32, 1, 128)),  # TITAN-SR recipe on 4 x L40S.
        (64, 2, 8, (16, 1, 128)),  # ... and on 8 x H100.
        (24, 4, 4, (24, 1, 96)),
        (8, 8, 8, (8, 1, 64)),
        (12, 2, 1, (12, 2, 24)),  # Below group_size on 1 GPU is allowed.
    ],
)
def test_resolve_batching_keeps_global_batch(per_gpu, accum, world, expected):
    cfg = {"batch_size_per_gpu": per_gpu, "grad_accum": accum}
    assert train.resolve_batching(cfg, world, 16) == expected
    micro, rank_accum, global_batch = expected
    assert micro * rank_accum * world == global_batch == per_gpu * accum
    assert micro <= per_gpu


def test_resolve_batching_rejects_unsplittable_global_batch():
    with pytest.raises(ValueError, match="not divisible"):
        train.resolve_batching({"batch_size_per_gpu": 6}, 4)
    # 24 rows over 4 GPUs = 6-row micro-batches < one 16-record review group.
    with pytest.raises(ValueError, match="review groups"):
        train.resolve_batching({"batch_size_per_gpu": 24}, 4, 16)
    assert train.resolve_batching({"batch_size_per_gpu": 24}, 4) == (6, 1, 24)


@pytest.mark.parametrize(
    "per_gpu,accum,world",
    [
        # 18-row micro-batches hold one 16-row group: 2 x 2 = 4 groups per
        # step on 2 GPUs vs 3 x 1 = 3 on one.
        (24, 3, 2),
        # 24-row micro-batches hold one group: 2 groups vs 48 // 16 = 3.
        (48, 1, 2),
        # The 1-GPU-only recipes: 24 x 1 (ModernBERT-large) on 2 GPUs.
        (24, 1, 2),
    ],
)
def test_resolve_batching_rejects_changed_group_count(per_gpu, accum, world):
    cfg = {"batch_size_per_gpu": per_gpu, "grad_accum": accum}
    with pytest.raises(ValueError, match="review groups"):
        train.resolve_batching(cfg, world, 16)
    # Samplers that draw single rows only need the row count preserved.
    micro, rank_accum, global_batch = train.resolve_batching(cfg, world)
    assert micro * rank_accum * world == global_batch == per_gpu * accum


@pytest.mark.parametrize(
    "per_gpu,accum,world",
    [(32, 1, 2), (16, 2, 2), (64, 2, 4), (24, 4, 4)],
)
def test_resolve_batching_keeps_group_count(per_gpu, accum, world):
    # biolinkbert/biomedbert 32 x 1 and deberta 16 x 2 pass on 2 GPUs.
    micro, rank_accum, _ = train.resolve_batching(
        {"batch_size_per_gpu": per_gpu, "grad_accum": accum}, world, 16
    )
    groups = max(1, micro // 16) * rank_accum * world
    assert groups == max(1, per_gpu // 16) * accum
