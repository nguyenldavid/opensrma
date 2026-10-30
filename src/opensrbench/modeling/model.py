"""TITAN-Open screener: decomposed PICOS soft-min cross-encoder.

Heads on top of any Hugging Face backbone (encoder or decoder):

  global head      pooled [CLS] (encoders) or last token (decoders) ->
                   final-inclusion and abstract-stage logits.
  criterion head   one satisfaction logit z_k per criterion marker,
                   aggregated by a differentiable soft-min
                     softmin = -(1/t) * log(mean_k exp(-t * z_k)),
                   so a single violated criterion can veto a record.
  gates            learnable sigmoid gates blend global and soft-min
                   logits separately for the final and abstract heads.

`architecture: pooled` drops the criterion head (TITAN-SR baseline).
The soft-min is numerically safe for rows with no markers: masked entries
never enter the log-sum-exp and the global logit is used instead.
"""

import json
import logging
import os
from typing import Any

import torch
from torch import nn
import transformers

from opensrbench.modeling import inputs

TINY_MODEL = "__tiny_bert__"


def load_tokenizer(spec: dict[str, Any], path: str | None = None):
    """Loads (or builds, for the smoke model) the backbone tokenizer."""
    source = path or spec["model_name"]
    if source == TINY_MODEL:
        return _tiny_tokenizer()
    tokenizer = transformers.AutoTokenizer.from_pretrained(source)
    if spec.get("add_crit_token", True) and path is None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [inputs.CRIT_TOKEN]}
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _tiny_tokenizer():
    """A small word-piece tokenizer built in memory (no network)."""
    import tokenizers  # pylint: disable=import-outside-toplevel
    from tokenizers import models  # pylint: disable=import-outside-toplevel
    from tokenizers import normalizers
    from tokenizers import pre_tokenizers

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", inputs.CRIT_TOKEN]
    vocab += [chr(c) for c in range(ord("a"), ord("z") + 1)]
    vocab += [str(d) for d in range(10)]
    vocab += ["##" + chr(c) for c in range(ord("a"), ord("z") + 1)]
    vocab += ["##" + str(d) for d in range(10)]
    vocab = {t: i for i, t in enumerate(dict.fromkeys(vocab))}
    core = tokenizers.Tokenizer(
        models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=30)
    )
    core.normalizer = normalizers.BertNormalizer(lowercase=True)
    core.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        additional_special_tokens=[inputs.CRIT_TOKEN],
    )
    return tok


def _load_backbone(spec: dict[str, Any], vocab_size: int) -> nn.Module:
    name = spec["model_name"]
    if name == TINY_MODEL:
        cfg = transformers.BertConfig(
            vocab_size=vocab_size,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=512,
        )
        return transformers.BertModel(cfg, add_pooling_layer=False)
    kwargs = {}
    attn = spec.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn
    try:
        model = transformers.AutoModel.from_pretrained(name, **kwargs)
    except (ImportError, ValueError) as exc:
        logging.warning(
            "attn_implementation=%s unavailable (%s); using defaults",
            attn,
            exc,
        )
        model = transformers.AutoModel.from_pretrained(name)
    # transformers >= 5 loads checkpoints in their stored dtype (often bf16).
    # Keep fp32 master weights; bf16 compute comes from autocast.
    if spec.get("lora") and spec.get("base_dtype") == "bf16":
        # Frozen LoRA base in bf16 halves its memory; adapters and heads
        # stay fp32 (peft keeps adapter weights in fp32).
        return model.to(torch.bfloat16)
    return model.float()


class TitanScreener(nn.Module):
    """Backbone + global/criterion heads (see module docstring)."""

    def __init__(self, spec: dict[str, Any], tokenizer):
        super().__init__()
        self.spec = spec
        self.encoder = _load_backbone(spec, len(tokenizer))
        embeddings = self.encoder.get_input_embeddings()
        if embeddings.num_embeddings < len(tokenizer):
            old = embeddings.num_embeddings
            self.encoder.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                weight = self.encoder.get_input_embeddings().weight
                weight[old:] = weight[:old].mean(dim=0, keepdim=True)
        if spec.get("gradient_checkpointing"):
            # Non-reentrant checkpointing works with frozen (LoRA) bases.
            try:
                self.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.encoder.gradient_checkpointing_enable()
        if spec.get("lora"):
            self.encoder = _apply_lora(self.encoder, spec["lora"])
        hidden = self.encoder.config.hidden_size
        dropout = float(spec.get("head_dropout", 0.1))
        self.pooling = spec.get("pooling", "cls")
        self.decomposed = spec.get("architecture", "decomposed") == "decomposed"
        self.softmin_temp = float(spec.get("softmin_temp", 4.0))
        self.dropout = nn.Dropout(dropout)
        self.global_final_head = nn.Linear(hidden, 1)
        self.global_abs_head = nn.Linear(hidden, 1)
        if self.decomposed:
            self.criterion_head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
            self.gate_final = nn.Parameter(torch.tensor(0.0))
            self.gate_abs = nn.Parameter(torch.tensor(0.0))
        types = getattr(self.encoder.config, "type_vocab_size", 0) or 0
        self.use_token_types = types > 1

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters that receive the (higher) head learning rate."""
        return [
            p
            for n, p in self.named_parameters()
            if not n.startswith("encoder.") and p.requires_grad
        ]

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last":
            last = mask.sum(dim=1).clamp(min=1) - 1
            return hidden[
                torch.arange(hidden.size(0), device=hidden.device), last
            ]
        return hidden[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        crit_positions: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.use_token_types and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        if hidden.dtype != self.global_final_head.weight.dtype:
            # bf16 LoRA base: heads are fp32 (autocast re-casts on GPU).
            hidden = hidden.to(self.global_final_head.weight.dtype)
        pooled = self.dropout(self._pool(hidden, attention_mask))
        global_final = self.global_final_head(pooled).squeeze(-1).float()
        global_abs = self.global_abs_head(pooled).squeeze(-1).float()
        if not self.decomposed:
            return {
                "logit_final": global_final,
                "logit_abs": global_abs,
                "logit_softmin": torch.zeros_like(global_final),
            }
        valid = crit_positions >= 0  # [B, K]
        index = crit_positions.clamp(min=0)
        gathered = torch.gather(
            hidden, 1, index.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        )
        z = self.criterion_head(self.dropout(gathered)).squeeze(-1).float()
        t = self.softmin_temp
        has_crit = valid.any(dim=1)
        neg = (-t * z).masked_fill(~valid, float("-inf"))
        neg = torch.where(has_crit.unsqueeze(1), neg, torch.zeros_like(neg))
        count = valid.sum(dim=1).clamp(min=1).to(z.dtype)
        softmin = -(torch.logsumexp(neg, dim=1) - torch.log(count)) / t
        g_final = torch.sigmoid(self.gate_final)
        g_abs = torch.sigmoid(self.gate_abs)
        final = torch.where(
            has_crit,
            g_final * global_final + (1 - g_final) * softmin,
            global_final,
        )
        abstract = torch.where(
            has_crit, g_abs * global_abs + (1 - g_abs) * softmin, global_abs
        )
        return {
            "logit_final": final,
            "logit_abs": abstract,
            "logit_softmin": torch.where(
                has_crit, softmin, torch.zeros_like(softmin)
            ),
            "criterion_logits": z.masked_fill(~valid, 0.0),
        }


def _apply_lora(model: nn.Module, lora_cfg: dict[str, Any]) -> nn.Module:
    try:
        import peft  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "LoRA backbones need `pip install peft` (opensrbench[lora])."
        ) from exc
    config = peft.LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    return peft.get_peft_model(model, config)


def layerwise_param_groups(
    model: TitanScreener,
    lr: float,
    head_lr: float,
    weight_decay: float,
    layer_decay: float,
) -> list[dict[str, Any]]:
    """AdamW groups with layer-wise LR decay (top layer gets `lr`)."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    layer_ids = {}
    n_layers = 0
    for name, _ in named:
        for part in name.split("."):
            if part.isdigit():
                layer_ids[name] = int(part)
                n_layers = max(n_layers, int(part) + 1)
                break
    groups: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, param in named:
        if not name.startswith("encoder."):
            group_lr = head_lr
        elif name in layer_ids:
            group_lr = lr * layer_decay ** (n_layers - 1 - layer_ids[name])
        elif "embed" in name:
            group_lr = lr * layer_decay**n_layers
        else:
            group_lr = lr
        no_decay = (
            param.ndim < 2 or "norm" in name.lower() or name.endswith("bias")
        )
        key = (group_lr, 0.0 if no_decay else weight_decay)
        groups.setdefault(key, []).append(param)
    return [
        {"params": params, "lr": group_lr, "weight_decay": wd}
        for (group_lr, wd), params in groups.items()
    ]


def trainable_state_dict(model: TitanScreener) -> dict[str, torch.Tensor]:
    """State to save: everything, or only trainable tensors for LoRA."""
    state = model.state_dict()
    if model.spec.get("lora"):
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        state = {k: v for k, v in state.items() if k in trainable}
    return {k: v.detach().cpu().contiguous() for k, v in state.items()}


def save_checkpoint(
    model: TitanScreener, tokenizer, out_dir: str, metadata: dict[str, Any]
) -> None:
    """Saves weights (safetensors), tokenizer and the resolved spec."""
    from safetensors import \
        torch as st  # pylint: disable=import-outside-toplevel

    os.makedirs(out_dir, exist_ok=True)
    st.save_file(
        trainable_state_dict(model), os.path.join(out_dir, "model.safetensors")
    )
    tokenizer.save_pretrained(out_dir)
    with open(
        os.path.join(out_dir, "screener.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump({"spec": model.spec, **metadata}, fh, indent=2, default=str)


def load_checkpoint(
    ckpt_dir: str, device: str = "cpu"
) -> tuple[TitanScreener, Any, dict[str, Any]]:
    """Rebuilds a screener from `save_checkpoint` output."""
    from safetensors import \
        torch as st  # pylint: disable=import-outside-toplevel

    with open(os.path.join(ckpt_dir, "screener.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    spec = meta["spec"]
    tokenizer = load_tokenizer(spec, path=ckpt_dir)
    model = TitanScreener(spec, tokenizer)
    state = st.load_file(os.path.join(ckpt_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    if missing and not spec.get("lora"):
        raise RuntimeError(f"missing keys in checkpoint: {missing[:5]}")
    return model.to(device).eval(), tokenizer, meta


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
