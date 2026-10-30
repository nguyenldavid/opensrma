"""Screening losses.

hybrid (TITAN-Open)
  * Asymmetric focal loss on the final head: easy negatives are strongly
    down-weighted (gamma_neg) while positives keep nearly full gradient
    (gamma_pos), which is what matters at >= 99% sensitivity.
  * The same loss on the abstract-stage head with label_abs.
  * Records excluded only at full text (label_abs=1, label_final=0) are
    ignored by the final head (`ft_excluded_policy: ignore`), so the model is
    never punished for keeping an abstract that humans also kept.
  * An in-review ranking term (listwise softmax or pairwise margin) that
    pushes every positive above the hard negatives *of the same review* —
    the quantity a within-review threshold actually depends on.

weighted_bce (TITAN-SR baseline)
  BCE with pos_weight (10:1 false-negative weighting).
"""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def asymmetric_focal(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    gamma_pos: float,
    gamma_neg: float,
) -> torch.Tensor:
    """Weighted mean asymmetric focal BCE."""
    probs = torch.sigmoid(logits)
    pos = labels * (1 - probs).pow(gamma_pos) * F.softplus(-logits)
    neg = (1 - labels) * probs.pow(gamma_neg) * F.softplus(logits)
    return ((pos + neg) * weights).sum() / weights.sum().clamp(min=1.0)


def in_review_ranking(
    logits: torch.Tensor,
    labels: torch.Tensor,
    review_idx: torch.Tensor,
    valid: torch.Tensor,
    rank_type: str,
    margin: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Ranking loss over (positive, negative) pairs from the same review.

    Args:
      logits: [B] scores.
      labels: [B] 0/1 targets.
      review_idx: [B] integer review ids.
      valid: [B] bool, records allowed in the ranking term.
      rank_type: "listwise" or "pairwise".
      margin: Pairwise hinge margin.
      temperature: Listwise softmax temperature.

    Returns:
      Scalar loss (0 with a graph when there are no valid pairs).
    """
    same = review_idx.unsqueeze(1) == review_idx.unsqueeze(0)
    pos = (labels == 1) & valid
    neg = (labels == 0) & valid
    pair = same & pos.unsqueeze(1) & neg.unsqueeze(0)  # [P_i, N_j]
    if not pair.any():
        return logits.sum() * 0.0
    if rank_type == "pairwise":
        diff = logits.unsqueeze(1) - logits.unsqueeze(0)
        return F.relu(margin - diff)[pair].mean()
    # Listwise: for each positive i, -log softmax_i over {i} ∪ negatives(i).
    scaled = logits / temperature
    neg_scores = scaled.unsqueeze(0).expand(len(logits), -1)
    neg_scores = neg_scores.masked_fill(~pair, float("-inf"))
    rows = pair.any(dim=1)
    lse = torch.logsumexp(
        torch.cat([scaled.unsqueeze(1), neg_scores], dim=1), dim=1
    )
    return (lse - scaled)[rows].mean()


class ScreeningLoss(nn.Module):
    """Combines the terms configured in `spec["loss"]`."""

    def __init__(self, loss_cfg: dict[str, Any]):
        super().__init__()
        self.cfg = loss_cfg
        self.kind = loss_cfg.get("type", "hybrid")

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        label_final: torch.Tensor,
        label_abs: torch.Tensor,
        review_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        final = label_final.float()
        logits = outputs["logit_final"]
        if self.kind == "weighted_bce":
            pos_weight = torch.tensor(
                float(cfg.get("pos_weight", 10.0)), device=logits.device
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, final, pos_weight=pos_weight
            )
            return loss, {"bce": float(loss.detach())}
        ft_excluded = (label_abs == 1) & (label_final == 0)
        if cfg.get("ft_excluded_policy", "ignore") == "ignore":
            w_final = (~ft_excluded).float()
        else:
            w_final = torch.ones_like(final)
        gp, gn = float(cfg["gamma_pos"]), float(cfg["gamma_neg"])
        l_final = asymmetric_focal(logits, final, w_final, gp, gn)
        l_abs = asymmetric_focal(
            outputs["logit_abs"],
            label_abs.float(),
            torch.ones_like(final),
            gp,
            gn,
        )
        total = l_final + float(cfg.get("abs_weight", 0.5)) * l_abs
        parts = {
            "asl_final": float(l_final.detach()),
            "asl_abs": float(l_abs.detach()),
        }
        rank_type = cfg.get("rank_type", "listwise")
        if rank_type != "none" and float(cfg.get("rank_weight", 0)) > 0:
            l_rank = in_review_ranking(
                logits,
                label_final,
                review_idx,
                w_final > 0,
                rank_type,
                float(cfg.get("margin", 1.0)),
                float(cfg.get("listwise_temp", 1.0)),
            )
            total = total + float(cfg["rank_weight"]) * l_rank
            parts["rank"] = float(l_rank.detach())
        return total, parts
