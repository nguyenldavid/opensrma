"""Validation-locked operating thresholds (protocol Sections 3.3, 5.4).

All thresholds use an inclusive comparator: a record is kept iff
score >= tau. Ties at the threshold are therefore kept (conservative).

pooled_conformal
  Split-conformal bound on a positive's miss probability. With validation
  positive scores sorted ascending s_(1) <= ... <= s_(n) and
  k = floor((n + 1) * alpha), tau = s_(k) guarantees
  P(score_new < tau) <= k / (n + 1) <= alpha for an exchangeable new
  positive (tau = -inf when k = 0). The v1.0 snippet's np.quantile on
  (n+1)/n-adjusted levels interpolates between order statistics and can
  break this guarantee; this module uses the exact order statistic.

equal_review
  Largest tau whose equal-review mean validation sensitivity >= target —
  the same estimand as the primary endpoint.

equal_review_lcb
  Largest tau whose family-bootstrap 5th percentile of equal-review
  sensitivity >= target (most conservative).

jmax
  tau maximising equal-review Youden's J on validation.
"""

import numpy as np
import pandas as pd

_NEG_INF = float("-inf")


def pooled_conformal(pos_scores: np.ndarray, target: float) -> float:
    """Split-conformal threshold for sensitivity >= target."""
    scores = np.sort(np.asarray(pos_scores, dtype=np.float64))
    n = len(scores)
    if n == 0:
        return _NEG_INF
    k = int(np.floor((n + 1) * (1.0 - target)))
    return _NEG_INF if k < 1 else float(scores[min(k, n) - 1])


class _ReviewIndex:
    """Pre-sorted per-review positive/negative scores for fast sweeps."""

    def __init__(self, scores, labels, review_ids, family_ids=None):
        df = pd.DataFrame({"s": scores, "y": labels, "r": review_ids})
        if family_ids is not None:
            df["f"] = family_ids
        self.pos = [
            np.sort(g["s"].to_numpy()) for _, g in df[df["y"] == 1].groupby("r")
        ]
        self.neg = [
            np.sort(g["s"].to_numpy()) for _, g in df[df["y"] == 0].groupby("r")
        ]
        self.pos_reviews = sorted(df.loc[df["y"] == 1, "r"].unique())
        self.neg_reviews = sorted(df.loc[df["y"] == 0, "r"].unique())
        if family_ids is not None:
            fam = df.drop_duplicates("r").set_index("r")["f"]
            self.pos_family = fam.loc[self.pos_reviews].to_numpy()
        self.candidates = np.unique(df.loc[df["y"] == 1, "s"].to_numpy())

    def sens(self, tau: float) -> np.ndarray:
        """Per-review sensitivity at tau (kept iff s >= tau)."""
        return np.array(
            [
                1.0 - np.searchsorted(p, tau, side="left") / len(p)
                for p in self.pos
            ]
        )

    def spec(self, tau: float) -> np.ndarray:
        return np.array(
            [np.searchsorted(n, tau, side="left") / len(n) for n in self.neg]
        )


def _largest_feasible(candidates: np.ndarray, feasible) -> float:
    """Binary search for the largest candidate with feasible(tau) True.

    Feasibility must be monotone (True for small tau, False for large).
    """
    lo, hi = 0, len(candidates) - 1
    if hi < 0 or not feasible(candidates[0]):
        return _NEG_INF
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(candidates[mid]):
            lo = mid
        else:
            hi = mid - 1
    return float(candidates[lo])


def equal_review(scores, labels, review_ids, target: float) -> float:
    index = _ReviewIndex(scores, labels, review_ids)
    return _largest_feasible(
        index.candidates, lambda t: index.sens(t).mean() >= target - 1e-12
    )


def equal_review_lcb(
    scores,
    labels,
    review_ids,
    family_ids,
    target: float,
    replicates: int = 1000,
    quantile: float = 0.05,
    seed: int = 0,
) -> float:
    index = _ReviewIndex(scores, labels, review_ids, family_ids)
    families, inverse = np.unique(index.pos_family, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(families),
        np.full(len(families), 1 / len(families)),
        size=replicates,
    )  # [B, F]
    weights = counts[:, inverse]  # [B, R+]

    def feasible(tau):
        sens = index.sens(tau)
        boot = (weights * sens).sum(axis=1) / np.maximum(weights.sum(axis=1), 1)
        return np.quantile(boot, quantile) >= target - 1e-12

    return _largest_feasible(index.candidates, feasible)


def jmax(scores, labels, review_ids, grid: int = 2000) -> float:
    index = _ReviewIndex(scores, labels, review_ids)
    taus = np.unique(
        np.quantile(
            np.asarray(scores, dtype=np.float64), np.linspace(0, 1, grid)
        )
    )
    best_tau, best_j = _NEG_INF, -np.inf
    for tau in taus:
        j = index.sens(tau).mean() + index.spec(tau).mean() - 1.0
        if j > best_j:
            best_tau, best_j = float(tau), j
    return best_tau


def op_name(target: float) -> str:
    """0.99 -> "tau99", 0.995 -> "tau99.5"."""
    pct = target * 100
    return f"tau{pct:g}"


def lock_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    review_ids: np.ndarray,
    family_ids: np.ndarray,
    targets: list[float],
    method: str,
) -> dict[str, float]:
    """Locks every target threshold plus tauJ with the chosen method."""
    out = {}
    for target in targets:
        if method == "pooled_conformal":
            tau = pooled_conformal(scores[labels == 1], target)
        elif method == "equal_review":
            tau = equal_review(scores, labels, review_ids, target)
        elif method == "equal_review_lcb":
            tau = equal_review_lcb(
                scores, labels, review_ids, family_ids, target
            )
        else:
            raise ValueError(f"unknown threshold method {method!r}")
        out[op_name(target)] = tau
    out["tauJ"] = jmax(scores, labels, review_ids)
    return out
