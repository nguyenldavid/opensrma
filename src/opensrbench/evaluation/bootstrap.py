"""Family cluster bootstrap, paired comparisons and Holm correction (3.3).

Whole connected review families are resampled with replacement. The
implementation is vectorised: per-family sums of the review metrics are
combined with a (B x F) multinomial count matrix, so 5,000 replicates over
hundreds of families take milliseconds and every system sees the *same*
replicates (required for paired differences).
"""

import numpy as np
import pandas as pd

METRICS = ("youdens_j", "sensitivity", "specificity", "workload")


class FamilyBootstrap:
    """Shared bootstrap replicates over a fixed set of families."""

    def __init__(
        self, families: np.ndarray, replicates: int = 5000, seed: int = 20261025
    ):
        self.families = np.array(sorted(set(families)))
        rng = np.random.default_rng(seed)
        f = len(self.families)
        self.counts = rng.multinomial(
            f, np.full(f, 1.0 / f), size=replicates
        ).astype(np.float64)

    def _family_sums(
        self, review_metrics: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        df = review_metrics.copy()
        out = {}
        for col in ("sensitivity", "specificity", "workload"):
            present = df[col].notna()
            sums = df[present].groupby("family_id")[col].sum()
            counts = df[present].groupby("family_id")[col].size()
            out[f"{col}_sum"] = sums.reindex(self.families).fillna(0).to_numpy()
            out[f"{col}_n"] = counts.reindex(self.families).fillna(0).to_numpy()
        return out

    def distributions(
        self, review_metrics: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """Bootstrap distribution of every equal-review metric."""
        sums = self._family_sums(review_metrics)
        out = {}
        for col in ("sensitivity", "specificity", "workload"):
            num = self.counts @ sums[f"{col}_sum"]
            den = self.counts @ sums[f"{col}_n"]
            out[col] = np.divide(
                num, den, out=np.full_like(num, np.nan), where=den > 0
            )
        out["youdens_j"] = out["sensitivity"] + out["specificity"] - 1.0
        return out


def percentile_ci(
    point: float, dist: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float]:
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0:
        return (point, float("nan"), float("nan"))
    lo, hi = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(point), float(lo), float(hi))


def paired_difference(
    dist_a: np.ndarray, dist_b: np.ndarray, point: float, alpha: float = 0.05
) -> dict[str, float]:
    """Paired bootstrap difference A - B with a two-sided p-value."""
    diff = dist_a - dist_b
    diff = diff[np.isfinite(diff)]
    p = 2.0 * min((diff <= 0).mean(), (diff >= 0).mean()) if len(diff) else 1.0
    point, lo, hi = percentile_ci(point, diff, alpha)
    return {
        "delta": point,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": float(min(1.0, p)),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
