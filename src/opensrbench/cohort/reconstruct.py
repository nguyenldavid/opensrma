"""Date-bounded candidate-pool reconstruction (protocol Section 2.2).

Network half (`fetch_candidates`):
  Tier 1  Executes the review's verbatim PubMed strategy bounded by
          [1900/01/01 : t_r][dp]. Accepted if the hit count lies within
          [tier1_min_count, tier1_max_count]; ESearch's 10k cap is handled by
          recursive date slicing.
  Tier 2  Otherwise, a strict PICO Boolean query (top K_r by relevance) and
          a broad one (up to 4*K_r) are executed under the same ceiling.

Offline half (`assemble_pool`):
  Tier 2 broad candidates are re-ranked by BM25 (+ MedCPT, fused with
  reciprocal-rank fusion) against the review's own objective and criteria;
  the pool is strict ∪ top-K_r(broad). Citations published after t_r, the
  review itself and already-labelled records are removed; the remaining
  candidates become negatives. Positives are always kept (C_r = P_r ∪ N_r).
"""

import datetime
import logging

import numpy as np
import pandas as pd

from opensrbench import schema
from opensrbench.cohort import ranking
from opensrbench.pubmed import eutils
from opensrbench.utils import dates
from opensrbench.utils import hashing


def search_ceiling(
    search_date: str | None,
    pub_date: str | None,
    positive_dates: list[str],
    lag_days: int = 90,
) -> datetime.date | None:
    """Returns t_r: reported search date, else pub date - lag, else a bound.

    Args:
      search_date: Reported literature-search cut-off (ISO), if any.
      pub_date: Review publication date (ISO), if any.
      positive_dates: Publication dates of the included studies.
      lag_days: Lag subtracted from the publication date.

    Returns:
      The ceiling date or None when nothing is known.
    """
    parsed = dates.parse_date(search_date) if search_date else None
    if parsed:
        return parsed
    parsed = dates.parse_date(pub_date) if pub_date else None
    if parsed:
        return parsed - datetime.timedelta(days=lag_days)
    known = [dates.parse_date(d) for d in positive_dates if d]
    known = [d for d in known if d]
    if known:
        return max(known) + datetime.timedelta(days=180)
    return None


def _range_term(term: str, lo: datetime.date, hi: datetime.date) -> str:
    return (
        f'({term}) AND ("{dates.to_pubmed(lo)}"[dp] : '
        f'"{dates.to_pubmed(hi)}"[dp])'
    )


def esearch_sliced(
    client: eutils.EUtilsClient,
    term: str,
    ceiling: datetime.date,
    max_total: int,
    cap: int = eutils.ESEARCH_HARD_CAP,
) -> tuple[int, list[str] | None]:
    """Retrieves all hits of a date-bounded query, slicing dates past 10k.

    Returns:
      (count, pmids) or (count, None) if count exceeds `max_total`.
    """
    lo = datetime.date(1900, 1, 1)
    count = client.count(_range_term(term, lo, ceiling))
    if count > max_total:
        return count, None
    out: list[str] = []
    stack = [(lo, ceiling)]
    while stack:
        a, b = stack.pop()
        n = client.count(_range_term(term, a, b))
        if n == 0:
            continue
        if n <= cap or (b - a).days < 2:
            out.extend(client.esearch(_range_term(term, a, b), cap)[1])
            continue
        mid = a + (b - a) / 2
        stack.extend([(a, mid), (mid + datetime.timedelta(days=1), b)])
    return count, list(dict.fromkeys(out))


def target_k(
    prisma_pubmed: object,
    prisma_total: object,
    k_min: int,
    k_max: int,
    k_default: int,
) -> int:
    """K_r: PRISMA database yield bounded to [k_min, k_max]."""
    for value in (prisma_pubmed, prisma_total):
        try:
            if value is not None and float(value) > 0:
                return int(min(max(float(value), k_min), k_max))
        except (TypeError, ValueError):
            continue
    return k_default


def fetch_candidates(
    review: dict,
    ceiling: datetime.date | None,
    client: eutils.EUtilsClient,
    query_generator,
    cohort_cfg: dict,
    k: int,
) -> dict:
    """Runs the network part of pool reconstruction for one review.

    Args:
      review: Review row as a dict (title, objective, selection_criteria,
        extra["search_strategy"]).
      ceiling: t_r date ceiling.
      client: E-utilities client.
      query_generator: Object with `.queries(title, objective, criteria)`.
      cohort_cfg: `cohort` section of the benchmark config.
      k: Target number of hybrid neighbours K_r.

    Returns:
      JSON-serialisable candidate description.
    """
    out = {
        "review_id": review["review_id"],
        "ceiling": dates.to_iso(ceiling),
        "k": k,
        "tier": "tier2_hybrid",
        "tier1_ids": [],
        "strict_ids": [],
        "broad_ids": [],
        "queries": {},
        "counts": {},
    }
    strategy = (review.get("extra") or {}).get("search_strategy")
    if strategy and ceiling:
        count, ids = esearch_sliced(
            client, strategy, ceiling, int(cohort_cfg["tier1_max_count"])
        )
        out["counts"]["tier1"] = count
        if ids is not None and count >= int(cohort_cfg["tier1_min_count"]):
            out.update(tier="tier1_exact", tier1_ids=ids)
            out["queries"]["tier1"] = strategy
            return out
    queries = query_generator.queries(
        review.get("title", ""),
        review.get("objective", ""),
        review.get("selection_criteria", ""),
    )
    out["queries"] = queries
    for name, size in (("strict", k), ("broad", min(4 * k, 9999))):
        if not queries.get(name):
            continue
        term = client.date_bounded(queries[name], ceiling)
        count, ids = client.esearch(term, size, sort="relevance")
        out["counts"][name] = count
        out[f"{name}_ids"] = ids
    return out


def assemble_pool(
    review: dict,
    labelled: pd.DataFrame,
    candidates: dict,
    texts: pd.DataFrame,
    pool_max: int,
    rrf_k: int = 60,
    dense: ranking.MedCptRanker | None = None,
    retrieval_label: str | None = None,
) -> pd.DataFrame:
    """Builds negative records for one review from fetched candidates.

    Args:
      review: Review row dict.
      labelled: The review's already-labelled records (positives etc.).
      candidates: Output of `fetch_candidates`.
      texts: PubMed frame (pmid, title, abstract, pub_date) for candidates.
      pool_max: Maximum pool size (labelled + new negatives).
      rrf_k: Reciprocal-rank-fusion constant.
      dense: Optional MedCPT ranker.
      retrieval_label: Override for retrieval_source (e.g. "augmented").

    Returns:
      New negative records in the canonical schema (may be empty).
    """
    ceiling = dates.parse_date(candidates.get("ceiling"))
    text_of = texts.drop_duplicates("pmid").set_index("pmid")
    exclude = set(labelled["pmid"].dropna()) | {review.get("pmid")}
    # Labelled records without PMIDs (e.g. Chan 2024) are matched by title
    # so a known positive can never re-enter the pool as a negative.
    labelled_titles = {
        hashing.text_fingerprint(t) for t in labelled["title"].fillna("")
    } - {""}

    def usable(pmid: str) -> bool:
        if pmid in exclude or pmid not in text_of.index:
            return False
        if hashing.text_fingerprint(text_of.at[pmid, "title"]) in (
            labelled_titles
        ):
            return False
        if ceiling is None:
            return True
        pub = dates.parse_date(text_of.at[pmid, "pub_date"])
        return pub is None or pub <= ceiling

    if candidates.get("tier") == "tier1_exact":
        ranked = [(p, "tier1") for p in candidates["tier1_ids"] if usable(p)]
    else:
        strict = [p for p in candidates.get("strict_ids", []) if usable(p)]
        broad = [
            p
            for p in dict.fromkeys(candidates.get("broad_ids", []))
            if usable(p) and p not in set(strict)
        ]
        top: list[str] = []
        if broad:
            query = " ".join(
                [
                    review.get("title", ""),
                    review.get("objective", ""),
                    review.get("selection_criteria", "")[:1500],
                ]
            )
            docs = (
                text_of.loc[broad, "title"].fillna("")
                + " "
                + text_of.loc[broad, "abstract"].fillna("")
            ).tolist()
            scores = [ranking.Bm25(docs).scores(query)]
            if dense is not None:
                try:
                    scores.append(
                        dense.scores(
                            query,
                            broad,
                            text_of.loc[broad, "title"].tolist(),
                            text_of.loc[broad, "abstract"].tolist(),
                        )
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logging.warning("dense ranking failed: %s", exc)
            fused = ranking.reciprocal_rank_fusion(scores, rrf_k)
            order = np.argsort(-fused, kind="stable")
            top = [broad[i] for i in order[: int(candidates.get("k", 2000))]]
        ranked = [(p, "boolean") for p in strict] + [(p, "hybrid") for p in top]
    budget = max(pool_max - len(labelled), 0)
    ranked = ranked[:budget]
    if not ranked:
        return schema.make_records_frame([])
    rows = []
    for rank, (pmid, source) in enumerate(ranked):
        rows.append(
            {
                "review_id": review["review_id"],
                "record_id": schema.pmid_record_id(pmid),
                "pmid": pmid,
                "label_final": 0,
                "label_abs": 0,
                "retrieval_rank": rank,
                "retrieval_source": retrieval_label or source,
            }
        )
    return schema.make_records_frame(rows)
