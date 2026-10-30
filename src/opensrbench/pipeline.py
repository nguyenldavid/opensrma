"""Data pipeline stages, in execution order.

  download  [network]  Fetch every enabled open dataset + Europe PMC harvest.
  ingest    [offline]  Parse originals into canonical interim tables.
  enrich    [network]  Resolve review PMIDs; fetch every citation's
                       title/abstract from PubMed (and OpenAlex) into the
                       SQLite caches.
  pools     [network]  Tier-1/Tier-2 candidate retrieval for reviews whose
                       pools are incomplete (harvested + legacy augmentation).
  build     [offline]  Merge sources, de-duplicate reviews, assemble pools
                       (BM25 + optional MedCPT on GPU), attach text, compute
                       fingerprints and criteria lists.
  split     [offline]  Family graph, split assignment, decontamination audit.
  freeze    [offline]  Frozen T10/T20/T40 tiers + pre-registration manifest.

Every stage is idempotent and resumable; network stages cache everything
they download, so they can be re-run after an interruption.
"""

import dataclasses
import glob
import json
import logging
import os

import pandas as pd
import tqdm

from opensrbench import config
from opensrbench import schema
from opensrbench.cohort import query as query_lib
from opensrbench.cohort import ranking
from opensrbench.cohort import reconstruct
from opensrbench.pubmed import eutils
from opensrbench.pubmed import openalex
from opensrbench.pubmed import parse
from opensrbench.pubmed import store as store_lib
from opensrbench.sources import base
from opensrbench.splits import assign
from opensrbench.splits import decontam
from opensrbench.splits import families as families_lib
from opensrbench.splits import tiers as tiers_lib
from opensrbench.text import criteria as criteria_lib
from opensrbench.utils import hashing
from opensrbench.utils import io

STAGES = ("download", "ingest", "enrich", "pools", "build", "split", "freeze")
NETWORK_STAGES = ("download", "enrich", "pools")


@dataclasses.dataclass
class Context:
    """Configuration bundle passed to every stage."""

    cfg: config.Config
    datasets: config.Config
    paths: config.Paths
    only: tuple[str, ...] = ()
    device: str | None = None

    @classmethod
    def create(
        cls,
        benchmark_config: str,
        datasets_config: str,
        only: tuple[str, ...] = (),
        device: str | None = None,
    ) -> "Context":
        cfg = config.Config.load(benchmark_config)
        return cls(
            cfg=cfg,
            datasets=config.Config.load(datasets_config),
            paths=config.Paths.from_config(cfg),
            only=only,
            device=device,
        )

    def enabled_sources(self) -> list[str]:
        sources = self.datasets.get("sources")
        names = [n for n, c in sources.items() if c.get("enabled", True)]
        if self.only:
            names = [n for n in names if n in self.only]
        return names

    def adapter(self, name: str) -> base.SourceAdapter:
        cls = base.registry()[name]
        return cls(
            self.datasets.get(f"sources.{name}"), self.paths, context=self.cfg
        )

    def eutils(self) -> eutils.EUtilsClient:
        return eutils.EUtilsClient(
            email=self.cfg.get("network.ncbi_email", ""),
            api_key=self.cfg.get("network.ncbi_api_key", ""),
            requests_per_second=float(
                self.cfg.get("network.ncbi_requests_per_second", 9)
            ),
        )

    def pubmed_store(self) -> store_lib.PubMedStore:
        return store_lib.PubMedStore(self.paths.pubmed_db)

    def openalex_store(self) -> store_lib.PubMedStore:
        return store_lib.PubMedStore(
            os.path.join(self.paths.root, "cache", "openalex.sqlite"),
            key_field="key",
        )


def _pool_path(paths: config.Paths, review_id: str) -> str:
    safe = review_id.replace(":", "__").replace("/", "_")
    return os.path.join(paths.pools, f"{safe}.json")


def load_interim(ctx: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenates interim tables of every enabled, ingested source."""
    reviews, records = [], []
    for name in ctx.enabled_sources():
        folder = ctx.paths.interim(name)
        if not os.path.exists(os.path.join(folder, "reviews.parquet")):
            logging.warning("source %s not ingested; skipping", name)
            continue
        reviews.append(io.read_parquet(os.path.join(folder, "reviews.parquet")))
        records.append(io.read_parquet(os.path.join(folder, "records.parquet")))
    if not reviews:
        raise RuntimeError(
            "No ingested sources. Run `osb data --stages "
            "download,ingest` first."
        )
    return (
        pd.concat(reviews, ignore_index=True),
        pd.concat(records, ignore_index=True),
    )


# ---------------------------------------------------------------------------
# Network stages.
# ---------------------------------------------------------------------------


def stage_download(ctx: Context) -> None:
    for name in ctx.enabled_sources():
        logging.info("== download %s", name)
        ctx.adapter(name).download()


def stage_ingest(ctx: Context) -> None:
    for name in ctx.enabled_sources():
        logging.info("== ingest %s", name)
        ctx.adapter(name).run_ingest()


def stage_enrich(ctx: Context) -> None:
    """Resolves review PMIDs and caches every citation's metadata."""
    reviews, records = load_interim(ctx)
    client = ctx.eutils()
    review_pmid = {
        r: p
        for r, p in zip(reviews["review_id"], reviews["pmid"])
        if schema.normalize_pmid(p)
    }
    todo = reviews[~reviews["review_id"].isin(review_pmid)]
    pmc = todo.dropna(subset=["pmcid"])
    if len(pmc):
        mapping = client.pmcid_to_pmid(pmc["pmcid"].tolist())
        review_pmid.update(
            {
                r: mapping[p]
                for r, p in zip(pmc["review_id"], pmc["pmcid"])
                if p in mapping
            }
        )
    todo = reviews[~reviews["review_id"].isin(review_pmid)]
    with_doi = todo.dropna(subset=["doi"])
    if len(with_doi):
        mapping = client.doi_to_pmid(with_doi["doi"].tolist())
        for review_id, doi in zip(with_doi["review_id"], with_doi["doi"]):
            key = str(doi).lower().split("doi.org/")[-1]
            if key in mapping:
                review_pmid[review_id] = mapping[key]
    todo = reviews[~reviews["review_id"].isin(review_pmid)]
    cochrane = todo[todo["review_key"].str.startswith("cochrane:")]
    for review_id, key in tqdm.tqdm(
        list(zip(cochrane["review_id"], cochrane["review_key"])),
        desc="cochrane",
    ):
        pmid = client.cochrane_to_pmid(key.split(":", 1)[1])
        if pmid:
            review_pmid[review_id] = pmid
    frame = pd.DataFrame(
        {
            "review_id": list(review_pmid),
            "review_pmid": list(review_pmid.values()),
        }
    )
    io.write_parquet(
        frame,
        os.path.join(
            ctx.paths.root, "interim", "enrich", "review_pmids.parquet"
        ),
    )
    # Records identified only by DOI (e.g. SYNERGY) -> PMID via PubMed.
    doi_path = os.path.join(
        ctx.paths.root, "interim", "enrich", "doi_pmids.parquet"
    )
    dois = (
        records.loc[records["record_id"].str.startswith("doi:"), "record_id"]
        .str.split(":", n=1)
        .str[1]
        .unique()
    )
    known = (
        io.read_parquet(doi_path)
        if os.path.exists(doi_path)
        else pd.DataFrame(columns=["doi", "pmid"])
    )
    todo_dois = sorted(set(dois) - set(known["doi"]))
    if todo_dois:
        found = client.doi_to_pmid(todo_dois)
        known = pd.concat(
            [
                known,
                pd.DataFrame(
                    {
                        "doi": todo_dois,
                        "pmid": [found.get(d) for d in todo_dois],
                    }
                ),
            ]
        )
        io.write_parquet(known, doi_path)
    batch = int(ctx.cfg.get("network.efetch_batch_size", 200))
    with ctx.pubmed_store() as store:
        pmids = (
            list(frame["review_pmid"])
            + records["pmid"].dropna().tolist()
            + known["pmid"].dropna().tolist()
        )
        store.fetch_missing(client, pmids, batch)
    unresolved = set(dois) - set(known.dropna()["doi"])
    oa_ids = (
        records.loc[
            records["record_id"].str.startswith("openalex:"), "record_id"
        ]
        .str.split(":", n=1)
        .str[1]
        .tolist()
    )
    oa_ids += [f"doi:{d}" for d in sorted(unresolved)]
    if oa_ids:
        try:
            with ctx.openalex_store() as oa_store:
                oa_store.fetch_missing(
                    openalex.OpenAlexClient(
                        mailto=ctx.cfg.get("network.ncbi_email", "")
                    ),
                    oa_ids,
                    50,
                )
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning(
                "OpenAlex fallback failed for %d records (%s); "
                "set OPENALEX_API_KEY and re-run enrich. These "
                "records will be dropped if they have no text.",
                len(oa_ids),
                exc,
            )


def _enriched_reviews(ctx: Context, reviews: pd.DataFrame) -> pd.DataFrame:
    """Fills missing review fields from the review's own PubMed record."""
    path = os.path.join(
        ctx.paths.root, "interim", "enrich", "review_pmids.parquet"
    )
    if not os.path.exists(path):
        return reviews
    mapping = io.read_parquet(path)
    out = reviews.merge(mapping, on="review_id", how="left")
    with ctx.pubmed_store() as store:
        cited = store.get_many(out["review_pmid"].dropna())
    for i, row in out.iterrows():
        citation = cited.get(row.get("review_pmid"))
        if not citation:
            continue
        fields = parse.review_fields_from_sections(
            citation.get("abstract_sections", [])
        )
        if not row["title"]:
            out.at[i, "title"] = citation.get("title", "")
        if not row["objective"]:
            out.at[i, "objective"] = fields["objective"]
        if not row["selection_criteria"]:
            out.at[i, "selection_criteria"] = fields["selection_criteria"]
        if not isinstance(row["pub_date"], str):
            out.at[i, "pub_date"] = citation.get("pub_date")
        if not schema.normalize_pmid(row["pmid"]):
            out.at[i, "pmid"] = citation.get("pmid")
    return out.drop(columns=["review_pmid"])


def _augmentation_targets(ctx: Context, reviews: pd.DataFrame) -> pd.DataFrame:
    incomplete = reviews[~reviews["pool_complete"].astype(bool)]
    harvested = incomplete[incomplete["source"] == "pmc_harvest"]
    legacy = incomplete[incomplete["source"] != "pmc_harvest"]
    if not ctx.cfg.get("cohort.augment_incomplete_pools", True):
        legacy = legacy.iloc[0:0]
    limit = int(ctx.cfg.get("cohort.augment_max_reviews", 25000))
    if len(legacy) > limit:
        order = legacy["review_id"].map(
            lambda r: hashing.sha256_hex(f"augment:{r}")
        )
        legacy = legacy.loc[order.sort_values().index[:limit]]
    return pd.concat([harvested, legacy])


def stage_pools(ctx: Context) -> None:
    """Fetches candidate pools (network) for incomplete-pool reviews."""
    reviews, records = load_interim(ctx)
    reviews = _enriched_reviews(ctx, reviews)
    targets = _augmentation_targets(ctx, reviews)
    cohort = ctx.cfg.get("cohort")
    lag = int(ctx.cfg.get("temporal.default_search_lag_days", 90))
    client = ctx.eutils()
    if cohort.get("query_generator") == "llm":
        from opensrbench.llm import \
            client as llm_client  # pylint: disable=import-outside-toplevel

        generator = query_lib.LlmQueryGenerator(
            llm_client.sync_chat_fn(ctx.cfg.get("llm.query_endpoint"))
        )
    else:
        generator = query_lib.HeuristicQueryGenerator()
    io.ensure_dir(ctx.paths.pools)
    positives = records[records["label_final"] == 1]
    pos_by_review = positives.groupby("review_id")["pmid"].apply(list)
    batch = int(ctx.cfg.get("network.efetch_batch_size", 200))
    with ctx.pubmed_store() as store:
        for review in tqdm.tqdm(targets.to_dict("records"), desc="pools"):
            path = _pool_path(ctx.paths, review["review_id"])
            if os.path.exists(path):
                continue
            review["extra"] = json.loads(review["extra"] or "{}")
            pos = [p for p in pos_by_review.get(review["review_id"], []) if p]
            pos_dates = [
                c.get("pub_date") for c in store.get_many(pos).values()
            ]
            ceiling = reconstruct.search_ceiling(
                review.get("search_date"),
                review.get("pub_date"),
                pos_dates,
                lag,
            )
            if review["source"] == "pmc_harvest":
                k = reconstruct.target_k(
                    review["extra"].get("prisma_pubmed"),
                    review["extra"].get("prisma_total"),
                    int(cohort["k_min"]),
                    int(cohort["k_max"]),
                    int(cohort["k_default"]),
                )
            else:
                k = int(cohort["augment_k"])
            candidates = reconstruct.fetch_candidates(
                review, ceiling, client, generator, cohort, k
            )
            ids = (
                candidates["tier1_ids"]
                + candidates["strict_ids"]
                + candidates["broad_ids"]
            )
            store.fetch_missing(client, ids, batch)
            io.write_json(candidates, path)


# ---------------------------------------------------------------------------
# Offline stages.
# ---------------------------------------------------------------------------


def _dedupe_reviews(ctx: Context, reviews: pd.DataFrame) -> pd.DataFrame:
    priority = {
        n: int(c.get("priority", 100))
        for n, c in ctx.datasets.get("sources").items()
    }
    df = reviews.copy()
    df["_priority"] = df["source"].map(priority).fillna(100)
    df = df.sort_values(["review_key", "_priority", "review_id"])
    kept = df.drop_duplicates("review_key", keep="first")
    logging.info("dedupe: %d -> %d reviews", len(df), len(kept))
    return kept.drop(columns=["_priority"])


def _attach_text(ctx: Context, records: pd.DataFrame) -> pd.DataFrame:
    """Fills title/abstract/pub_date from the PubMed and OpenAlex caches."""
    out = records.copy()
    doi_path = os.path.join(
        ctx.paths.root, "interim", "enrich", "doi_pmids.parquet"
    )
    is_doi = out["record_id"].str.startswith("doi:")
    if is_doi.any() and os.path.exists(doi_path):
        doi_map = io.read_parquet(doi_path).dropna().set_index("doi")["pmid"]
        resolved = out.loc[is_doi, "record_id"].str[4:].map(doi_map)
        out.loc[is_doi, "pmid"] = out.loc[is_doi, "pmid"].fillna(resolved)
    need = out["title"].isna() | (out["title"].fillna("") == "")
    pmids = out.loc[need & out["pmid"].notna(), "pmid"].unique().tolist()
    with ctx.pubmed_store() as store:
        texts = store.frame(pmids).set_index("pmid")
    for col in ("title", "abstract", "pub_date"):
        fill = out["pmid"].map(texts[col]) if len(texts) else None
        if fill is not None:
            out[col] = out[col].where(out[col].notna() & (out[col] != ""), fill)
    need = out["title"].isna() | (out["title"].fillna("") == "")
    oa_mask = need & (
        out["record_id"].str.startswith("openalex:")
        | out["record_id"].str.startswith("doi:")
    )
    if oa_mask.any():
        oa_ids = out.loc[oa_mask, "record_id"].map(
            lambda r: r.split(":", 1)[1] if r.startswith("openalex:") else r
        )
        with ctx.openalex_store() as oa_store:
            oa = oa_store.frame(oa_ids.unique()).set_index("key")
        for col in ("title", "abstract", "pub_date"):
            if len(oa):
                out.loc[oa_mask, col] = oa_ids.map(oa[col]).values
    pub_types = {}
    if pmids:
        with ctx.pubmed_store() as store:
            frame = store.frame(out["pmid"].dropna().unique())
            pub_types = dict(zip(frame["pmid"], frame["publication_types"]))
    out["publication_types"] = out["pmid"].map(
        lambda p: "|".join(pub_types.get(p) or []) if p else ""
    )
    for col in ("title", "abstract"):
        out[col] = out[col].fillna("").astype(str)
    return out


def _quality_filter_positives(
    records: pd.DataFrame, reviews: pd.DataFrame
) -> pd.DataFrame:
    """Drops implausible harvested positives (mis-parsed citations).

    For harvested reviews an "included study" that is itself a systematic
    review/meta-analysis (except for umbrella reviews), or that was
    published after the review, is almost always a parsing artefact.
    """
    info = reviews.set_index("review_id")
    source = records["review_id"].map(info["source"])
    rtype = records["review_id"].map(info["review_type"])
    review_pub = records["review_id"].map(info["pub_date"])
    harvested_pos = (source == "pmc_harvest") & (records["label_final"] == 1)
    is_sr = records["publication_types"].str.contains(
        "Systematic Review|Meta-Analysis", na=False
    )
    late = [
        bool(a and b and a > b)
        for a, b in zip(records["pub_date"].fillna(""), review_pub.fillna(""))
    ]
    drop = harvested_pos & (
        ((rtype != "rapid_scoping_umbrella") & is_sr)
        | pd.Series(late, index=records.index)
    )
    logging.info(
        "quality filter dropped %d harvested positives", int(drop.sum())
    )
    return records[~drop]


def stage_build(ctx: Context) -> None:
    """Merges all sources into processed/reviews + processed/records."""
    reviews, records = load_interim(ctx)
    reviews = _enriched_reviews(ctx, reviews)
    reviews = _dedupe_reviews(ctx, reviews)
    records = records[records["review_id"].isin(reviews["review_id"])]
    cohort = ctx.cfg.get("cohort")
    harvest = ctx.cfg.get("harvest")
    dense = None
    if cohort.get("dense", {}).get("enabled") and ctx.device != "none":
        try:
            dense = ranking.MedCptRanker(
                cohort["dense"]["query_encoder"],
                cohort["dense"]["article_encoder"],
                ctx.device,
                int(cohort["dense"]["batch_size"]),
            )
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("MedCPT unavailable (%s); BM25 only.", exc)
    new_records, tiers, pool_recall = [], {}, {}
    pool_files = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(ctx.paths.pools, "*.json"))
    }
    by_review = dict(tuple(records.groupby("review_id")))
    with ctx.pubmed_store() as store:
        for review in tqdm.tqdm(reviews.to_dict("records"), desc="assemble"):
            name = os.path.basename(_pool_path(ctx.paths, review["review_id"]))
            if name not in pool_files:
                continue
            candidates = io.read_json(pool_files[name])
            ids = (
                candidates["tier1_ids"]
                + candidates["strict_ids"]
                + candidates["broad_ids"]
            )
            texts = store.frame(ids)
            labelled = by_review.get(
                review["review_id"], schema.make_records_frame([])
            )
            augment = review["source"] != "pmc_harvest"
            negatives = reconstruct.assemble_pool(
                review,
                labelled,
                candidates,
                texts,
                int(harvest["pool_max"]),
                int(cohort["rrf_k"]),
                dense,
                retrieval_label="augmented" if augment else None,
            )
            new_records.append(negatives)
            tiers[review["review_id"]] = (
                "augmented" if augment else candidates["tier"]
            )
            positives = set(
                labelled.loc[labelled["label_final"] == 1, "pmid"].dropna()
            )
            if positives:
                pool_recall[review["review_id"]] = len(
                    positives & set(ids)
                ) / len(positives)
    if new_records:
        records = pd.concat([records] + new_records, ignore_index=True)
    records = schema.make_records_frame(records)
    reviews["pool_tier"] = (
        reviews["review_id"].map(tiers).fillna(reviews["pool_tier"])
    )
    reviews.loc[reviews["review_id"].isin(tiers), "pool_complete"] = True
    # Fraction of known positives the reconstructed search itself retrieved
    # (a search-sensitivity diagnostic reported in the cohort flow figure).
    reviews["pool_recall"] = reviews["review_id"].map(pool_recall)

    records = _attach_text(ctx, records)
    has_text = (records["title"] != "") | (records["abstract"] != "")
    records = records[has_text]
    records = _quality_filter_positives(records, reviews)
    records = decontam.add_fingerprints(records)
    # Same citation twice in one review (e.g. a PMID record and a text-only
    # copy): keep one, preferring the higher label and a PMID.
    with_fp = records["title_fp"] != ""
    ranked = records[with_fp].assign(_has_pmid=records["pmid"].notna())
    ranked = ranked.sort_values(
        ["label_final", "label_abs", "_has_pmid"], ascending=False
    )
    deduped = ranked.drop_duplicates(["review_id", "title_fp"])
    logging.info(
        "within-review title dedupe removed %d records",
        int(with_fp.sum() - len(deduped)),
    )
    records = pd.concat(
        [deduped.drop(columns="_has_pmid"), records[~with_fp]],
        ignore_index=True,
    )

    reviews["objective"] = reviews["objective"].where(
        reviews["objective"] != "", reviews["title"]
    )
    reviews["criteria_list"] = reviews["selection_criteria"].map(
        lambda t: json.dumps(criteria_lib.split_criteria(t))
    )
    stats = records.groupby("review_id").agg(
        n_pos=("label_final", "sum"), n_total=("label_final", "size")
    )
    pos_abs = (
        records[(records["label_final"] == 1) & (records["abstract"] != "")]
        .groupby("review_id")
        .size()
    )
    stats["n_pos_with_abstract"] = pos_abs.reindex(stats.index).fillna(0)
    stats["n_neg"] = stats["n_total"] - stats["n_pos"]
    reviews = reviews.merge(stats.reset_index(), on="review_id", how="inner")
    reviews = reviews[
        (reviews["n_pos"] > 0)
        & (reviews["n_neg"] > 0)
        & (reviews["selection_criteria"] != "")
    ]
    reviews["eligible_eval"] = (
        (reviews["source"] == "pmc_harvest")
        & (reviews["n_pos_with_abstract"] >= int(harvest["min_positives"]))
        & reviews["n_total"].between(
            int(harvest["pool_min"]), int(harvest["pool_max"])
        )
        & reviews["pool_tier"].isin(["tier1_exact", "tier2_hybrid"])
        & (
            reviews["pool_recall"].fillna(1.0)
            >= float(harvest.get("min_pool_recall", 0.0))
        )
    )
    records = records[records["review_id"].isin(reviews["review_id"])]
    schema.validate(reviews, records)
    io.write_parquet(reviews, ctx.paths.processed_file("reviews"))
    io.write_parquet(records, ctx.paths.processed_file("records"))
    summary = reviews.groupby("source").agg(
        reviews=("review_id", "size"),
        positives=("n_pos", "sum"),
        records=("n_total", "sum"),
        eligible_eval=("eligible_eval", "sum"),
        mean_pool_recall=("pool_recall", "mean"),
    )
    logging.info("build summary:\n%s", summary.to_string())
    io.write_json(
        summary.reset_index().to_dict("records"),
        os.path.join(ctx.paths.processed, "build_summary.json"),
    )


def _embedder(ctx: Context):
    name = ctx.cfg.get("splits.embedder", "tfidf")
    if name == "tfidf":
        return families_lib.tfidf_embed
    try:
        ranker = ranking.MedCptRanker(
            name, ctx.cfg.get("cohort.dense.article_encoder"), ctx.device
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning(
            "embedder %s unavailable (%s); using TF-IDF. Note "
            "the 0.88 threshold is calibrated for dense "
            "embeddings.",
            name,
            exc,
        )
        return families_lib.tfidf_embed
    return ranker.encode_queries


def _titan_test_keys(ctx: Context) -> set[str]:
    path = os.path.join(
        ctx.paths.interim("titan_sr"), "titan_split_keys.parquet"
    )
    keys: set[str] = set()
    if os.path.exists(path):
        frame = io.read_parquet(path)
        keys |= set(frame.loc[frame["titan_split"] == "test", "review_key"])
    reviews_path = os.path.join(
        ctx.paths.interim("titan_sr"), "reviews.parquet"
    )
    if os.path.exists(reviews_path):
        rv = io.read_parquet(reviews_path)
        ext = rv[rv["extra"].str.contains('"titan_split": "external"')]
        keys |= set(ext["review_key"])
    return keys


def stage_split(ctx: Context) -> None:
    """Families, split assignment, decontamination and audit."""
    reviews = io.read_parquet(ctx.paths.processed_file("reviews"))
    records = io.read_parquet(ctx.paths.processed_file("records"))
    scfg = ctx.cfg.get("splits")
    fams, fam_stats = families_lib.build_families(
        reviews,
        records,
        float(scfg["overlap_threshold"]),
        float(scfg["cosine_threshold"]),
        _embedder(ctx),
    )
    split_cfg = {**ctx.cfg.get("temporal"), **scfg}
    salt = ctx.cfg.get("benchmark.salt")
    splits = assign.assign_splits(
        reviews, fams, split_cfg, salt, _titan_test_keys(ctx)
    )
    split_of = splits.set_index("review_id")["split"]
    clean, audit = decontam.purge(
        records,
        split_of,
        tuple(scfg["purge_against"]),
        bool(scfg["purge_train_against_val"]),
    )
    bench = assign.select_benchmark(
        reviews, splits, int(scfg["eval_families_per_type"]), salt
    )
    splits["in_benchmark"] = splits["review_id"].map(bench).fillna(False)
    in_bench = clean["review_id"].isin(
        splits.loc[splits["in_benchmark"], "review_id"]
    )
    n_fams = splits.loc[splits["in_benchmark"], "family_id"].nunique()
    n_pos = int(clean.loc[in_bench, "label_final"].sum())
    release_ok = n_fams >= int(scfg["min_release_families"]) and n_pos >= int(
        scfg["min_release_appearances"]
    )
    if not release_ok:
        logging.warning(
            "Benchmark below release floor: %d families, %d "
            "positive appearances (floor %s/%s).",
            n_fams,
            n_pos,
            scfg["min_release_families"],
            scfg["min_release_appearances"],
        )
    report = {
        "families": fam_stats,
        "split_reviews": splits["split"].value_counts().to_dict(),
        "split_records": (
            clean["review_id"].map(split_of).value_counts().to_dict()
        ),
        "decontamination": audit,
        "benchmark": {
            "families": int(n_fams),
            "positives": n_pos,
            "reviews": int(splits["in_benchmark"].sum()),
            "meets_release_floor": bool(release_ok),
        },
    }
    io.write_parquet(splits, os.path.join(ctx.paths.splits, "splits.parquet"))
    io.write_parquet(clean, os.path.join(ctx.paths.splits, "records.parquet"))
    io.write_json(report, os.path.join(ctx.paths.splits, "audit.json"))
    logging.info("split report: %s", json.dumps(report, default=str)[:2000])


def load_split_data(
    paths: config.Paths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (reviews, decontaminated records, splits) for modelling."""
    reviews = io.read_parquet(paths.processed_file("reviews"))
    records = io.read_parquet(os.path.join(paths.splits, "records.parquet"))
    splits = io.read_parquet(os.path.join(paths.splits, "splits.parquet"))
    return reviews, records, splits


def stage_freeze(ctx: Context) -> None:
    """Writes frozen tiers + manifest for each held-out benchmark."""
    reviews, records, splits = load_split_data(ctx.paths)
    salt = ctx.cfg.get("benchmark.salt")
    tier_sizes = tuple(int(k) for k in ctx.cfg.get("benchmark.tiers"))
    audit = io.read_json(os.path.join(ctx.paths.splits, "audit.json"))
    benchmarks = {
        "open_sr_bench": splits.loc[splits["in_benchmark"], "review_id"],
        "open_sr_bench_extended": splits.loc[
            splits["split"] == "eval", "review_id"
        ],
        "titan_test": splits.loc[splits["split"] == "titan_test", "review_id"],
        "prospective": splits.loc[
            splits["split"] == "prospective", "review_id"
        ],
    }
    family_of = splits.set_index("review_id")["family_id"]
    rtype_of = reviews.set_index("review_id")["review_type"]
    for name, review_ids in benchmarks.items():
        subset = records[records["review_id"].isin(set(review_ids))]
        if subset.empty:
            logging.info("benchmark %s is empty; skipping", name)
            continue
        totals = (
            subset.groupby("review_id")["label_final"]
            .agg(n_total_pos="sum", n_total="size")
            .reset_index()
        )
        totals["n_total_neg"] = totals["n_total"] - totals["n_total_pos"]
        totals["family_id"] = totals["review_id"].map(family_of)
        totals["review_type"] = totals["review_id"].map(rtype_of)
        tier_df = tiers_lib.assign_frozen_tiers(
            subset[["review_id", "record_id", "label_final", "label_abs"]],
            salt,
            tier_sizes,
        )
        primary = f"in_tier_t{int(ctx.cfg.get('benchmark.primary_tier'))}"
        tier_df["in_stability_subset"] = False
        primary_rows = tier_df[tier_df[primary]]
        tier_df.loc[primary_rows.index, "in_stability_subset"] = (
            tiers_lib.stability_subset(
                primary_rows,
                float(ctx.cfg.get("benchmark.stability_fraction")),
                ctx.cfg.get("benchmark.stability_salt"),
            )
        )
        tier_df["family_id"] = tier_df["review_id"].map(family_of)
        out_dir = os.path.join(ctx.paths.frozen, name)
        io.write_parquet(tier_df, os.path.join(out_dir, "tiers.parquet"))
        io.write_parquet(totals, os.path.join(out_dir, "review_totals.parquet"))
        io.write_json(
            tiers_lib.manifest(
                tier_df, tier_sizes, salt, audit["decontamination"]
            ),
            os.path.join(out_dir, "manifest.json"),
        )


_STAGE_FNS = {
    "download": stage_download,
    "ingest": stage_ingest,
    "enrich": stage_enrich,
    "pools": stage_pools,
    "build": stage_build,
    "split": stage_split,
    "freeze": stage_freeze,
}


def run(ctx: Context, stages: list[str]) -> None:
    """Runs the requested stages in canonical order."""
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(
            f"unknown stages {sorted(unknown)}; choose from {STAGES}"
        )
    for stage in STAGES:
        if stage in stages:
            logging.info("======== stage: %s ========", stage)
            _STAGE_FNS[stage](ctx)
