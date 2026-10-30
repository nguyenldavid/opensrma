"""Automated Europe PMC harvest of open-access systematic reviews.

download()  (network)
  1. Europe PMC search over the configured window(s) -> hits.jsonl.
  2. JATS full text for every hit (gzip cache, resumable).
  3. For reviews with >= min_positives candidate included studies: resolve
     references lacking a <pub-id pmid> via Europe PMC's reference API
     (title/DOI alignment) and NCBI DOI->PMID lookups -> refmap/<pmcid>.json.

ingest()    (offline)
  Applies eligibility criteria 1-3 and 5 of the protocol and emits reviews
  with their included (label_final=1) and full-text-excluded (label_abs=1,
  label_final=0) studies. Negatives are added by the cohort reconstruction
  stage (criterion 4), so every review starts with pool_complete=False.
"""

import gzip
import json
import logging
import os

import pandas as pd
import tqdm

from opensrbench import schema
from opensrbench.harvest import europepmc
from opensrbench.harvest import jats
from opensrbench.pubmed import eutils
from opensrbench.sources import base
from opensrbench.text import review_type
from opensrbench.utils import hashing
from opensrbench.utils import io


def _norm_title(text: str) -> str:
    return hashing.normalize_for_fingerprint(text)[:120]


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / max(1, len(ta | tb))


def resolve_references(
    parsed: jats.ParsedReview, api_refs: list[dict], doi_map: dict[str, str]
) -> dict[str, str]:
    """Maps ref ids to PMIDs using pub-ids, the Europe PMC list and DOIs.

    Args:
      parsed: Parsed review.
      api_refs: Europe PMC `references` payload for the review.
      doi_map: Lower-cased DOI -> PMID mapping.

    Returns:
      rid -> PMID for every resolvable reference.
    """
    by_title, by_doi = {}, {}
    for ref in api_refs:
        if ref.get("source") == "MED" and str(ref.get("id", "")).isdigit():
            if ref.get("title"):
                by_title[_norm_title(ref["title"])] = (ref["id"], ref["title"])
            if ref.get("doi"):
                by_doi[ref["doi"].lower()] = ref["id"]
    out = {}
    for rid, ref in parsed.references.items():
        pmid = ref.pmid
        if not pmid and ref.doi:
            pmid = by_doi.get(ref.doi) or doi_map.get(ref.doi)
        if not pmid and ref.title:
            hit = by_title.get(_norm_title(ref.title))
            norm_ref = hashing.normalize_for_fingerprint(ref.title)
            if not hit and len(norm_ref) > 60:
                # Unstructured citation string: look for a known title in it.
                hit = next(
                    (
                        v
                        for k, v in by_title.items()
                        if len(k) >= 25 and k in norm_ref
                    ),
                    None,
                )
            if hit:
                pmid = hit[0]
            else:
                best = max(
                    by_title.values(),
                    default=None,
                    key=lambda v, t=ref.title: _token_jaccard(t, v[1]),
                )
                if best and _token_jaccard(ref.title, best[1]) >= 0.7:
                    pmid = best[0]
        if pmid:
            out[rid] = str(pmid)
    return out


@base.register
class PmcHarvestAdapter(base.SourceAdapter):
    """Europe PMC JATS harvester (the Open-SR-Bench core)."""

    name = "pmc_harvest"

    @property
    def harvest_cfg(self) -> dict:
        return self.context.get("harvest", {})

    @property
    def network_cfg(self) -> dict:
        return self.context.get("network", {})

    def _windows(self) -> list[tuple[str, str]]:
        windows = [
            (self.harvest_cfg["window_start"], self.harvest_cfg["window_end"])
        ]
        if self.harvest_cfg.get("train_window"):
            windows.append(tuple(self.harvest_cfg["train_window"]))
        return windows

    def download(self) -> None:
        epmc = europepmc.EuropePmcClient(
            self.paths.europepmc_cache,
            float(self.network_cfg.get("europepmc_requests_per_second", 8)),
        )
        ncbi = eutils.EUtilsClient(
            email=self.network_cfg.get("ncbi_email", ""),
            api_key=self.network_cfg.get("ncbi_api_key", ""),
            requests_per_second=float(
                self.network_cfg.get("ncbi_requests_per_second", 9)
            ),
        )
        io.ensure_dir(self.raw_dir)
        hits_path = os.path.join(self.raw_dir, "hits.jsonl")
        if not os.path.exists(hits_path):
            tmp = hits_path + ".part"
            with open(tmp, "w", encoding="utf-8") as fh:
                for start, end in self._windows():
                    query = (
                        f"({self.harvest_cfg['query']}) AND "
                        f"(FIRST_PDATE:[{start} TO {end}])"
                    )
                    for hit in epmc.search(
                        query, int(self.harvest_cfg.get("max_reviews", 50000))
                    ):
                        if hit.get("pmcid"):
                            fh.write(json.dumps(hit) + "\n")
            os.replace(tmp, hits_path)
        hits = pd.read_json(hits_path, lines=True, dtype=False)
        refmap_dir = io.ensure_dir(os.path.join(self.raw_dir, "refmap"))
        min_pos = int(self.harvest_cfg.get("min_positives", 5))
        for pmcid in tqdm.tqdm(hits["pmcid"].drop_duplicates(), desc="jats"):
            out_path = os.path.join(refmap_dir, f"{pmcid}.json")
            if os.path.exists(out_path):
                continue
            xml = epmc.fetch_fulltext(pmcid)
            if not xml:
                continue
            parsed = jats.parse_jats(xml, pmcid)
            needed = parsed.included_rids + parsed.excluded_rids
            mapping: dict[str, str] = {}
            if (
                len(parsed.included_rids) >= min_pos
                and parsed.selection_criteria
            ):
                api_refs = []
                if any(not parsed.references[r].pmid for r in needed):
                    api_refs = epmc.references(pmcid)
                dois = [
                    parsed.references[r].doi
                    for r in needed
                    if not parsed.references[r].pmid
                    and parsed.references[r].doi
                ]
                doi_map = ncbi.doi_to_pmid(dois) if dois else {}
                mapping = resolve_references(parsed, api_refs, doi_map)
            io.write_json({"pmcid": pmcid, "rid_to_pmid": mapping}, out_path)

    def ingest(self):
        hits = pd.read_json(
            os.path.join(self.raw_dir, "hits.jsonl"), lines=True, dtype=False
        ).drop_duplicates("pmcid")
        min_pos = int(self.harvest_cfg.get("min_positives", 5))
        reviews, records = [], []
        rejected: dict[str, int] = {}

        def reject(reason: str) -> None:
            rejected[reason] = rejected.get(reason, 0) + 1

        for hit in tqdm.tqdm(hits.to_dict("records"), desc="ingest"):
            pmcid = hit["pmcid"]
            refmap_path = os.path.join(self.raw_dir, "refmap", f"{pmcid}.json")
            xml_path = os.path.join(
                self.paths.europepmc_cache, f"{pmcid}.xml.gz"
            )
            if not (os.path.exists(refmap_path) and os.path.exists(xml_path)):
                reject("no_fulltext")
                continue
            with gzip.open(xml_path, "rb") as fh:
                parsed = jats.parse_jats(fh.read(), pmcid)
            mapping = io.read_json(refmap_path)["rid_to_pmid"]
            if not parsed.selection_criteria:
                reject("no_criteria")
                continue
            positives = list(
                dict.fromkeys(
                    mapping[r] for r in parsed.included_rids if r in mapping
                )
            )
            ft_excluded = [
                mapping[r]
                for r in parsed.excluded_rids
                if r in mapping and mapping[r] not in positives
            ]
            if len(positives) < min_pos:
                reject("few_resolved_positives")
                continue
            review_id = f"{self.name}:{pmcid}"
            rtype = review_type.classify(
                parsed.title, parsed.objective, parsed.abstract
            )
            reviews.append(
                {
                    "review_id": review_id,
                    "source": self.name,
                    "review_key": f"pmc:{pmcid}",
                    "title": parsed.title or hit.get("title", ""),
                    "objective": parsed.objective,
                    "selection_criteria": parsed.selection_criteria,
                    "review_type": rtype,
                    "pub_date": (
                        hit.get("firstPublicationDate") or parsed.pub_date
                    ),
                    "search_date": parsed.search_date,
                    "pool_complete": False,
                    "pool_tier": "pending",
                    "has_abs_labels": bool(ft_excluded),
                    "doi": parsed.doi or hit.get("doi"),
                    "pmid": parsed.pmid or hit.get("pmid"),
                    "pmcid": pmcid,
                    "extra": {
                        "positive_source": parsed.positive_source,
                        "search_strategy": parsed.search_strategy,
                        "prisma_pubmed": parsed.prisma_pubmed,
                        "prisma_total": parsed.prisma_total,
                        "n_included_rids": len(parsed.included_rids),
                    },
                }
            )
            for pmid in positives:
                records.append(
                    {
                        "review_id": review_id,
                        "record_id": schema.pmid_record_id(pmid),
                        "pmid": pmid,
                        "label_final": 1,
                        "retrieval_source": "included",
                    }
                )
            for pmid in ft_excluded:
                records.append(
                    {
                        "review_id": review_id,
                        "record_id": schema.pmid_record_id(pmid),
                        "pmid": pmid,
                        "label_final": 0,
                        "label_abs": 1,
                        "retrieval_source": "ft_excluded",
                    }
                )
        logging.info(
            "[pmc_harvest] kept %d reviews; rejected %s", len(reviews), rejected
        )
        io.write_json(
            {"kept": len(reviews), "rejected": rejected},
            os.path.join(self.paths.interim(self.name), "harvest_audit.json"),
        )
        return (
            schema.make_reviews_frame(reviews),
            schema.make_records_frame(records),
        )
