"""Cohen et al. (2006) drug-class reviews (15 AHRQ/DERP reviews).

The TSV gives both the abstract-triage and article-triage decision for
every PubMed citation, i.e. both labels of the dual-stage design.
https://dmice.ohsu.edu/cohenaa/systematic-drug-class-review-data.html
"""

import os

import pandas as pd

from opensrbench import schema
from opensrbench.sources import base

_FILENAME = "epc-ir.clean.tsv"
_TOPIC_TITLES = {
    "ACEInhibitors": "Angiotensin-converting enzyme (ACE) inhibitors",
    "ADHD": (
        "Pharmacologic treatments for attention deficit hyperactivity "
        "disorder (ADHD)"
    ),
    "Antihistamines": "Newer antihistamines",
    "AtypicalAntipsychotics": "Atypical antipsychotics",
    "BetaBlockers": "Beta adrenergic blockers",
    "CalciumChannelBlockers": "Calcium channel blockers",
    "Estrogens": "Estrogens",
    "NSAIDS": "Nonsteroidal anti-inflammatory drugs (NSAIDs)",
    "Opiods": "Long-acting opioid analgesics",
    "OralHypoglycemics": "Oral hypoglycemics",
    "ProtonPumpInhibitors": "Proton pump inhibitors",
    "SkeletalMuscleRelaxants": "Skeletal muscle relaxants",
    "Statins": "HMG-CoA reductase inhibitors (statins)",
    "Triptans": "Triptans",
    "UrinaryIncontinence": (
        "Drugs for overactive bladder / urinary incontinence"
    ),
}


@base.register
class Cohen2006Adapter(base.SourceAdapter):
    """Adapter for the Cohen 2006 epc-ir TSV."""

    name = "cohen2006"

    def download(self) -> None:
        self.fetch(self.cfg["url"], _FILENAME)

    def ingest(self):
        df = pd.read_csv(
            os.path.join(self.raw_dir, _FILENAME),
            sep="\t",
            header=None,
            dtype=str,
            names=["topic", "endnote_id", "pmid", "abs", "art"],
        )
        reviews, records = [], []
        for topic, group in df.groupby("topic"):
            review_id = f"{self.name}:{topic}"
            drug_class = _TOPIC_TITLES.get(topic, topic)
            reviews.append(
                {
                    "review_id": review_id,
                    "source": self.name,
                    "review_key": f"derp:{topic.lower()}",
                    "title": f"Drug class review: {drug_class}",
                    "objective": (
                        f"To compare the comparative effectiveness and "
                        f"harms of {drug_class}."
                    ),
                    "selection_criteria": (
                        "Include: controlled trials and observational "
                        f"studies comparing {drug_class} head-to-head or "
                        "against placebo that report efficacy, effectiveness "
                        "or adverse events in the target population."
                    ),
                    "review_type": "intervention_no_ma",
                    "pool_complete": True,
                    "has_abs_labels": True,
                    "extra": {"criteria_quality": "templated"},
                }
            )
            for _, row in group.iterrows():
                pmid = schema.normalize_pmid(row["pmid"])
                if not pmid:
                    continue
                final = int(str(row["art"]).strip().upper() == "I")
                records.append(
                    {
                        "review_id": review_id,
                        "record_id": schema.pmid_record_id(pmid),
                        "pmid": pmid,
                        "label_final": final,
                        "label_abs": int(
                            str(row["abs"]).strip().upper() == "I" or final
                        ),
                    }
                )
        return (
            schema.make_reviews_frame(reviews),
            schema.make_records_frame(records),
        )
