"""Design-cue scan for review-conditional exclusion rules (ideas R1-R4).

Tests a large set of study-design / article-type cues (text phrases,
PubMed publication types, MeSH, language, abstract section labels) on the
pilot_v2 TRAIN split (inclusion rate with a one-sided 95% Wilson upper
bound) under several review conditions (all reviews, RCT-only reviews,
human-only reviews, non-diagnostic, non-qualitative, ...), then on VAL
counts how many negatives and *hard* negatives (scoring at or above their
review's lowest positive under a trained member) each cue would remove and
how many positives it would lose. A cue x condition is a veto candidate
when its train upper bound is <= --safe (default 0.1%). Only train and val
are read; nothing is locked or used by the pipeline.

  python scripts/design_cue_scan.py --root <FNX>/data_pilot_v2 \
      --preds <FNX>/runs_v2/biolinkbert_large/fold0/preds.parquet --out <dir>
"""

import argparse
import json
import math
import os
import sqlite3

import numpy as np
import pandas as pd

# name -> (field, regex). Fields: text = title + abstract (lower case),
# title (lower case), pt = publication types, mesh, lang, sections =
# structured-abstract labels (upper case), journal (lower case).
CUES = {
    # Case reports and small series.
    "we present/report a case": (
        "text",
        r"we (?:present|report|describe) (?:an?|the) (?:rare |unusual |unique |novel |first )?case",
    ),
    "case report (text)": ("text", r"\bcase reports?\b"),
    "case presentation": (
        "text",
        r"\bcase (?:presentation|description|summary)\b",
    ),
    "case series": ("text", r"\bcase series\b"),
    "N-year-old": ("text", r"\b\d{1,3}[- ](?:year|month|week|day)s?[- ]old\b"),
    "a/an N-year-old (wo)man/boy/girl": (
        "text",
        r"\ban? \d{1,3}[- ]year[- ]old (?:man|woman|male|female|boy|girl|patient|child|infant)\b",
    ),
    "rare case/entity": (
        "text",
        r"\brare (?:case|entity|presentation|complication|condition|cause)\b",
    ),
    "first reported case": ("text", r"\bfirst (?:reported )?(?:case|report)\b"),
    "to our knowledge first": (
        "text",
        r"to (?:the best of )?our knowledge,? (?:this is )?the first",
    ),
    "presented with / was admitted": (
        "text",
        r"\b(?:presented|was admitted|was referred) (?:to|with)\b",
    ),
    "literature review (with case)": (
        "text",
        r"(?:review of (?:the )?literature|literature review)",
    ),
    "images in / clinical image": (
        "text",
        r"\b(?:images? in|clinical image|picture quiz|photo quiz)\b",
    ),
    "novel mutation / pedigree": (
        "text",
        r"\b(?:novel (?:mutation|variant)|pedigree|kindred|proband)\b",
    ),
    # Opinion, correspondence, news.
    "letter to the editor": (
        "text",
        r"letter to the editor|dear (?:editor|sir)",
    ),
    "editorial": ("text", r"\beditorial\b"),
    "commentary/comment on": ("text", r"\bcommentary\b|\bcomment(?:s)? on\b"),
    "in reply / response to": (
        "text",
        r"\bin reply\b|\bresponse to\b|\breply to\b",
    ),
    "perspective(s)": ("text", r"\bperspectives?\b"),
    "viewpoint/opinion": (
        "text",
        r"\b(?:viewpoint|opinion|point of view|essay)\b",
    ),
    "we argue / we propose": (
        "text",
        r"\bwe (?:argue|propose|contend|suggest that)\b",
    ),
    "debate/controversy": ("text", r"\b(?:debate|controvers(?:y|ies|ial))\b"),
    "news/interview/obituary": (
        "text",
        r"\b(?:obituary|in memoriam|interview with|news)\b",
    ),
    "erratum/correction": (
        "text",
        r"\b(?:erratum|corrigendum|correction to|retraction)\b",
    ),
    # Reviews, guidelines, consensus.
    "narrative review": ("text", r"\bnarrative review\b"),
    "systematic review": ("text", r"\bsystematic(?:al(?:ly)?)? review"),
    "meta-analysis": ("text", r"\bmeta-?analys[ie]s\b"),
    "scoping/umbrella/rapid review": (
        "text",
        r"\b(?:scoping|umbrella|rapid|integrative|critical|state[- ]of[- ]the[- ]art) review\b",
    ),
    "we review / this review": (
        "text",
        r"\bwe (?:review|summari[sz]e|discuss|overview)\b|\bthis (?:review|article reviews|paper reviews)\b",
    ),
    "review (title)": ("title", r"\breview\b"),
    "overview/update/primer": (
        "title",
        r"\b(?:overview|an update|update on|primer|tutorial|state of the art)\b",
    ),
    "guideline/recommendation": (
        "text",
        r"\b(?:guidelines?|recommendations?)\b(?: for| on| from)",
    ),
    "consensus/Delphi": (
        "text",
        r"\b(?:consensus statement|consensus conference|expert consensus|delphi)\b",
    ),
    "position/policy statement": (
        "text",
        r"\b(?:position|policy|scientific) statement\b",
    ),
    "expert opinion/panel": ("text", r"\bexpert (?:opinion|panel)\b"),
    # Protocols, registrations, conference material.
    "study/trial protocol": (
        "text",
        r"\b(?:study|trial|review) protocol\b|\bprotocol for an?\b",
    ),
    "we will / will be randomised": (
        "text",
        r"\bwe will\b|\bwill be (?:randomi[sz]ed|recruited|enrolled|allocated)\b",
    ),
    "trial registration": (
        "text",
        r"\btrial registration\b|\bclinicaltrials\.gov\b|\bnct\d{8}\b|\bisrctn\b|\bprospero\b",
    ),
    "design and rationale": (
        "text",
        r"\b(?:design and rationale|rationale and design|methods paper)\b",
    ),
    "conference/meeting abstract": (
        "text",
        r"\b(?:abstract presented|presented at the|annual meeting|congress|symposium)\b",
    ),
    # Non-human, preclinical, laboratory.
    "mice/murine": ("text", r"\b(?:mice|mouse|murine)\b"),
    "rats": ("text", r"\brats?\b"),
    "rodent(s)": ("text", r"\brodents?\b"),
    "dogs/cats/horses/livestock": (
        "text",
        r"\b(?:dogs?|canine|cats?|feline|horses?|equine|cattle|bovine|sheep|ovine|pigs?|porcine|swine|goats?|poultry|chickens?)\b",
    ),
    "rabbits/guinea pigs/hamsters": (
        "text",
        r"\b(?:rabbits?|guinea pigs?|hamsters?|ferrets?)\b",
    ),
    "primates/monkeys": (
        "text",
        r"\b(?:non-?human primates?|monkeys?|macaques?|baboons?|marmosets?)\b",
    ),
    "zebrafish/drosophila/c elegans/yeast": (
        "text",
        r"\b(?:zebrafish|drosophila|c\. elegans|caenorhabditis|yeast|saccharomyces)\b",
    ),
    "in vitro": ("text", r"\bin vitro\b"),
    "ex vivo": ("text", r"\bex vivo\b"),
    "in silico / docking": (
        "text",
        r"\bin silico\b|\bmolecular docking\b|\bmolecular dynamics\b",
    ),
    "cell line(s)/cultured cells": (
        "text",
        r"\bcell lines?\b|\bcultured cells\b|\bhela\b|\bhek ?293\b",
    ),
    "animal model": ("text", r"\banimal models?\b|\bexperimental model\b"),
    "knockout/transgenic": (
        "text",
        r"\b(?:knock-?out|transgenic|wild-?type)\b",
    ),
    "xenograft": ("text", r"\bxenografts?\b"),
    "cadaver(s)": ("text", r"\bcadaver(?:s|ic)?\b"),
    "phantom/simulation": (
        "text",
        r"\b(?:phantom|simulation study|simulated|monte carlo)\b",
    ),
    "bacteria/strains/isolates": ("text", r"\b(?:strains?|isolates?)\b(?: of)"),
    "gene expression/proteomics": (
        "text",
        r"\b(?:gene expression|proteomic|transcriptom|rna-?seq|microarray)\w*",
    ),
    # Study designs (may be eligible or not depending on the review).
    "randomised (any)": ("text", r"\brandomi[sz](?:ed|ation)\b"),
    "double-blind / placebo": ("text", r"\bdouble-?blind\w*|\bplacebo\b"),
    "open-label / single-arm": ("text", r"\bopen-?label\b|\bsingle-?arm\b"),
    "non-randomised / quasi-experimental": (
        "text",
        r"\bnon-?randomi[sz]ed\b|\bquasi-?experimental\b",
    ),
    "before-after / pre-post": (
        "text",
        r"\bbefore[- ]and[- ]after\b|\bpre-?post\b|\bpre- and post-",
    ),
    "interrupted time series": ("text", r"\binterrupted time[- ]series\b"),
    "retrospective": ("text", r"\bretrospective(?:ly)?\b"),
    "chart/record review": (
        "text",
        r"\b(?:chart|medical record|case note)s? review\b",
    ),
    "prospective cohort": (
        "text",
        r"\bprospective (?:cohort|observational|study)\b",
    ),
    "cohort study": ("text", r"\bcohort stud(?:y|ies)\b"),
    "case-control": ("text", r"\bcase-?control\b"),
    "cross-sectional": ("text", r"\bcross-?sectional\b"),
    "survey/questionnaire": ("text", r"\b(?:survey|questionnaire)\b"),
    "online/web survey": (
        "text",
        r"\b(?:online|web-based|internet|national) survey\b",
    ),
    "qualitative/interviews/focus groups": (
        "text",
        r"\bqualitative\b|\bsemi-?structured interviews?\b|\bfocus groups?\b|\bthematic analysis\b",
    ),
    "mixed methods": ("text", r"\bmixed[- ]methods?\b"),
    "registry / database study": (
        "text",
        r"\bregistry\b|\bnationwide database\b|\bclaims data\b|\badministrative data\b",
    ),
    "ecological/population-level": (
        "text",
        r"\becological stud\w*|\bpopulation-level\b",
    ),
    "mendelian randomisation/GWAS": (
        "text",
        r"\bmendelian randomi[sz]ation\b|\bgenome-?wide association\b|\bgwas\b",
    ),
    "machine/deep learning": (
        "text",
        r"\b(?:machine learning|deep learning|neural network|artificial intelligence|random forest)\b",
    ),
    "prediction/prognostic model": (
        "text",
        r"\b(?:prediction|prognostic|risk) (?:model|score|tool|nomogram)\b",
    ),
    "diagnostic accuracy": (
        "text",
        r"\bdiagnostic accuracy\b|\bsensitivity and specificity\b|\breceiver operating\b|\barea under the (?:roc )?curve\b",
    ),
    "validation study": (
        "text",
        r"\bvalidation (?:study|cohort)\b|\bvalidated\b",
    ),
    "cost-effectiveness/economic": (
        "text",
        r"\bcost-?effective\w*|\bcost-?utility\b|\beconomic evaluation\b|\bqaly\b|\bmarkov model\b",
    ),
    "modelling study": (
        "text",
        r"\bmodel(?:l)?ing study\b|\bdecision analytic\b|\bmathematical model\b",
    ),
    "secondary/post hoc analysis": (
        "text",
        r"\b(?:secondary|post hoc|post-hoc|exploratory|subgroup) analys[ie]s\b",
    ),
    "pooled analysis / individual patient data": (
        "text",
        r"\bpooled analysis\b|\bindividual (?:patient|participant) data\b",
    ),
    "extension / long-term follow-up": (
        "text",
        r"\bopen-label extension\b|\blong-term follow-?up\b|\bextension study\b",
    ),
    "pilot/feasibility": (
        "text",
        r"\bpilot (?:study|trial|rct)\b|\bfeasibility (?:study|trial)\b",
    ),
    "phase I / dose escalation / PK": (
        "text",
        r"\bphase (?:i|1|0)\b(?!i)|\bdose[- ]escalation\b|\bpharmacokinetic\w*|\bfirst-in-human\b",
    ),
    "phase II": ("text", r"\bphase (?:ii|2)\b(?!i)"),
    "phase III/IV": ("text", r"\bphase (?:iii|3|iv|4)\b"),
    "healthy volunteers": (
        "text",
        r"\bhealthy (?:volunteers|subjects|adults|participants)\b",
    ),
    "bioequivalence": ("text", r"\bbioequivalen\w*"),
    "quality improvement / audit": (
        "text",
        r"\bquality improvement\b|\bclinical audit\b|\baudit\b",
    ),
    "implementation / process evaluation": (
        "text",
        r"\bimplementation (?:study|science|research)\b|\bprocess evaluation\b",
    ),
    "technical note / surgical technique": (
        "text",
        r"\btechnical note\b|\bsurgical technique\b|\bhow (?:i|we) do it\b|\bstep-by-step\b",
    ),
    "video / educational": (
        "text",
        r"\bvideo\b|\beducational (?:module|program|intervention)\b|\bcurriculum\b",
    ),
    "historical/biography": (
        "text",
        r"\bhistor(?:y|ical) of\b|\bbiography\b|\bcentury\b",
    ),
    "methodological / statistical methods": (
        "text",
        r"\bmethodological (?:study|review|paper)\b|\bstatistical methods? for\b|\bsample size calculation\b",
    ),
    "reporting guideline / checklist": (
        "text",
        r"\breporting guideline\b|\bchecklist\b|\bconsort\b|\bprisma\b|\bstrobe\b",
    ),
    # Populations (often decisive for eligibility).
    "children/paediatric": (
        "text",
        r"\b(?:children|child|paediatric|pediatric|infants?|neonat\w*|adolescents?|newborns?)\b",
    ),
    "pregnancy": ("text", r"\bpregnan\w*|\bantenatal\b|\bprenatal\b"),
    "elderly/older adults": (
        "text",
        r"\b(?:elderly|older adults|aged 65|nursing home)\b",
    ),
    "healthy population / general population": (
        "text",
        r"\bgeneral population\b|\bcommunity-dwelling\b",
    ),
    "students/health professionals": (
        "text",
        r"\b(?:medical|nursing) students?\b|\bhealth(?:care)? professionals?\b|\bphysicians'?\b|\bnurses'?\b",
    ),
    # Structured-abstract section labels.
    "sections: CASE PRESENTATION/REPORT": (
        "sections",
        r"CASE (?:PRESENTATION|REPORT|DESCRIPTION|SUMMARY)",
    ),
    "sections: TRIAL REGISTRATION": (
        "sections",
        r"TRIAL REGISTRATION|REGISTRATION",
    ),
    "sections: ETHICS AND DISSEMINATION": (
        "sections",
        r"ETHICS AND DISSEMINATION",
    ),
    "sections: METHODS AND ANALYSIS": ("sections", r"METHODS AND ANALYSIS"),
    "sections: DATA SOURCES/SELECTION": (
        "sections",
        r"DATA SOURCES|STUDY SELECTION|DATA EXTRACTION|ELIGIBILITY CRITERIA",
    ),
    "sections: EVIDENCE ACQUISITION": (
        "sections",
        r"EVIDENCE ACQUISITION|EVIDENCE SYNTHESIS",
    ),
    # Language and journal.
    "non-English": ("lang", r"^(?!eng$).+"),
    "journal: case reports": ("journal", r"case rep"),
    "journal: veterinary": ("journal", r"veterinar|vet\b|animal"),
    "journal: protocols": ("journal", r"protocols?"),
    "journal: cochrane database": ("journal", r"cochrane database"),
    "journal: letters/news": ("journal", r"letters?|news"),
    # MeSH.
    "MeSH Animals without Humans": ("mesh_animals_only", None),
    "MeSH Humans": ("mesh", r"(?:^|\|)Humans(?:\||$)"),
    "MeSH Animals": ("mesh", r"(?:^|\|)Animals(?:\||$)"),
    "MeSH Mice/Rats": ("mesh", r"(?:^|\|)(?:Mice|Rats)(?:\||$)"),
    "MeSH Cell Line": ("mesh", r"Cell Line"),
    "MeSH In Vitro Techniques": ("mesh", r"In Vitro Techniques"),
    "MeSH Child/Infant": (
        "mesh",
        r"(?:^|\|)(?:Child|Infant|Infant, Newborn|Child, Preschool)(?:\||$)",
    ),
    "MeSH Pregnancy": ("mesh", r"(?:^|\|)Pregnancy(?:\||$)"),
    "MeSH Surveys and Questionnaires": ("mesh", r"Surveys and Questionnaires"),
    "MeSH Qualitative Research": ("mesh", r"Qualitative Research"),
    "MeSH Cost-Benefit Analysis": (
        "mesh",
        r"Cost-Benefit Analysis|Costs and Cost Analysis",
    ),
    "MeSH Retrospective Studies": ("mesh", r"Retrospective Studies"),
    "MeSH Cross-Sectional Studies": ("mesh", r"Cross-Sectional Studies"),
    "MeSH Case-Control Studies": ("mesh", r"Case-Control Studies"),
    "MeSH Cohort/Prospective Studies": (
        "mesh",
        r"Cohort Studies|Prospective Studies|Longitudinal Studies",
    ),
    "MeSH Sensitivity and Specificity": (
        "mesh",
        r"Sensitivity and Specificity|ROC Curve",
    ),
}
# Every PubMed publication type seen in the data is added automatically.

CONDITIONS = {
    "all reviews": lambda r: np.ones(len(r), bool),
    "RCT-only reviews": lambda r: r["rct_only"],
    "human-only reviews": lambda r: ~r["animal_review"],
    "non-diagnostic reviews": lambda r: ~r["diagnostic_review"],
    "non-qualitative reviews": lambda r: ~r["qualitative_review"],
    "no case reports allowed": lambda r: ~r["allows_case_reports"],
}


def _review_flags(reviews: pd.DataFrame) -> pd.DataFrame:
    text = (
        reviews["objective"].fillna("")
        + " "
        + reviews["selection_criteria"].fillna("")
    ).str.lower()
    randomised = text.str.contains(r"randomi[sz]ed|\brcts?\b")
    broader = text.str.contains(
        r"case reports?|case series|observational|cohort|case.control|"
        r"non.?randomi[sz]ed|quasi|before.and.after|cross.sectional|"
        r"any (?:study )?design|all study designs|uncontrolled"
    )
    return pd.DataFrame(
        {
            "rct_only": randomised & ~broader,
            "allows_case_reports": text.str.contains(
                r"case reports?|case series|any (?:study )?design|all study designs"
            ),
            "animal_review": text.str.contains(
                r"\banimals?\b|preclinical|rodents?|\bmice\b|\brats?\b|in vivo"
            ),
            "diagnostic_review": text.str.contains(
                r"diagnostic|sensitivity|specificity|accuracy|index test|reference standard"
            ),
            "qualitative_review": text.str.contains(
                r"qualitative|experiences?|perceptions?|views of|interviews?|focus groups?"
            ),
        },
        index=reviews.index,
    )


def _wilson_upper(pos: int, n: int, z: float = 1.645) -> float:
    if n == 0:
        return 1.0
    p = pos / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return min(1.0, (centre + margin) / (1 + z * z / n))


def _load_meta(cache: str, pmids) -> pd.DataFrame:
    con = sqlite3.connect(cache)
    rows = []
    pmids = [str(p) for p in pmids if isinstance(p, str) and p]
    for start in range(0, len(pmids), 900):
        batch = pmids[start : start + 900]
        marks = ",".join("?" * len(batch))
        for pmid, payload in con.execute(
            f"SELECT pmid, payload FROM citations WHERE pmid IN ({marks})",
            batch,
        ):
            d = json.loads(payload)
            mesh = d.get("mesh_terms") or []
            if isinstance(mesh, str):
                mesh = [m.strip(" '\"") for m in mesh.strip("[]").split(",")]
            sections = d.get("abstract_sections") or []
            labels = [
                s.get("label", "") if isinstance(s, dict) else str(s)
                for s in sections
            ]
            rows.append(
                {
                    "pmid": pmid,
                    "mesh": "|".join(mesh),
                    "lang": d.get("language") or "",
                    "journal": (d.get("journal") or "").lower(),
                    "sections": " ".join(labels).upper(),
                }
            )
    return pd.DataFrame(
        rows, columns=["pmid", "mesh", "lang", "journal", "sections"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--safe", type=float, default=0.001)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    splits = pd.read_parquet(f"{args.root}/splits/splits.parquet")
    split_of = splits.drop_duplicates("review_id").set_index("review_id")[
        "split"
    ]
    reviews = pd.read_parquet(
        f"{args.root}/processed/reviews.parquet",
        columns=["review_id", "objective", "selection_criteria"],
    ).set_index("review_id")
    flags = _review_flags(reviews)
    keep = list(split_of[split_of.isin(["train", "val"])].index)
    recs = pd.read_parquet(
        f"{args.root}/processed/records.parquet",
        columns=[
            "review_id",
            "record_id",
            "pmid",
            "title",
            "abstract",
            "label_final",
            "publication_types",
        ],
        filters=[("review_id", "in", keep)],
    )
    recs["split"] = recs["review_id"].map(split_of)
    meta = _load_meta(
        f"{args.root}/cache/pubmed.sqlite", recs["pmid"].dropna().unique()
    )
    recs = recs.merge(meta, on="pmid", how="left")
    for col in ("mesh", "lang", "journal", "sections"):
        recs[col] = recs[col].fillna("")
    recs = recs.join(flags, on="review_id")
    recs["text"] = (
        recs["title"].fillna("") + " " + recs["abstract"].fillna("")
    ).str.lower()
    recs["title"] = recs["title"].fillna("").str.lower()
    recs["pt"] = recs["publication_types"].fillna("")

    preds = pd.read_parquet(args.preds)
    preds = preds[preds["set"] == "val"][
        ["review_id", "record_id", "logit_final"]
    ]
    recs = recs.merge(preds, on=["review_id", "record_id"], how="left")
    min_pos = (
        recs[(recs.split == "val") & (recs.label_final == 1)]
        .groupby("review_id")["logit_final"]
        .min()
    )
    recs["hard"] = (
        (recs.split == "val")
        & (recs.label_final == 0)
        & (recs["logit_final"] >= recs["review_id"].map(min_pos))
    )

    cues = dict(CUES)
    for pt in sorted({t for v in recs["pt"] for t in v.split("|") if t}):
        cues[f"PT: {pt}"] = ("pt_exact", pt)

    hits = {}
    for name, (field, pattern) in cues.items():
        if field == "mesh_animals_only":
            m = recs["mesh"].str.contains(r"(?:^|\|)Animals(?:\||$)") & ~recs[
                "mesh"
            ].str.contains(r"(?:^|\|)Humans(?:\||$)")
        elif field == "pt_exact":
            m = recs["pt"].str.split("|").apply(lambda ts, t=pattern: t in ts)
        else:
            m = recs[field].str.contains(pattern, regex=True)
        hits[name] = m.to_numpy()

    train = (recs.split == "train").to_numpy()
    val = (recs.split == "val").to_numpy()
    y = recs["label_final"].to_numpy() == 1
    hard = recs["hard"].to_numpy()
    rows = []
    for name, m in hits.items():
        for cond, fn in CONDITIONS.items():
            c = np.asarray(fn(recs), bool)
            t = m & c & train
            v = m & c & val
            n_t, p_t = int(t.sum()), int((t & y).sum())
            rows.append(
                {
                    "cue": name,
                    "condition": cond,
                    "train_hits": n_t,
                    "train_pos": p_t,
                    "train_incl_rate": p_t / n_t if n_t else np.nan,
                    "train_ub95": _wilson_upper(p_t, n_t),
                    "val_neg_removed": int((v & ~y).sum()),
                    "val_hard_neg_removed": int((v & hard).sum()),
                    "val_pos_lost": int((v & y).sum()),
                }
            )
    table = pd.DataFrame(rows)
    table["safe"] = table["train_ub95"] <= args.safe
    table.to_csv(f"{args.out}/cue_scan.csv", index=False)

    safe = table[table.safe & (table.train_hits > 0)]
    union = np.zeros(len(recs), bool)
    for _, r in safe.iterrows():
        union |= hits[r.cue] & np.asarray(CONDITIONS[r.condition](recs), bool)
    n_hard = int(hard.sum())
    summary = {
        "cues": len(cues),
        "cue_x_condition": len(table),
        "safe_rules": int(len(safe)),
        "union_val_neg_removed": int((union & val & ~y).sum()),
        "union_val_hard_neg_removed": int((union & hard).sum()),
        "val_hard_negatives": n_hard,
        "union_val_pos_lost": int((union & val & y).sum()),
        "val_negatives": int((val & ~y).sum()),
        "val_positives": int((val & y).sum()),
    }
    with open(f"{args.out}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))
    show = table.sort_values(["safe", "val_hard_neg_removed"], ascending=False)
    cols = [
        "cue",
        "condition",
        "train_hits",
        "train_pos",
        "train_ub95",
        "val_hard_neg_removed",
        "val_neg_removed",
        "val_pos_lost",
    ]
    print(
        "\nSafe rules (train UB95 <= %.3f), by val hard negatives removed:"
        % args.safe
    )
    print(
        safe.sort_values("val_hard_neg_removed", ascending=False)[cols]
        .head(40)
        .to_string(index=False)
    )
    print("\nNear-safe (UB95 <= 1%), by val hard negatives removed:")
    near = table[(table.train_ub95 <= 0.01) & ~table.safe]
    print(
        near.sort_values("val_hard_neg_removed", ascending=False)[cols]
        .head(25)
        .to_string(index=False)
    )
    print("\nLargest hard-negative coverage overall (any rate):")
    print(
        table[table.condition == "all reviews"]
        .sort_values("val_hard_neg_removed", ascending=False)[
            cols + ["train_incl_rate"]
        ]
        .head(25)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
