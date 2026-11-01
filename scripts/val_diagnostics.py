"""Validation-only CPU diagnostics for the pilot_v2 members (no titan_test).

  V1  Ensembles: TITAN-SR vs TITAN-Open combiners. Oracle per-review
      ranking metrics on val, and deployable operating points locked on
      the train-fold holdout (equal_review / equal_review_lcb) and applied
      to val: the dry run of "lock on val, evaluate on titan_test". Paired
      family bootstrap of the Open - SR differences.
  V2  Positive-label noise (idea N1): are non-primary "included" studies
      (reviews, editorials, letters, guidelines) the positives that set the
      99%-recall cut-off?
  V3  Error analysis: which val reviews have low specificity at 99% recall
      (source, size, prevalence, criteria length / truncation, noise).
  V4  Bi-encoder feature (idea B3; needs `--biencoder`): MedCPT z-scores
      added to the ensemble z-score, weight chosen on the holdout.
  V6  Review-adaptive thresholds: the global threshold is applied to the
      ensemble score re-centred on statistics of the review's own
      (unlabelled) candidate scores; locked on the holdout, applied to val.
  V5  Publication-type coverage per source and how much the PubMed cache
      could backfill (idea R5).

  python scripts/val_diagnostics.py --data-root $FNX/data_pilot_v2 \
      --runs-root $FNX/runs_v2 --out $FNX/val_diagnostics \
      [--biencoder $FNX/biencoder_probe_v2/medcpt]
"""

import argparse
import json
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from design_cue_scan import _review_flags  # pylint: disable=wrong-import-position
from opensrbench.calibration import thresholds
from opensrbench.evaluation import ranking

SR = ["titan_sr_pubmedbert", "titan_sr_biolinkbert"]
OPEN = ["biolinkbert_large", "biomedbert_large", "deberta_v3_large"]
KEYS = ["review_id", "record_id"]
NONPRIMARY_PT = {
    "Review",
    "Systematic Review",
    "Meta-Analysis",
    "Editorial",
    "Letter",
    "Comment",
    "Practice Guideline",
    "Guideline",
    "Consensus Development Conference",
    "News",
}
NONPRIMARY_TITLE = (
    r"(?:narrative|systematic|scoping|literature|critical|umbrella|"
    r"an? (?:brief |short |comprehensive )?)review\b|meta-analys|"
    r"^editorial|letter to the editor|^comment|commentary|"
    r"(?:practice |clinical )?guidelines?\b|consensus statement|"
    r"position statement|^reply|^response to"
)
TARGETS = (0.99, 0.995)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_members(runs_root: str, file: str) -> tuple[pd.DataFrame, list]:
    wide, have = None, []
    for b in SR + OPEN:
        path = f"{runs_root}/{b}/fold0/{file}"
        finished = os.path.exists(f"{runs_root}/{b}/fold0/summary.json")
        if not finished or not os.path.exists(path):
            continue
        p = pd.read_parquet(path)
        if "set" in p:
            p = p[p["set"] == "val"]
        p = p[KEYS + ["label_final", "logit_final"]].rename(
            columns={"logit_final": b}
        )
        wide = (
            p
            if wide is None
            else wide.merge(p.drop(columns="label_final"), on=KEYS)
        )
        have.append(b)
    return wide, have


def add_systems(df: pd.DataFrame, have: list) -> list:
    """Adds system score columns; returns their names."""
    names = []
    z = df.groupby("review_id")[have].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-6)
    )
    for b in have:
        df[f"z:{b}"] = z[b]
    groups = {
        "titan_sr": [b for b in SR if b in have],
        "titan_open": [b for b in OPEN if b in have],
        "all": have,
    }
    for name, members in groups.items():
        if not members:
            continue
        df[f"{name}:mean_sigmoid"] = _sigmoid(df[members]).mean(axis=1)
        df[f"{name}:mean_z"] = df[[f"z:{b}" for b in members]].mean(axis=1)
        names += [f"{name}:mean_sigmoid", f"{name}:mean_z"]
    for b in have:
        df[f"single:{b}"] = df[b]
        names.append(f"single:{b}")
    return names


def deployable(val, hold, col, fam_of) -> dict:
    """Thresholds locked on the holdout, applied to val."""
    out = {}
    y_h, r_h = hold["label_final"].to_numpy(), hold["review_id"].to_numpy()
    s_h = hold[col].to_numpy()
    for method in ("equal_review", "equal_review_lcb"):
        for target in TARGETS:
            if method == "equal_review":
                tau = thresholds.equal_review(s_h, y_h, r_h, target)
            else:
                tau = thresholds.equal_review_lcb(
                    s_h, y_h, r_h, r_h_fam(hold, fam_of), target
                )
            per = per_review_at(val, col, tau)
            out[f"{method}@{target}"] = {
                "tau": tau,
                "val_sensitivity": per["sens"].mean(),
                "val_specificity": per["spec"].mean(),
                "val_j": per["sens"].mean() + per["spec"].mean() - 1,
                "val_reviews_below_target": int((per["sens"] < target).sum()),
            }
    return out


def r_h_fam(df, fam_of):
    return df["review_id"].map(fam_of).to_numpy()


def per_review_at(df, col, tau) -> pd.DataFrame:
    keep = df[col].to_numpy() >= tau
    d = pd.DataFrame(
        {"r": df["review_id"].to_numpy(), "y": df["label_final"], "k": keep}
    )
    sens = d[d.y == 1].groupby("r")["k"].mean()
    spec = 1 - d[d.y == 0].groupby("r")["k"].mean()
    return pd.DataFrame({"sens": sens, "spec": spec})


def paired_bootstrap(per_a, per_b, fam_of, reps=2000, seed=0) -> dict:
    """Family bootstrap of mean(b) - mean(a) over common reviews."""
    common = per_a.index.intersection(per_b.index)
    diff = (per_b.loc[common] - per_a.loc[common]).dropna()
    fams = diff.index.map(fam_of).to_numpy()
    uniq, inv = np.unique(fams, return_inverse=True)
    sums = np.bincount(inv, weights=diff.to_numpy())
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        len(uniq), np.full(len(uniq), 1 / len(uniq)), size=reps
    )
    boot = (draws @ sums) / np.maximum(draws @ counts, 1)
    return {
        "diff": float(diff.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "p_le_0": float((boot <= 0).mean()),
        "reviews": int(len(diff)),
    }


def v1(val, hold, systems, fam_of) -> dict:
    res, per = {}, {}
    for col in systems:
        p = ranking.per_review_ranking(val, col).set_index("review_id")
        per[col] = p
        res[col] = {
            "oracle": {c: float(p[c].dropna().mean()) for c in p.columns},
            "deployable": deployable(val, hold, col, fam_of),
        }
    pairs = [("titan_sr:mean_sigmoid", "titan_open:mean_sigmoid")]
    pairs += [("titan_sr:mean_sigmoid", "titan_open:mean_z")]
    res["paired_open_minus_sr"] = {}
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        key = f"{b} - {a}"
        res["paired_open_minus_sr"][key] = {
            m: paired_bootstrap(per[a][m], per[b][m], fam_of)
            for m in ("auc", "spec_at_99_recall", "spec_at_995_recall")
        }
    return res, per


def noise_flags(recs: pd.DataFrame) -> pd.Series:
    pts = recs["publication_types"].fillna("").str.split("|")
    by_pt = pts.apply(lambda ts: bool(NONPRIMARY_PT & set(ts)))
    by_title = recs["title"].fillna("").str.lower().str.contains(
        NONPRIMARY_TITLE, regex=True
    )
    # Primary studies typed Review only through "Review" + RCT etc. stay.
    primary_pt = pts.apply(
        lambda ts: bool(
            {"Randomized Controlled Trial", "Clinical Trial", "Case Reports",
             "Observational Study", "Comparative Study"} & set(ts)
        )
    )
    return (by_pt | by_title) & ~primary_pt


def v2(val, systems, flags) -> dict:
    out = {}
    pos = val[val.label_final == 1].copy()
    pos["noisy"] = pos["noisy"].astype(bool)
    rct_only = pos["review_id"].map(flags["rct_only"])
    out["val_positives"] = int(len(pos))
    out["noisy_positives"] = int(pos.noisy.sum())
    out["noisy_rate"] = float(pos.noisy.mean())
    out["noisy_rate_rct_only_reviews"] = float(pos.noisy[rct_only].mean())
    out["noisy_rate_by_source"] = (
        pos.groupby(pos.review_id.str.split(":").str[0])["noisy"]
        .mean()
        .round(4)
        .to_dict()
    )
    for col in systems:
        if not col.endswith("mean_sigmoid"):
            continue
        pct = pos.groupby("review_id")[col].rank(pct=True)
        low = pos.loc[pos.groupby("review_id")[col].idxmin()]
        clean = val[~val["noisy"].astype(bool) | (val.label_final == 0)]
        p_all = ranking.per_review_ranking(val, col).set_index("review_id")
        p_clean = ranking.per_review_ranking(clean, col).set_index(
            "review_id"
        )
        common = p_all.index.intersection(p_clean.index)
        out[col] = {
            "noisy_pos_mean_pct_rank": float(pct[pos.noisy].mean()),
            "clean_pos_mean_pct_rank": float(pct[~pos.noisy].mean()),
            "reviews_whose_lowest_positive_is_noisy": int(low.noisy.sum()),
            "reviews_with_positives": int(len(low)),
            "spec99_all_labels": float(
                p_all.loc[common, "spec_at_99_recall"].mean()
            ),
            "spec99_noisy_positives_dropped": float(
                p_clean.loc[common, "spec_at_99_recall"].mean()
            ),
            "noisy_in_rct_only_reviews": int((pos.noisy & rct_only).sum()),
        }
    return out


def v3(val, per, reviews, flags, out_dir) -> dict:
    col = "titan_open:mean_sigmoid"
    if col not in per:
        col = "titan_sr:mean_sigmoid"
    p = per[col][["auc", "spec_at_99_recall"]].copy()
    g = val.groupby("review_id")
    p["n_pos"] = g["label_final"].sum()
    p["n_records"] = g.size()
    p["prevalence"] = p["n_pos"] / p["n_records"]
    p["noisy_share"] = val[val.label_final == 1].groupby("review_id")[
        "noisy"
    ].mean()
    r = reviews.reindex(p.index)
    p["source"] = p.index.str.split(":").str[0]
    p["objective_chars"] = r["objective"].fillna("").str.len()
    p["criteria_chars"] = r["selection_criteria"].fillna("").str.len()
    p["n_criteria"] = r["criteria_list"].fillna("[]").apply(
        lambda s: len(json.loads(s))
    )
    tok = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
    )
    crit_tokens = [
        len(tok(t, add_special_tokens=False)["input_ids"])
        for t in r["selection_criteria"].fillna("")
    ]
    p["criteria_tokens"] = crit_tokens
    p["criteria_kept"] = np.minimum(1.0, 224 / np.maximum(p.criteria_tokens, 1))
    p["rct_only"] = flags.reindex(p.index)["rct_only"].to_numpy()
    p["review_title"] = r["title"].fillna("").str[:90]
    p.to_csv(f"{out_dir}/v3_per_review.csv")
    num = [
        "n_pos", "n_records", "prevalence", "noisy_share", "objective_chars",
        "criteria_chars", "n_criteria", "criteria_tokens", "criteria_kept",
    ]
    spear = {
        c: float(p["spec_at_99_recall"].corr(p[c], method="spearman"))
        for c in num
    }
    by_source = (
        p.groupby("source")[["auc", "spec_at_99_recall"]]
        .agg(["mean", "count"])
        .round(4)
    )
    by_source.columns = ["_".join(c) for c in by_source.columns]
    quart = pd.qcut(p["n_pos"], 4, duplicates="drop")
    return {
        "system": col,
        "spearman_with_spec99": {k: round(v, 3) for k, v in spear.items()},
        "by_source": by_source.to_dict(orient="index"),
        "spec99_by_n_pos_quartile": p.groupby(quart, observed=True)[
            "spec_at_99_recall"
        ]
        .mean()
        .round(4)
        .rename(index=str)
        .to_dict(),
        "spec99_rct_only": float(p.loc[p.rct_only, "spec_at_99_recall"].mean()),
        "spec99_other": float(p.loc[~p.rct_only, "spec_at_99_recall"].mean()),
        "worst_10": p.nsmallest(10, "spec_at_99_recall")[
            ["source", "n_pos", "n_records", "spec_at_99_recall", "auc",
             "criteria_tokens", "noisy_share", "review_title"]
        ]
        .round(3)
        .reset_index()
        .to_dict(orient="records"),
    }


def v4(val, hold, bienc_dir, fam_of) -> dict:
    out = {}
    feats = ["obj_doc_z", "mean_sent_z", "min_sent_z"]
    for name, df in (("val", val), ("holdout", hold)):
        b = pd.read_parquet(f"{bienc_dir}/{name}_scores.parquet")
        b = b[KEYS + feats].drop_duplicates(KEYS)
        merged = df.merge(b, on=KEYS, how="left")
        for f in feats:
            merged[f] = merged[f].fillna(0.0)
        if name == "val":
            val = merged
        else:
            hold = merged
    base = "titan_open:mean_z"
    if base not in val:
        base = "titan_sr:mean_z"
    for f in feats:
        best_w, best = 0.0, -1.0
        for w in np.arange(0, 1.01, 0.05):
            hold["_s"] = hold[base] + w * hold[f]
            m = ranking.summarize_ranking(hold, "_s")["spec_at_99_recall"]
            if m > best:
                best_w, best = float(w), m
        val["_s"] = val[base] + best_w * val[f]
        base_per = ranking.per_review_ranking(val, base).set_index(
            "review_id"
        )
        new_per = ranking.per_review_ranking(val, "_s").set_index("review_id")
        out[f] = {
            "weight_from_holdout": best_w,
            "holdout_spec99": best,
            "val": {
                m: paired_bootstrap(base_per[m], new_per[m], fam_of)
                for m in ("auc", "spec_at_99_recall")
            },
        }
    out["base"] = base
    return out


def _recentred(df: pd.DataFrame, members: list) -> dict:
    """Score variants that use only the review's unlabelled scores."""
    logit = df[members].mean(axis=1)
    g = logit.groupby(df["review_id"])
    top3 = g.transform(lambda x: x.nlargest(min(3, len(x))).mean())
    out = {
        "mean_logit": logit,
        "minus_max": logit - g.transform("max"),
        "minus_top3": logit - top3,
        "minus_median": logit - g.transform("median"),
        "minus_q75": logit - g.transform(lambda x: x.quantile(0.75)),
        "minus_q90": logit - g.transform(lambda x: x.quantile(0.9)),
    }
    for lam in (0.25, 0.5, 0.75):
        out[f"minus_{lam}max"] = logit - lam * g.transform("max")
        out[f"minus_{lam}top3"] = logit - lam * top3
    return out


def v6(val, hold, members, fam_of) -> dict:
    res = {}
    hv, vv = _recentred(hold, members), _recentred(val, members)
    for name in hv:
        hold["_s"], val["_s"] = hv[name].to_numpy(), vv[name].to_numpy()
        row = deployable(val, hold, "_s", fam_of)
        tau = row["equal_review@0.99"]["tau"]
        h = per_review_at(hold, "_s", tau)
        row["holdout_spec_at_eq99"] = float(h["spec"].mean())
        res[name] = row
    best = max(res, key=lambda k: res[k]["holdout_spec_at_eq99"])
    res["chosen_on_holdout"] = best
    return res


def v5(data_root: str) -> dict:
    recs = pd.read_parquet(
        f"{data_root}/processed/records.parquet",
        columns=["review_id", "pmid", "publication_types"],
    )
    recs["source"] = recs["review_id"].str.split(":").str[0]
    recs["has_pt"] = recs["publication_types"].fillna("").str.len() > 0
    missing = recs.loc[~recs.has_pt & recs.pmid.notna(), "pmid"].unique()
    con = sqlite3.connect(f"{data_root}/cache/pubmed.sqlite")
    fillable = 0
    for i in range(0, len(missing), 900):
        batch = [str(p) for p in missing[i : i + 900]]
        marks = ",".join("?" * len(batch))
        for (payload,) in con.execute(
            f"SELECT payload FROM citations WHERE pmid IN ({marks})", batch
        ):
            if json.loads(payload).get("publication_types"):
                fillable += 1
    return {
        "has_publication_types_by_source": recs.groupby("source")["has_pt"]
        .mean()
        .round(4)
        .to_dict(),
        "has_pmid_by_source": recs.groupby("source")["pmid"]
        .apply(lambda s: float(s.notna().mean()))
        .round(4)
        .to_dict(),
        "missing_pt_unique_pmids": int(len(missing)),
        "backfillable_from_cache": fillable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--biencoder")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    splits = pd.read_parquet(f"{args.data_root}/splits/splits.parquet")
    fam_of = splits.drop_duplicates("review_id").set_index("review_id")[
        "family_id"
    ]
    reviews = pd.read_parquet(
        f"{args.data_root}/processed/reviews.parquet",
        columns=[
            "review_id", "title", "objective", "selection_criteria",
            "criteria_list",
        ],
    ).set_index("review_id")
    flags = _review_flags(reviews)
    val, have = load_members(args.runs_root, "preds.parquet")
    hold, have_h = load_members(args.runs_root, "holdout_preds.parquet")
    assert have == have_h, (have, have_h)
    systems = add_systems(val, have)
    add_systems(hold, have)
    meta = pd.read_parquet(
        f"{args.data_root}/splits/records.parquet",
        columns=KEYS + ["title", "publication_types"],
        filters=[("review_id", "in", list(val.review_id.unique()))],
    ).drop_duplicates(KEYS)
    meta["noisy"] = noise_flags(meta)
    val = val.merge(meta[KEYS + ["noisy"]], on=KEYS, how="left")
    val["noisy"] = val["noisy"].fillna(False).astype(bool)

    result = {"members": have, "val_reviews": int(val.review_id.nunique())}
    result["V1"], per = v1(val, hold, systems, fam_of)
    print("V1 done", flush=True)
    result["V2"] = v2(val, systems, flags)
    print("V2 done", flush=True)
    result["V3"] = v3(val, per, reviews, flags, args.out)
    print("V3 done", flush=True)
    if args.biencoder:
        result["V4"] = v4(val, hold, args.biencoder, fam_of)
        print("V4 done", flush=True)
    for name in ("titan_open", "titan_sr"):
        members = [b for b in (OPEN if name == "titan_open" else SR) if b in have]
        result[f"V6:{name}"] = v6(val, hold, members, fam_of)
    print("V6 done", flush=True)
    result["V5"] = v5(args.data_root)
    with open(f"{args.out}/results.json", "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
