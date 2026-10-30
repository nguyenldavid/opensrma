"""`osb llm-sweep`: budgeted async LLM screening on a frozen tier.

For every enabled model x condition the runner screens all records of tier
T_k (default T20), caches each raw response in JSONL (resumable, keyed by
review/record/seed), applies the parser, and writes a standard decision
file to <runs>/systems/<model>__<condition>.parquet. A seed-1 repeat on the
stability subset gives repeat agreement and Cohen's kappa.
"""

import asyncio
import json
import logging
import os
from typing import Any

import pandas as pd

from opensrbench import config
from opensrbench import pipeline
from opensrbench.evaluation import agreement
from opensrbench.llm import client as client_lib
from opensrbench.llm import prompts
from opensrbench.utils import hashing
from opensrbench.utils import io


def _cache_key(review_id: str, record_id: str, seed: int) -> str:
    return hashing.short_id(f"{review_id}|{record_id}|{seed}", 24)


def _read_cache(path: str) -> dict[str, dict]:
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    out[row["key"]] = row
                except (json.JSONDecodeError, KeyError):
                    continue
    return out


async def _screen(
    rows: list[dict],
    model_cfg: dict,
    cond_cfg: dict,
    template: str,
    parser,
    endpoint: dict,
    run_cfg: dict,
    budget: client_lib.Budget,
    cache_path: str,
    seed: int,
) -> list[dict]:
    cache = _read_cache(cache_path)
    client = client_lib.ChatClient(
        endpoint["base_url"],
        endpoint.get("api_key_env"),
        float(run_cfg.get("timeout_seconds", 300)),
        int(run_cfg.get("max_retries", 6)),
    )
    price = model_cfg.get("price_per_mtok", {"input": 0.0, "output": 0.0})
    max_tokens = int(cond_cfg.get("max_tokens", 256))
    sem = asyncio.Semaphore(int(run_cfg.get("max_concurrency", 16)))
    lock = asyncio.Lock()
    extra = model_cfg.get("conditions", {}).get(cond_cfg["name"], {})
    exhausted = False

    async def one(row: dict) -> dict:
        nonlocal exhausted
        key = _cache_key(row["review_id"], row["record_id"], seed)
        if key in cache:
            return cache[key]
        prompt = prompts.render(
            template,
            row["objective"],
            row["selection_criteria"],
            row["title"],
            row["abstract"],
        )
        worst = (
            len(prompt) / 3.0 * price["input"] + max_tokens * price["output"]
        ) / 1e6
        async with sem:
            try:
                await budget.reserve(worst)
            except client_lib.BudgetExceededError as exc:
                exhausted = True
                return {
                    "key": key,
                    "review_id": row["review_id"],
                    "record_id": row["record_id"],
                    "text": None,
                    "error": str(exc),
                    "cost_usd": 0.0,
                    "cached": False,
                }
            result = await client.chat(
                model_cfg["model"],
                prompt,
                int(extra.get("max_tokens", max_tokens)),
                float(extra.get("temperature", 0.0)),
                seed,
                extra.get("extra_body"),
            )
            cost = (
                result.prompt_tokens * price["input"]
                + result.completion_tokens * price["output"]
            ) / 1e6
            await budget.settle(worst, cost)
        entry = {
            "key": key,
            "review_id": row["review_id"],
            "record_id": row["record_id"],
            "text": result.text,
            "error": result.error,
            "cost_usd": cost,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.raw_finish_reason,
        }
        if result.error is None:  # Errors are retried on the next run.
            async with lock:
                with open(cache_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
        return entry

    try:
        results = await asyncio.gather(*(one(r) for r in rows))
    finally:
        await client.close()
    if exhausted:
        logging.error(
            "Budget exhausted: some records were not screened and "
            "will be scored as keep/invalid. Raise budget_usd and "
            "re-run; cached responses are reused."
        )
    out = []
    for entry in results:
        keep, valid = parser(entry.get("text"))
        out.append(
            {
                **entry,
                "decision_keep": keep,
                "is_valid_answer": valid and entry.get("error") is None,
            }
        )
    return out


def run(args: Any) -> dict:
    bench_cfg = config.Config.load(args.config)
    run_cfg = config.Config.load(args.llm_config).as_dict()
    paths = config.Paths.from_config(bench_cfg)
    runs_root = args.runs_dir or paths.runs
    benchmark = args.benchmark or run_cfg.get("benchmark", "open_sr_bench")
    tier = int(args.tier or run_cfg.get("tier", 20))
    tiers = io.read_parquet(
        os.path.join(paths.frozen, benchmark, "tiers.parquet")
    )
    tier_rows = tiers[tiers[f"in_tier_t{tier}"]]
    reviews, records, _ = pipeline.load_split_data(paths)
    rows = tier_rows.merge(
        records[["review_id", "record_id", "title", "abstract"]],
        on=["review_id", "record_id"],
    ).merge(
        reviews[["review_id", "objective", "selection_criteria"]],
        on="review_id",
    )
    if args.limit:
        rows = rows.head(args.limit)
    budget = client_lib.Budget(float(run_cfg.get("budget_usd", 0)))
    models = {
        k: v
        for k, v in run_cfg["models"].items()
        if v.get("enabled", True) and (not args.models or k in args.models)
    }
    conditions = args.conditions or list(run_cfg["conditions"])
    summary = {}
    for model_name, model_cfg in models.items():
        endpoint = run_cfg["endpoints"][model_cfg["endpoint"]]
        for cond in conditions:
            if cond not in model_cfg.get("conditions", {}):
                continue
            cond_cfg = {**run_cfg["conditions"][cond], "name": cond}
            try:
                template = prompts.load_template(cond_cfg["template"])
            except prompts.PromptNotConfiguredError as exc:
                logging.warning("skipping %s/%s: %s", model_name, cond, exc)
                continue
            parser = prompts.PARSERS[cond_cfg.get("parser", "tag")]
            out_dir = io.ensure_dir(
                os.path.join(runs_root, "llm", model_name, cond)
            )
            logging.info(
                "screening %d records with %s/%s", len(rows), model_name, cond
            )
            results = asyncio.run(
                _screen(
                    rows.to_dict("records"),
                    model_cfg,
                    cond_cfg,
                    template,
                    parser,
                    endpoint,
                    run_cfg,
                    budget,
                    os.path.join(out_dir, "responses_seed0.jsonl"),
                    seed=0,
                )
            )
            system = f"{model_name}/{cond}"
            decisions = pd.DataFrame(results)[
                [
                    "review_id",
                    "record_id",
                    "decision_keep",
                    "is_valid_answer",
                    "cost_usd",
                ]
            ].assign(system=system, set=benchmark)
            io.write_parquet(
                decisions,
                os.path.join(
                    runs_root, "systems", f"{model_name}__{cond}.parquet"
                ),
            )
            entry = {
                "records": len(decisions),
                "invalid_rate": float(1 - decisions["is_valid_answer"].mean()),
                "cost_usd": float(decisions["cost_usd"].sum()),
            }
            if run_cfg.get("stability", True):
                subset = rows[rows["in_stability_subset"]]
                repeat = asyncio.run(
                    _screen(
                        subset.to_dict("records"),
                        model_cfg,
                        cond_cfg,
                        template,
                        parser,
                        endpoint,
                        run_cfg,
                        budget,
                        os.path.join(out_dir, "responses_seed1.jsonl"),
                        seed=1,
                    )
                )
                entry["repeat"] = agreement.repeat_agreement(
                    decisions,
                    pd.DataFrame(repeat)[
                        ["review_id", "record_id", "decision_keep"]
                    ],
                )
            summary[system] = entry
            io.write_json(entry, os.path.join(out_dir, "summary.json"))
    summary["_budget"] = {"limit": budget.limit, "spent": budget.spent}
    io.write_json(summary, os.path.join(runs_root, "llm", "summary.json"))
    return summary
