# Killarney experiment plan (runs in parallel with the local 3080 Ti queue)

Written 2026-10-31 for the agent operating on Killarney. Read
[CLAUDE.md](../CLAUDE.md), [KILLARNEY.md](KILLARNEY.md) (setup, quick start)
and [EXPERIMENT_IDEAS.md](EXPERIMENT_IDEAS.md) first. Branch:
`claude/practical-lovelace-7oe49s`.

## Rules that override everything below

1. **titan_test is evaluated exactly once**, in K1, by
   `slurm/pilot_l40s.sh --finalize`, after `slurm/preregister.sh` has
   committed the locked thresholds (the evaluate gate enforces this). No
   other experiment may predict on, evaluate on or look at titan_test.
   Smoke and debug runs use val and the train-fold holdout only.
2. Every choice (members, recipes, rules) rests on code bugs or
   train / holdout / val evidence and is committed before it is used.
   Screens follow the pattern of [BACKBONE_SCREEN.md](BACKBONE_SCREEN.md):
   rule committed first, then the runs, then the outcome.
3. Never weaken the family graph, decontamination or the leakage audit.
4. Never commit data, checkpoints, keys or emails. The W&B key lives in
   `~/.config/opensrma/wandb.env` (mode 600) on each machine.
5. Two agents share this branch. The Killarney agent commits on its own
   branch `killarney/<topic>` and merges through a pull request; the local
   agent does the same. Do not edit `scripts/v2_queue.sh` or anything the
   local queue is running.
6. Report every result, including failures, in this file's "Results"
   section (one line per run, W&B link) and on the ntfy topic the user
   gives you.

## Data transfer (no cloud bucket needed)

Everything is small: pilot_v2 data ~5.5 GB including the PubMed cache;
protocol raw downloads ~3 GB and growing (Europe PMC harvest). Move it
over SSH from the workstation (`/mnt/FNX/dav_Projects_2026/opensrma`) to
Killarney scratch, using one multiplexed connection so MFA is answered
once:

```bash
# on the workstation
ssh -o ControlMaster=yes -o ControlPath=~/.ssh/cm-%r@%h -o ControlPersist=4h \
    -fN <user>@killarney.alliancecan.ca
SRC=/mnt/FNX/dav_Projects_2026/opensrma
DST=<user>@killarney.alliancecan.ca:/scratch/<user>/opensrbench
R="ssh -o ControlPath=~/.ssh/cm-%r@%h"
rsync -aP -e "$R" --copy-links $SRC/data_pilot_v2/ $DST/killarney_pilot/datasets/
rsync -aP -e "$R" $SRC/shared_cache/ $DST/shared_cache/   # PubMed cache
rsync -aP -e "$R" $SRC/data_protocol/raw/ $DST/datasets/raw/  # when done
```

Notes: `--copy-links` because `data_pilot_v2/cache` is a symlink to
`shared_cache`. If Killarney's Globus endpoint is available, Globus is the
better tool for anything above ~50 GB (resumable, no open terminal); at
today's sizes rsync is simpler. Code goes through GitHub only (the user
pushes the branch). Copying the frozen pilot_v2 data (rather than
rebuilding it) keeps the Killarney pilot on byte-identical splits and
tiers to the local runs; check `frozen/titan_test/manifest.json` hashes
match after copying.

## Experiments (priority order)

GPU-hours are rough estimates from the 3080 Ti measurements (L40S ~3x,
H100 ~7x); re-measure in K0.

### K0. Bring-up and verification (first, ~2 GPU-h)

* `setup_env.sh`, `prefetch_models.sh` (add
  `thomas-sounack/BioClinical-ModernBERT-large`,
  `NeuML/pubmedbert-base-embeddings`, `ncbi/MedCPT-*` if missing),
  `sbatch slurm/smoke.sbatch`.
* **DDP on real GPUs** (only CPU/gloo tested so far): 20-step runs of
  every titan_open member and both titan_sr members on 2 and 4 L40S
  (`torchrun`, `--max-steps 20 --limit-reviews 50`, val/holdout only);
  check the logged global batch equals the 1-GPU value and nothing crashes.
* 200-step throughput / memory probe per backbone on L40S and H100;
  replace the estimates in KILLARNEY.md section 4 with measurements.
* Start `scripts/wandb_sync.py` on the login node over
  `$OSB_STORAGE/*/runs` (compute nodes are offline; the sidecar reads log
  files only).

### K1. L40S pilot head-to-head on titan_test (~60-90 GPU-h, 1 GPU per task)

The one titan_test evaluation of this round.

* Data: the copied pilot_v2 data, so `SKIP_ONLINE=1 SKIP_OFFLINE=1`.
* Members: titan_sr_baseline = titan_sr_pubmedbert + titan_sr_biolinkbert;
  titan_open = the screen outcome in BACKBONE_SCREEN.md. Addendum A
  (BioClinical-ModernBERT at 512 / 1024 tokens) is being screened locally
  and decides around 16:00 on 2026-10-31: `git pull` and read the outcome
  before submitting the training array. If it is not decided yet, submit
  the certain members first (titan_sr x2, biolinkbert_large,
  biomedbert_large) with `OPEN_MEMBERS`, and the third member once decided;
  `--finalize` must only run when all members exist.
* `bash slurm/pilot_l40s.sh --dry-run`, then the real run. Thresholds:
  `equal_review_lcb` (primary) + `pooled_conformal` (secondary), locked on
  val, committed by `preregister.sh` before evaluate.
* Report the leaderboard, paired family-bootstrap dJ with Holm
  correction, achieved vs. target sensitivity, and the local pilot result
  alongside (it was the first titan_test look; disclose it).

### K2. Qwen3-Reranker-4B LoRA screen (idea I4; ~40-60 H100-h)

Long-context decoder member (2048 tokens: full criteria and abstract).

* Commit a screen addendum to BACKBONE_SCREEN.md first: same holdout
  metric, same 0.05 rule against the best encoder mean, budget matched by
  wall-clock rather than steps (state it).
* Two seeds on fold 0 holdout; if it passes, it joins `titan_open_xl`
  (not titan_open) for a later K1-style run.

### K3. LLM-structured PICOS criteria (idea I1; ~10-20 H100-h)

* vLLM + `openai/gpt-oss-20b` on one H100 over every review's criteria
  (criteria are inputs, not labels; all splits may be processed). Output
  per review: atomic items with role (P, I, C, O, S, exclusion) and the
  **verbatim source span** for each; reject items whose span is not found
  in the source text. Cache as parquet keyed by review_id + prompt hash;
  the prompt is committed verbatim.
* Also extract allowed / excluded study designs per review (input to K5).
* Measure on val: span-faithfulness rate, items per review, tokens per
  review vs. the regex splitter.
* Then (needs a small code change, PR first): an `inputs` option to use the
  LLM items instead of `split_criteria`; one screen arm per seed on the
  best encoder (holdout AUC, 0.05-style rule committed first).

### K4. Bi-encoder PICO veto with LLM items (idea B2; CPU, ~2 h)

`scripts/pico_biencoder_probe.py` with the K3 items instead of the regex
fragments (MedCPT and NeuML/pubmedbert-base-embeddings). Compare with the
local B1 run (results in EXPERIMENT_IDEAS.md once done). Val only.

### K5. Review-conditional design veto (ideas R1-R4; CPU, ~2 h)

With K3's allowed-design extraction: compute per-rule inclusion rates on
train with 95% upper bounds (publication types + phrase rules + any
LLM-proposed regexes), keep rules with upper bound <= 0.1%, confirm on val,
and report how many val negatives they remove and whether those negatives
are among the high-scoring ones (use val predictions from K1's members,
never titan_test). Commit the rule set before any use.

### K6. Verifier cascade dry run (idea C5; ~10 H100-h)

`slurm/llm_verifier.sbatch` on **val** only with K1's members: gray-zone
size, verifier exclusions, sensitivity cost. No titan_test.

### Not yet (blocked)

Full-protocol `submit_all.sh` (all datasets, 5 folds): blocked on the
records-table memory fix and the 24 h offline-build fix (see the
readiness plan), and on the family-graph decision (largest family 7,252 of
16,993 reviews on pilot_v2). Do not submit it.

## Results

| Date | Exp | Run | Result | W&B / log |
|---|---|---|---|---|
| | | | | |
