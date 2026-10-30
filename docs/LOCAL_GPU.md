# Running on a single workstation GPU (RTX 3080 Ti, 12 GB)

Everything that runs on Killarney also runs on one consumer GPU. The
**training recipe is identical**: hardware profiles only change micro-batch
size, gradient checkpointing and worker counts, and gradient accumulation
keeps every backbone's effective batch size, so a local model is directly
comparable with a cluster-trained one.

| Profile (`--hardware` / `$OSB_HARDWARE`) | Typical GPUs |
|---|---|
| `rtx3080ti` | RTX 3080 Ti / 3080 12 GB / 4070 Ti / A2000 12 GB |
| `gpu_24gb` | RTX 3090 / 4090 / A5000 / L4 |
| `killarney_l40s`, `killarney_h100` | Killarney nodes |
| `cpu` | tests, smoke runs |
| `auto` (default) | picked from the detected GPU's name and memory |

## Requirements

* Linux or WSL2 (Ubuntu 22.04+), NVIDIA driver ≥ 535, Python ≥ 3.11.
* 12 GB VRAM (Ampere or newer for bf16), **≥ 32 GB RAM** (64 GB
  recommended for the full corpus), ~300–500 GB free disk.
* An NCBI API key (free) for the data stages.

## Setup

```bash
git clone https://github.com/nguyenldavid/opensrma && cd opensrma
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[all]"
```

```bash
pip install flash-attn --no-build-isolation   # optional; falls back to SDPA
```

```bash
export NCBI_EMAIL=you@lab.org NCBI_API_KEY=xxxx
export OSB_HARDWARE=rtx3080ti
osb doctor --network
osb smoke                      # ~1 min synthetic end-to-end check
python -m pytest -q            # unit tests
```

## One command

```bash
bash scripts/run_local.sh
```

Runs, resumably: data (network + offline build with MedCPT on the GPU,
leakage audit) → RCT classifier → training of every run of
`titan_open_lite` and `titan_sr_baseline` → whole-cohort prediction →
calibration (with and without RCT features) → development leaderboard on
`val`. Re-run the same command after any interruption.

Useful variants:

```bash
STEPS="train calibrate dev" bash scripts/run_local.sh
```

```bash
ENSEMBLES="titan_open_local titan_sr_baseline" bash scripts/run_local.sh
```

```bash
CONFIG=configs/benchmark_v1.yaml bash scripts/run_local.sh    # full-size data
```

`configs/benchmark_local.yaml` inherits the benchmark definition unchanged
(dates, tiers, salts, splits) and only shrinks the most expensive data knobs
(300 retrieved negatives per augmented legacy review instead of 1,000,
≤ 8,000 augmented reviews, smaller RCT corpus).

## Which ensemble on 12 GB?

| Ensemble | Members | Runs | Fits 12 GB | Notes |
|---|---|---:|:-:|---|
| `titan_open_lite` | ModernBERT-base, BioLinkBERT-base, PubMedBERT-base | 9 | ✓ | default locally; same backbones as the TITAN-SR baseline, so the comparison isolates the method |
| `titan_open_local` | ModernBERT-large, BioLinkBERT-large, DeBERTa-v3-large, Qwen3-Reranker-0.6B (LoRA) | 20 | ✓ (gradient checkpointing, micro-batch 4) | best quality locally; several times slower |
| `titan_open` | the three large encoders | 15 | ✓ | the protocol ensemble |
| `titan_open_xl` | + Qwen3-Reranker-4B | — | ✗ | the profile refuses it (needs ≥ 24 GB) |
| `titan_sr_baseline` | TITAN-SR recipe, PubMedBERT-base + BioLinkBERT-base | 2 | ✓ | reference |

Measure before you plan: the training log prints `seq_per_s` every 50 steps.

```bash
osb train --backbone pubmedbert_base --fold 0 --max-steps 100 --limit-reviews 200
```

```bash
tail -n 3 runs/pubmedbert_base/fold0/train_log.jsonl
```

Time per run ≈ `optim.epochs × pairs_per_epoch / seq_per_s`, where
pairs_per_epoch ≈ (1 + `negatives_per_positive`) × capped positives (printed
at start-up as "train N rows"). Lower `sampler.max_pos_per_review` or
`optim.epochs` with `--override` if needed.

## Out-of-memory safety

* `auto_batch: true` (default) probes a full-length synthetic batch — with a
  buffer the size of the AdamW states — before training and halves the
  micro-batch until it fits; accumulation is recomputed so the effective
  batch is unchanged. If even micro-batch 1 fails the run stops with a clear
  message.
* If the desktop/compositor uses VRAM, close other GPU apps or set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* Prediction uses the profile's `eval_batch_size`; override with
  `osb predict ... --batch-size 16`.
* Note: on micro-batches smaller than a review group (16), groups shrink to
  the micro-batch with the same positive:negative mix, so the in-review
  ranking loss sees fewer negatives per step than on Killarney. Everything
  else is identical.

## Held-out results

Development numbers come from `val`. Held-out benchmarks
(`internal_test`, `open_sr_bench`, `titan_test`, `prospective`) are blinded:

```bash
osb preregister
git add datasets/manifests && git commit -m "Pre-register thresholds"
osb evaluate --benchmark internal_test --unblind
```

Every `--unblind` call is appended to `runs/unblinding_log.jsonl`.
