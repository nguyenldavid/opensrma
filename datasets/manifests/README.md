# Manifests

Small, versioned artefacts that pin a benchmark release (commit them):

* `open_sr_bench_v1.manifest.json` — copy of
  `frozen/open_sr_bench/manifest.json` at freeze time (tier counts, SHA-256
  content hashes and the zero-leakage audit). Anyone rebuilding the
  benchmark must reproduce these hashes exactly.
* `locked_operating_points.<ensemble>.json` — copy of
  `runs/calibration/<ensemble>/locked_operating_points.json`, committed
  **before** the held-out benchmark is unblinded (pre-registration).
