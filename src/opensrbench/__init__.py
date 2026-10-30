"""Open-SR-Bench: leak-audited systematic-review screening benchmark.

Subpackages:
  sources:      adapters that turn existing open screening datasets into the
                canonical review/record schema.
  pubmed:       NCBI E-utilities client, PubMed XML parser and SQLite store.
  harvest:      Europe PMC JATS harvester for post-cutoff systematic reviews.
  cohort:       candidate-pool reconstruction (Boolean + BM25 + MedCPT).
  splits:       family graph, split assignment, fingerprint decontamination
                and frozen T10/T20/T40 tiers.
  modeling:     TITAN-Open decomposed PICOS soft-min cross-encoder.
  calibration:  review-conditional stacking + conformal threshold locking.
  llm:          prompt contracts and the async budgeted LLM sweep.
  evaluation:   IPW equal-review estimators and family cluster bootstrap.
"""

__version__ = "1.1.0"
BENCHMARK_SALT = "OPEN-SR-BENCH-V1"
