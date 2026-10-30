"""`osb` command-line interface.

  osb doctor                         Check the environment (GPU, deps, keys).
  osb data --stages online           download,ingest,enrich,pools  [network]
  osb data --stages offline          build,split,freeze            [offline]
  osb plan-runs --ensemble titan_open
  osb train --backbone modernbert_large --fold 0      (under torchrun)
  osb predict --backbone modernbert_large --fold 0    (under torchrun)
  osb calibrate --ensemble titan_open
  osb llm-sweep
  osb cascade --ensemble titan_open
  osb evaluate --benchmark open_sr_bench
  osb smoke                          Synthetic end-to-end run on CPU.

Run `osb <command> --help` for options. Configs default to ./configs.
"""

import argparse
import json
import sys

from opensrbench import config
from opensrbench.utils import io

_STAGE_GROUPS = {
    "online": ["download", "ingest", "enrich", "pools"],
    "offline": ["build", "split", "freeze"],
    "all": [
        "download",
        "ingest",
        "enrich",
        "pools",
        "build",
        "split",
        "freeze",
    ],
}


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=config.DEFAULT_BENCHMARK_CONFIG,
        help="Benchmark config (default: %(default)s).",
    )
    parser.add_argument(
        "--ensemble-config",
        default=config.DEFAULT_ENSEMBLE_CONFIG,
        help="Ensemble config (default: %(default)s).",
    )
    parser.add_argument(
        "--runs-dir", default=None, help="Override paths.runs_root."
    )
    parser.add_argument("--log-level", default="INFO")


def _cmd_data(args) -> None:
    from opensrbench import pipeline  # pylint: disable=import-outside-toplevel

    stages = []
    for part in args.stages.split(","):
        stages.extend(_STAGE_GROUPS.get(part.strip(), [part.strip()]))
    ctx = pipeline.Context.create(
        args.config,
        args.datasets_config,
        tuple(args.only.split(",")) if args.only else (),
        args.device,
    )
    pipeline.run(ctx, stages)


def _cmd_plan_runs(args) -> None:
    from opensrbench.modeling import \
        spec  # pylint: disable=import-outside-toplevel

    runs = spec.plan_runs(
        config.Config.load(args.ensemble_config), args.ensemble
    )
    if args.index is not None:
        if not 0 <= args.index < len(runs):
            raise SystemExit(
                f"index {args.index} out of range [0, {len(runs)})"
            )
        backbone, fold = runs[args.index]
        print(f"{backbone} {fold}")
        return
    for backbone, fold in runs:
        print(f"{backbone} {fold}")


def _cmd_train(args) -> None:
    from opensrbench.modeling import \
        train  # pylint: disable=import-outside-toplevel

    args.output_dir = args.runs_dir
    print(json.dumps(train.run(args), indent=2))


def _cmd_predict(args) -> None:
    from opensrbench.modeling import \
        predict  # pylint: disable=import-outside-toplevel

    args.output_dir = args.runs_dir
    args.sets = args.sets.split(",")
    print(predict.run(args))


def _cmd_calibrate(args) -> None:
    from opensrbench.calibration import \
        run  # pylint: disable=import-outside-toplevel

    print(json.dumps(run.run(args)["thresholds"], indent=2))


def _cmd_llm(args) -> None:
    from opensrbench.llm import \
        runner  # pylint: disable=import-outside-toplevel

    args.models = args.models.split(",") if args.models else None
    args.conditions = args.conditions.split(",") if args.conditions else None
    print(json.dumps(runner.run(args), indent=2, default=str))


def _cmd_cascade(args) -> None:
    from opensrbench import cascade  # pylint: disable=import-outside-toplevel

    print(cascade.run(args))


def _cmd_evaluate(args) -> None:
    from opensrbench.evaluation import \
        report  # pylint: disable=import-outside-toplevel

    result = report.run(args)
    print(report.markdown_table(result))


def _cmd_doctor(args) -> None:
    from opensrbench import doctor  # pylint: disable=import-outside-toplevel

    sys.exit(doctor.run(args))


def _cmd_smoke(args) -> None:
    from opensrbench import smoke  # pylint: disable=import-outside-toplevel

    smoke.run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osb",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="Run data-pipeline stages.")
    _common(p)
    p.add_argument("--datasets-config", default=config.DEFAULT_DATASETS_CONFIG)
    p.add_argument(
        "--stages",
        default="offline",
        help="Comma list of stages or groups online/offline/all.",
    )
    p.add_argument(
        "--only", default="", help="Restrict to these sources (comma list)."
    )
    p.add_argument(
        "--device",
        default=None,
        help="cuda / cpu / none (none disables MedCPT).",
    )
    p.set_defaults(fn=_cmd_data)

    p = sub.add_parser("plan-runs", help="List (backbone, fold) runs.")
    _common(p)
    p.add_argument("--ensemble", default="titan_open")
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Print only run #index (Slurm array task id).",
    )
    p.set_defaults(fn=_cmd_plan_runs)

    p = sub.add_parser("train", help="Train one backbone fold (torchrun).")
    _common(p)
    p.add_argument("--backbone", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--limit-reviews", type=int, default=0)
    p.add_argument(
        "--override",
        action="append",
        help="Spec override, e.g. optim.lr=1e-5 (repeatable).",
    )
    p.set_defaults(fn=_cmd_train)

    p = sub.add_parser("predict", help="Score review sets with a checkpoint.")
    _common(p)
    p.add_argument("--backbone", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--sets", default="val,open_sr_bench,titan_test,prospective")
    p.add_argument("--batch-size", type=int, default=128)
    p.set_defaults(fn=_cmd_predict)

    p = sub.add_parser(
        "calibrate", help="Stack, lock thresholds, emit systems."
    )
    _common(p)
    p.add_argument("--ensemble", default="titan_open")
    p.add_argument(
        "--method",
        default=None,
        choices=[None, "pooled_conformal", "equal_review", "equal_review_lcb"],
    )
    p.set_defaults(fn=_cmd_calibrate)

    p = sub.add_parser("llm-sweep", help="Budgeted LLM screening sweep.")
    _common(p)
    p.add_argument("--llm-config", default="configs/llm_models.yaml")
    p.add_argument("--models", default="")
    p.add_argument("--conditions", default="")
    p.add_argument("--benchmark", default=None)
    p.add_argument("--tier", type=int, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=_cmd_llm)

    p = sub.add_parser("cascade", help="Two-stage encoder -> LLM cascade.")
    _common(p)
    p.add_argument("--ensemble", default="titan_open")
    p.add_argument("--benchmark", default="open_sr_bench")
    p.add_argument("--verifier", default=None)
    p.add_argument("--low", default=None)
    p.add_argument("--high", default=None)
    p.set_defaults(fn=_cmd_cascade)

    p = sub.add_parser("evaluate", help="Leaderboard with bootstrap CIs.")
    _common(p)
    p.add_argument("--benchmark", default="open_sr_bench")
    p.add_argument("--tier", type=int, default=None)
    p.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Decision-file globs (default runs/systems/*).",
    )
    p.add_argument("--reference", default=None)
    p.add_argument("--replicates", type=int, default=None)
    p.set_defaults(fn=_cmd_evaluate)

    p = sub.add_parser("doctor", help="Check environment readiness.")
    _common(p)
    p.add_argument(
        "--network",
        action="store_true",
        help="Also test NCBI / Europe PMC / HF reachability.",
    )
    p.set_defaults(fn=_cmd_doctor)

    p = sub.add_parser("smoke", help="Synthetic end-to-end run (CPU).")
    _common(p)
    p.add_argument("--workdir", default="smoke_run")
    p.set_defaults(fn=_cmd_smoke)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    io.setup_logging(args.log_level)
    args.fn(args)


if __name__ == "__main__":
    main()
