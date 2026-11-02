"""Command-line entry point.

    python run.py configs/brown.yaml
"""

import argparse

from lm.controller import run_from_config
from lm.tokenization import tokenizer_names


def percentage(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 100") from exc
    if not 0 < parsed <= 100:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 100")
    return parsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare neural language models.")
    parser.add_argument("config", nargs="?", help="path to a legacy YAML experiment config")
    parser.add_argument("--suite", help="path to a prepared-document sweep YAML config")
    parser.add_argument("--stage", choices=("preflight", "prepare", "train", "seal", "evaluate", "report", "all"), default="preflight")
    parser.add_argument("--corpora", nargs="+", help="restrict suite execution to named corpora")
    parser.add_argument(
        "--resume", action="store_true",
        help="resume interrupted neural fits from epoch checkpoints; for --suite, reuse verified fits",
    )
    parser.add_argument("--dry-run", action="store_true", help="display suite matrix without starting work")
    parser.add_argument(
        "--train-percent",
        type=percentage,
        default=100,
        metavar="PERCENT",
        help="use the first PERCENT of train.txt sentences (default: 100)",
    )
    parser.add_argument(
        "--algorithm",
        "--model",
        dest="algorithm",
        help="run one configured algorithm (for example: ffnn or lstm); default: all configured models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="override configured seeds and run once with this seed",
    )
    parser.add_argument(
        "--context-size",
        "--ffnn-context-size",
        dest="context_size",
        type=int,
        metavar="WORDS",
        help=(
            "override FFNN context size (number of preceding words); "
            "1 is bigram-style, 2 is trigram-style"
        ),
    )
    parser.add_argument(
        "--token",
        choices=tokenizer_names(),
        default=None,
        help="tokenizer: space (default), wp (WordPunct), or tb (Treebank)",
    )
    args = parser.parse_args()
    if args.suite:
        if (args.config or args.train_percent != 100 or args.algorithm or args.seed is not None
                or args.context_size is not None or args.token is not None):
            parser.error("Suite settings belong in its YAML; legacy runner overrides cannot be combined with --suite")
        from lm.experiments import ExperimentSuite
        ExperimentSuite(args.suite, args.corpora).run(args.stage, resume=args.resume, dry_run=args.dry_run)
    else:
        if not args.config:
            parser.error("Provide a legacy config or --suite")
        if args.corpora or args.dry_run or args.stage != "preflight":
            parser.error("--stage, --corpora and --dry-run require --suite")
        run_from_config(
            args.config,
            train_percent=args.train_percent,
            algorithm=args.algorithm,
            seed_override=args.seed,
            context_size_override=args.context_size,
            tokenizer_override=args.token,
            resume=args.resume,
        )
