"""Command-line entry point.

    python run.py configs/brown.yaml
"""

import argparse

from lm.controller import run_from_config


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
    parser.add_argument("config", help="path to a YAML experiment config")
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
    args = parser.parse_args()
    run_from_config(
        args.config,
        train_percent=args.train_percent,
        algorithm=args.algorithm,
        seed_override=args.seed,
        context_size_override=args.context_size,
    )
