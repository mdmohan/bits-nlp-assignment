"""Compare vocabulary sizes from common tokenizers on a line-based text file.

Usage from the repository root:
    python src/analyze_tokenizer_vocab.py data/processed/blog-posts/train.txt

NLTK's WordPunctTokenizer and TreebankWordTokenizer do not require punkt data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer


def whitespace(line: str) -> list[str]:
    return line.split()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare token and vocabulary counts using common tokenizers."
    )
    parser.add_argument("text", type=Path, help="text file; each nonempty line is processed separately")
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="preserve case (default: lowercase, matching neural training)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        metavar="LINES",
        help="print progress after this many lines (default: 500000; 0 disables)",
    )
    args = parser.parse_args()

    path = args.text.expanduser()
    if not path.is_file():
        parser.error(f"text file not found: {path}")
    if args.progress_every < 0:
        parser.error("--progress-every must be zero or greater")

    treebank = TreebankWordTokenizer()
    wordpunct = WordPunctTokenizer()
    tokenizers: list[tuple[str, Callable[[str], list[str]]]] = [
        ("whitespace (current)", whitespace),
        ("WordPunct", wordpunct.tokenize),
        ("Treebank", treebank.tokenize),
    ]

    print(f"File: {path}")
    print("Case handling: preserved" if args.case_sensitive else "Case handling: lowercase")
    print("Each tokenizer reads the file separately to limit memory use.\n")
    print(f"{'Tokenizer':<24} {'Lines':>12} {'Tokens':>16} {'Unique types':>16}")
    print("-" * 72)

    for name, tokenize in tokenizers:
        types: set[str] = set()
        lines = tokens = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                words = tokenize(line)
                if not args.case_sensitive:
                    words = [word.lower() for word in words]
                types.update(words)
                tokens += len(words)
                lines += 1
                if args.progress_every and lines % args.progress_every == 0:
                    print(f"  {name}: {lines:,} lines processed", flush=True)

        print(f"{name:<24} {lines:>12,} {tokens:>16,} {len(types):>16,}")
        del types

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
