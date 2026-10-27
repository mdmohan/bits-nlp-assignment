"""Find rare and unseen bigrams/trigrams in held-out sentence text.

Input files should contain one sentence per line. Vocabulary counts are
collected from training text only. To keep memory use bounded, the script
discovers candidate n-grams from a reproducible sample of test sentences,
then counts those candidates exactly over the complete train and test files.
The reported examples therefore have exact counts, but the candidate search
is sampled and the script does not estimate corpus-wide unseen percentages.

Example (run from the assignment folder)::

    python3 src/analyze_rare_ngrams.py \
        --train data/processed/train_sentences.txt \
        --test data/processed/test_sentences.txt \
        --csv data/processed/rare_ngrams.csv
"""

from __future__ import annotations

import argparse
import csv
import heapq
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

from nltk.tokenize import TreebankWordTokenizer
from nltk.util import ngrams
from log_utils import display_path, log_event


TOKENIZER = TreebankWordTokenizer()
ORDERS = (2, 3)
SPECIAL_TOKENS = {"<s>", "</s>", "<UNK>"}


class ProgressBar:
    """Small stdlib-only progress bar for long file and list passes."""

    def __init__(self, label: str, total: int, unit: str) -> None:
        self.label = label
        self.total = max(total, 1)
        self.unit = unit
        self.current = 0
        self.started = time.monotonic()
        self.last_update = 0.0
        self.update_interval = 10.0

    def update(self, amount: int = 1) -> None:
        self.current += amount
        now = time.monotonic()
        percent = min(100, int(self.current * 100 / self.total))
        if percent == 100 or now - self.last_update >= self.update_interval:
            self._draw(percent, now)
            self.last_update = now

    def _draw(self, percent: int, now: float) -> None:
        width = 24
        filled = int(width * percent / 100)
        elapsed = now - self.started
        rate = self.current / elapsed if elapsed else 0
        message = (
            f"{self.label}: [{'#' * filled}{'-' * (width - filled)}] "
            f"{percent:3d}% {self.current:,}/{self.total:,} {self.unit} "
            f"({rate:,.0f} {self.unit}/s)"
        )
        if sys.stderr.isatty():
            sys.stderr.write("\r" + message)
        else:
            sys.stderr.write(message + "\n")
        sys.stderr.flush()

    def close(self) -> None:
        if self.current < self.total:
            self.current = self.total
            self._draw(100, time.monotonic())
        if sys.stderr.isatty():
            sys.stderr.write("\n")
            sys.stderr.flush()


def iter_sentences(path: Path, progress_label: str | None = None) -> Iterator[str]:
    """Yield non-empty lines, optionally showing byte progress, without buffering the file."""
    if not path.is_file():
        raise FileNotFoundError(f"Sentence file does not exist: {path}")
    progress = (
        ProgressBar(progress_label, path.stat().st_size, "bytes")
        if progress_label
        else None
    )
    first_line = True
    try:
        with path.open("rb") as source:
            for raw_line in source:
                if progress:
                    progress.update(len(raw_line))
                encoding = "utf-8-sig" if first_line else "utf-8"
                first_line = False
                sentence = raw_line.decode(encoding, errors="replace").strip()
                if sentence:
                    yield sentence
    finally:
        if progress:
            progress.close()


def tokenize(sentence: str) -> list[str]:
    return TOKENIZER.tokenize(sentence.lower())


def build_vocabulary(train_path: Path, min_word_count: int) -> set[str]:
    """Count training words in a streaming pass and return retained types."""
    counts: Counter[str] = Counter()
    for sentence in iter_sentences(train_path, "Pass 1/4 vocabulary"):
        counts.update(tokenize(sentence))
    return {
        word for word, count in counts.items() if count >= min_word_count
    }


def map_oov(tokens: list[str], vocabulary: set[str]) -> tuple[list[str], int]:
    mapped: list[str] = []
    oov_count = 0
    for token in tokens:
        if token in vocabulary:
            mapped.append(token)
        else:
            mapped.append("<UNK>")
            oov_count += 1
    return mapped, oov_count


def padded_ngrams(tokens: list[str], order: int):
    padded = ["<s>"] * (order - 1) + tokens + ["</s>"]
    return ngrams(padded, order)


def sample_test_sentences(test_paths: list[Path], limit: int, seed: int) -> list[str]:
    """Reservoir-sample sentences uniformly without loading the test file."""
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    for test_path in test_paths:
        for sentence in iter_sentences(test_path, "Pass 2/4 test sample"):
            seen += 1
            if len(sample) < limit:
                sample.append(sentence)
            else:
                replacement_index = rng.randrange(seen)
                if replacement_index < limit:
                    sample[replacement_index] = sentence
    return sample


def find_candidate_ngrams(
    sampled_sentences: list[str], vocabulary: set[str]
) -> dict[int, set[tuple[str, ...]]]:
    candidates = {order: set() for order in ORDERS}
    progress = ProgressBar("Pass 2/4 candidate discovery", len(sampled_sentences), "sentences")
    try:
        for sentence in sampled_sentences:
            tokens, _ = map_oov(tokenize(sentence), vocabulary)
            for order in ORDERS:
                candidates[order].update(padded_ngrams(tokens, order))
            progress.update()
    finally:
        progress.close()
    return candidates


def count_rare_candidates_in_train(
    train_path: Path,
    vocabulary: set[str],
    candidates: dict[int, set[tuple[str, ...]]],
    rare_train_count: int,
) -> dict[int, dict[tuple[str, ...], int]]:
    """Count only sampled candidates, discarding any above the rare cutoff."""
    active = {order: set(values) for order, values in candidates.items()}
    rare_counts: dict[int, dict[tuple[str, ...], int]] = {
        order: {} for order in ORDERS
    }

    for sentence in iter_sentences(train_path, "Pass 3/4 train n-gram counts"):
        tokens, _ = map_oov(tokenize(sentence), vocabulary)
        for order in ORDERS:
            active_order = active[order]
            for sequence in padded_ngrams(tokens, order):
                if sequence not in active_order:
                    continue
                count = rare_counts[order].get(sequence, 0) + 1
                if count > rare_train_count:
                    # Its exact count is no longer needed for this report.
                    active_order.remove(sequence)
                    rare_counts[order].pop(sequence, None)
                else:
                    rare_counts[order][sequence] = count
    # Candidates never observed in training are still unseen examples (count 0).
    return {
        order: {
            sequence: rare_counts[order].get(sequence, 0)
            for sequence in active[order]
        }
        for order in ORDERS
    }


def count_rare_candidates_in_test(
    test_paths: list[Path],
    vocabulary: set[str],
    rare_train_counts: dict[int, dict[tuple[str, ...], int]],
) -> tuple[dict[int, Counter], int, int, int]:
    """Count rare candidates exactly over test and tally test OOV rate."""
    test_counts = {order: Counter() for order in ORDERS}
    oov_count = 0
    total_words = 0
    sentence_count = 0

    for test_path in test_paths:
        for sentence in iter_sentences(test_path, "Pass 4/4 test n-gram counts"):
            sentence_count += 1
            tokens, sentence_oov = map_oov(tokenize(sentence), vocabulary)
            oov_count += sentence_oov
            total_words += len(tokens)
            for order in ORDERS:
                rare_sequences = rare_train_counts[order]
                for sequence in padded_ngrams(tokens, order):
                    if sequence in rare_sequences:
                        test_counts[order][sequence] += 1
    return test_counts, oov_count, total_words, sentence_count


def select_examples(
    train_counts: dict[tuple[str, ...], int],
    test_counts: Counter,
    top: int,
) -> list[tuple[tuple[str, ...], int, int]]:
    def is_word_sequence(sequence: tuple[str, ...]) -> bool:
        # Keep lexical-looking tokens; discard punctuation, boundary markers,
        # unknown-word markers, and one-letter fragments such as "'s".
        return all(
            token not in SPECIAL_TOKENS
            and sum(character.isalpha() for character in token) >= 2
            for token in sequence
        )

    progress = ProgressBar("Selecting reported examples", len(train_counts), "types")

    def readable_rows():
        try:
            for sequence, train_count in train_counts.items():
                if test_counts[sequence] and is_word_sequence(sequence):
                    yield (sequence, train_count, test_counts[sequence])
                progress.update()
        finally:
            progress.close()

    # Keep only the best requested examples in memory while scanning types.
    # The sort key prioritizes lower train counts, then higher test counts.
    rows = heapq.nsmallest(
        top,
        readable_rows(),
        key=lambda row: (row[1], -row[2], row[0]),
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sample test sentences to discover rare/unseen bigrams and trigrams, "
            "then count the discovered sequences exactly in train and test."
        )
    )
    parser.add_argument("--train", type=Path, required=True, help="Training sentences, one per line.")
    parser.add_argument("--test", type=Path, nargs="+", required=True, help="One or more held-out sentence files (combined as one evaluation chunk).")
    parser.add_argument(
        "--min-word-count",
        type=int,
        default=2,
        help="Minimum training frequency to retain a word (default: 2); others map to <UNK>.",
    )
    parser.add_argument(
        "--rare-train-count",
        type=int,
        default=2,
        help="Report n-grams with at most this many train occurrences (default: 2).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=100,
        help="Maximum readable examples per n-gram order to display/write (default: 100).",
    )
    parser.add_argument(
        "--candidate-sentences",
        type=int,
        default=20_000,
        help="Number of test sentences to sample for finding candidates (default: 20000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV path for the examples.")
    args = parser.parse_args()

    if args.min_word_count < 1:
        parser.error("--min-word-count must be at least 1")
    if args.rare_train_count < 0:
        parser.error("--rare-train-count must be zero or greater")
    if args.top < 1 or args.candidate_sentences < 1:
        parser.error("--top and --candidate-sentences must be at least 1")

    train_path = args.train.expanduser().resolve()
    test_paths = [path.expanduser().resolve() for path in args.test]
    if train_path in test_paths:
        parser.error("--train and --test must refer to different files")

    started_at = time.perf_counter()
    log_event(
        "info",
        f"Rare n-gram analysis started: train={display_path(train_path)}, "
        f"held-out files={len(test_paths)}, seed={args.seed}",
    )
    try:
        print("Pass 1/4: counting training vocabulary…", flush=True)
        vocabulary = build_vocabulary(train_path, args.min_word_count)
        print(f"Train vocabulary: {len(vocabulary):,} words", flush=True)

        print("Pass 2/4: sampling test sentences for candidate n-grams…", flush=True)
        sampled = sample_test_sentences(
            test_paths, limit=args.candidate_sentences, seed=args.seed
        )
        candidates = find_candidate_ngrams(sampled, vocabulary)
        print(
            f"Sampled {len(sampled):,} test sentences; candidate types: "
            f"{len(candidates[2]):,} bigrams, {len(candidates[3]):,} trigrams",
            flush=True,
        )

        print("Pass 3/4: counting candidate n-grams in all training text…", flush=True)
        rare_train_counts = count_rare_candidates_in_train(
            train_path, vocabulary, candidates, args.rare_train_count
        )
        # Candidate sets and sampled sentence strings are no longer needed.
        del candidates, sampled

        print("Pass 4/4: counting rare candidates in all test text…", flush=True)
        test_counts, oov_count, total_words, test_sentence_count = count_rare_candidates_in_test(
            test_paths, vocabulary, rare_train_counts
        )
    except (OSError, UnicodeError, ValueError) as exc:
        log_event("error", f"Rare n-gram analysis failed: {exc}")
        parser.error(str(exc))

    print(f"Test sentences: {test_sentence_count:,}")
    print(
        f"Test OOV words: {oov_count:,}/{total_words:,} "
        f"({oov_count / total_words:.2%})"
        if total_words
        else "Test OOV words: 0/0 (0.00%)"
    )
    print(
        "Candidate discovery is sampled; counts for reported examples are "
        "exact over the full train and test files."
    )

    csv_rows: list[dict[str, object]] = []
    for order in ORDERS:
        examples = select_examples(
            rare_train_counts[order], test_counts[order], args.top
        )
        unseen_types = sum(1 for _, count, _ in examples if count == 0)
        print(
            f"\n{order}-gram word examples (training count <= {args.rare_train_count}; "
            "punctuation-only and one-letter tokens omitted)"
        )
        print(f"  Sampled candidate types still rare in train: {len(rare_train_counts[order]):,}")
        print(f"  Unseen examples in displayed top {len(examples)}: {unseen_types}")
        print("  train_count | test_count | sequence")
        print("  ------------|------------|---------")
        for sequence, train_count, test_count in examples:
            print(f"  {train_count:11,} | {test_count:10,} | {' '.join(sequence)}")
            csv_rows.append(
                {
                    "order": order,
                    "sequence": " ".join(sequence),
                    "train_count": train_count,
                    "test_count": test_count,
                }
            )
        if not examples:
            print("  (no rare candidates found; increase --candidate-sentences)")

    if args.csv:
        output_path = args.csv.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=("order", "sequence", "train_count", "test_count")
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote examples to: {display_path(output_path)}")

    log_event("ok", f"Rare n-gram analysis finished in {time.perf_counter() - started_at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
