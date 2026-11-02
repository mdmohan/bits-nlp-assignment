#!/usr/bin/env python3
"""Find held-out n-grams with unseen contexts and unseen full sequences."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lm.tokenization import canonical_tokenizer_name, get_tokenizer, tokenizer_names
from log_utils import display_path, log_event


BOS = "<s>"
EOS = "</s>"
SPECIALS = {BOS, EOS}
CSV_FIELDS = ["order", "sequence", "train_count", "test_count", "context_oov", "tokenizer"]


def sentences(paths: list[Path], tokenizer_name: str):
    tokenizer = get_tokenizer(tokenizer_name)
    for path in paths:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line in handle:
                words = [word.lower() for word in tokenizer.tokenize(line)]
                if words:
                    yield words


def padded_ngrams(words: list[str], order: int, vocabulary: set[str]):
    sequence = [BOS] * (order - 1) + words + [EOS]
    for index in range(len(sequence) - order + 1):
        gram = tuple(sequence[index : index + order])
        context = gram[:-1]
        context_oov = any(word not in SPECIALS and word not in vocabulary for word in context)
        yield gram, context_oov


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write held-out context-OOV and unseen-sequence n-gram CSVs."
    )
    parser.add_argument("--train", required=True, type=Path, help="Training sentences, one per line")
    parser.add_argument("--test", required=True, nargs="+", type=Path, help="Held-out sentence file(s)")
    parser.add_argument("--n", required=True, type=int, choices=(2, 3), help="N-gram order: 2 or 3")
    parser.add_argument("--token", choices=tokenizer_names(), default="space", help="Tokenizer used for model input")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.token = canonical_tokenizer_name(args.token)
    train_path = args.train.expanduser().resolve()
    test_paths = [path.expanduser().resolve() for path in args.test]
    for path in [train_path, *test_paths]:
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {display_path(path)}")

    output_dir = test_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    label = "bigram" if args.n == 2 else "trigram"
    context_path = output_dir / f"rare-context-{label}.csv"
    sequence_path = output_dir / f"rare-seq-{label}.csv"

    started = time.perf_counter()
    log_event("info", f"Analyzing {label}: tokenizer={args.token}, train={display_path(train_path)}")
    train_vocab: set[str] = set()
    train_sentences = 0
    for words in sentences([train_path], args.token):
        train_sentences += 1
        train_vocab.update(words)
    log_event("info", f"Training vocabulary: {len(train_vocab):,} words from {train_sentences:,} sentences")

    # SQLite keeps the potentially large n-gram type inventories off the Python heap.
    with tempfile.TemporaryDirectory(prefix="rare-ngram-") as temp_dir:
        db = sqlite3.connect(str(Path(temp_dir) / "counts.sqlite"))
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE train (key TEXT PRIMARY KEY, count INTEGER NOT NULL)")
        db.execute(
            "CREATE TABLE heldout (key TEXT PRIMARY KEY, sequence TEXT NOT NULL, "
            "test_count INTEGER NOT NULL, context_oov INTEGER NOT NULL)"
        )

        batch: list[tuple[str]] = []
        for sentence in sentences([train_path], args.token):
            for gram, _ in padded_ngrams(sentence, args.n, train_vocab):
                batch.append((json.dumps(gram, ensure_ascii=False, separators=(",", ":")),))
                if len(batch) >= 50_000:
                    db.executemany(
                        "INSERT INTO train(key,count) VALUES (?,1) "
                        "ON CONFLICT(key) DO UPDATE SET count=count+1", batch
                    )
                    batch.clear()
        if batch:
            db.executemany(
                "INSERT INTO train(key,count) VALUES (?,1) "
                "ON CONFLICT(key) DO UPDATE SET count=count+1", batch
            )
        db.commit()
        log_event("info", "Training n-gram counts indexed")

        batch_rows: list[tuple[str, str, int, int]] = []
        heldout_sentences = heldout_events = 0
        for test_path in test_paths:
            for sentence in sentences([test_path], args.token):
                heldout_sentences += 1
                for gram, context_oov in padded_ngrams(sentence, args.n, train_vocab):
                    key = json.dumps(gram, ensure_ascii=False, separators=(",", ":"))
                    pretty = " ".join(gram)
                    batch_rows.append((key, pretty, 1, int(context_oov)))
                    heldout_events += 1
                    if len(batch_rows) >= 50_000:
                        db.executemany(
                            "INSERT INTO heldout(key,sequence,test_count,context_oov) VALUES (?,?,?,?) "
                            "ON CONFLICT(key) DO UPDATE SET test_count=test_count+1", batch_rows
                        )
                        batch_rows.clear()
        if batch_rows:
            db.executemany(
                "INSERT INTO heldout(key,sequence,test_count,context_oov) VALUES (?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET test_count=test_count+1", batch_rows
            )
        db.commit()

        query = (
            "SELECT h.sequence, COALESCE(t.count,0), h.test_count, h.context_oov "
            "FROM heldout h LEFT JOIN train t ON t.key=h.key "
        )
        order_by = " ORDER BY h.test_count DESC, h.sequence COLLATE NOCASE"
        context_count = sequence_count = 0
        with context_path.open("w", encoding="utf-8", newline="") as context_handle, \
             sequence_path.open("w", encoding="utf-8", newline="") as sequence_handle:
            context_writer = csv.DictWriter(context_handle, fieldnames=CSV_FIELDS)
            sequence_writer = csv.DictWriter(sequence_handle, fieldnames=CSV_FIELDS)
            context_writer.writeheader()
            sequence_writer.writeheader()
            cursor = db.execute(query + order_by)
            for sequence, train_count, test_count, context_oov in cursor:
                row = {
                    "order": args.n,
                    "sequence": sequence,
                    "train_count": train_count,
                    "test_count": test_count,
                    "context_oov": bool(context_oov),
                    "tokenizer": args.token,
                }
                if context_oov:
                    context_writer.writerow(row)
                    context_count += 1
                if train_count == 0:
                    sequence_writer.writerow(row)
                    sequence_count += 1
        db.close()

    log_event(
        "ok",
        f"{label}: {heldout_sentences:,} held-out sentences, {heldout_events:,} events; "
        f"{context_count:,} unseen-context types, {sequence_count:,} unseen-sequence types",
    )
    log_event("ok", f"Wrote {display_path(context_path)} and {display_path(sequence_path)}")
    log_event("info", f"Finished in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
