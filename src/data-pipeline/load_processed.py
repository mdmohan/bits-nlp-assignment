import shelve
from pathlib import Path
import random
from typing import Generator, List, Sequence, Union
import argparse
import math
from database_integrity import database_files
from log_utils import display_path


def stream_sentences(source_path: Union[str, Path]) -> Generator[str, None, None]:
    """
    Generator that yields sentences one by one from either a .txt text dump 
    or a shelve database file.
    """
    path = Path(source_path)

    if not path.exists() and not database_files(path):
        raise FileNotFoundError(f"Source file or database does not exist: {path}")

    # 1. Try reading as a Shelve DB using exact path
    try:
        with shelve.open(str(path), flag='r') as db:
            for key in db:
                record = db[key]
                if isinstance(record, list):
                    for paragraph in record:
                        for sentence in paragraph.get('sentences', []):
                            clean_sent = sentence.strip()
                            if clean_sent:
                                yield clean_sent
            return  # Successfully processed as DB
    except Exception as e:
        # Not a valid shelve DB or failed to open; proceed to plain text file fallback
        print("Exception:", str(e))

    print("Not a shelf DB")
    # Load from plain text dump file (one sentence per line)
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            clean_sent = line.strip()
            if clean_sent:
                yield clean_sent


def load_sentences(source_path: Union[str, Path], stream: bool = False) -> Union[List[str], Generator[str, None, None]]:
    """
    Loads sentences from a database or text dump.

    :param source_path: Path to parsed_corpus.db or all_sentences.txt.
    :param stream: If True, returns an iterator/generator (memory-efficient).
                   If False, returns a complete List[str].
    """
    generator = stream_sentences(source_path)
    if stream:
        return generator
    return list(generator)


def _sentences_in_record(record: object) -> list[str]:
    """Flatten one shelve value while retaining its document as the split unit."""
    sentences: list[str] = []
    if not isinstance(record, list):
        return sentences
    for paragraph in record:
        if not isinstance(paragraph, dict):
            continue
        for sentence in paragraph.get("sentences", []):
            clean_sentence = sentence.strip()
            if clean_sentence:
                sentences.append(clean_sentence)
    return sentences


def load_document_sentences(source_path: Union[str, Path]) -> list[tuple[str, list[str]]]:
    """Load sentences grouped by the source-file key stored in the shelve DB."""
    path = Path(source_path)
    if not path.exists() and not database_files(path):
        raise FileNotFoundError(f"Database does not exist: {path}")

    documents: list[tuple[str, list[str]]] = []
    with shelve.open(str(path), flag="r") as db:
        for document_id in sorted(db.keys()):
            documents.append((document_id, _sentences_in_record(db[document_id])))
    return documents


def split_document_sentences(
    documents: Sequence[tuple[str, list[str]]],
    ratios: Sequence[float],
    seed: int = 42,
) -> list[list[str]]:
    """Split whole documents reproducibly using train, valid, test ratios.

    For a two-way split, ratios are train, test. For a three-way split they
    are train, valid, test. Documents stay intact and are flattened afterward.
    """
    if len(ratios) not in (2, 3):
        raise ValueError("Provide exactly two ratios (train,test) or three (train,valid,test)")
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("Every split ratio must be a finite number greater than 0")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Split ratios must sum to 1.0; got {sum(ratios):.10g}")
    if len(documents) < 2:
        raise ValueError("At least two source documents are required for a split")
    if len(documents) < len(ratios):
        raise ValueError(f"At least {len(ratios)} source documents are required for this split")

    shuffled_ids = [document_id for document_id, _ in documents]
    random.Random(seed).shuffle(shuffled_ids)
    exact_counts = [len(shuffled_ids) * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact_counts]
    # Largest-remainder allocation makes the document counts sum exactly.
    for index in sorted(range(len(ratios)), key=lambda i: exact_counts[i] - counts[i], reverse=True)[:len(shuffled_ids) - sum(counts)]:
        counts[index] += 1
    # Ensure every requested partition is represented, moving documents from
    # the largest partition when rounding would otherwise produce a zero.
    for index, count in enumerate(counts):
        if count == 0:
            donor = max(range(len(counts)), key=counts.__getitem__)
            counts[donor] -= 1
            counts[index] += 1

    partition_ids: list[set[str]] = []
    offset = 0
    for count in counts:
        partition_ids.append(set(shuffled_ids[offset:offset + count]))
        offset += count

    partitions: list[list[str]] = [[] for _ in ratios]
    # Keep a stable document order in the resulting lists.
    for document_id, sentences in sorted(documents, key=lambda item: item[0]):
        for index, ids in enumerate(partition_ids):
            if document_id in ids:
                partitions[index].extend(sentences)
                break
    return partitions


# ==========================================
# Example Usage & NLP Pipeline Integration
# ==========================================
if __name__ == '__main__':
    # 1. Load into in-memory List

    parser_cli = argparse.ArgumentParser(
        description="Load parsed corpus text or db"
    )
    source_group = parser_cli.add_mutually_exclusive_group()
    source_group.add_argument(
        "--db-path", "-d",
        type=str,
        default=None,
        help="Path to db file"
    )
    source_group.add_argument(
        "--text", "-t",
        type=str,
        default=None,
        help="path to sentence dump"
    )
    parser_cli.add_argument(
        "--split",
        type=float,
        default=None,
        metavar="TEST_RATIO",
        help="Legacy test fraction; splits into train/test. Prefer --split-ratios.",
    )
    parser_cli.add_argument(
        "--split-ratios",
        type=float,
        nargs="+",
        metavar="RATIO",
        help="Ratios summing to 1: train test, or train valid test; requires --db-path.",
    )
    parser_cli.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible document assignment (default: 42).",
    )
    parser_cli.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Optional path to write train_sentences, one sentence per line.",
    )
    parser_cli.add_argument(
        "--test-output",
        type=Path,
        default=None,
        help="Optional path to write test_sentences, one sentence per line.",
    )
    parser_cli.add_argument(
        "--valid-output",
        type=Path,
        default=None,
        help="Optional path to write validation sentences for a three-way split.",
    )
    parser_cli.add_argument(
        "--verbose",
        action="store_true",
        help="Print sample sentences after splitting.",
    )

    args = parser_cli.parse_args()

    if args.split is not None and args.split_ratios is not None:
        parser_cli.error("Use only one of --split and --split-ratios")
    ratios = args.split_ratios
    if args.split is not None:
        ratios = [1.0 - args.split, args.split]
    if ratios is not None and args.db_path is None:
        parser_cli.error("--split/--split-ratios requires --db-path so documents remain grouped")
    provided_outputs = [args.train_output, args.valid_output, args.test_output]
    if ratios is None and any(provided_outputs):
        parser_cli.error("output paths require --split or --split-ratios")
    if ratios is not None:
        expected_outputs = 3 if len(ratios) == 3 else 2 if len(ratios) == 2 else 0
        if expected_outputs == 0:
            parser_cli.error("--split-ratios requires exactly two or three values")
        selected_outputs = [args.train_output, args.test_output] if expected_outputs == 2 else provided_outputs
        if any(provided_outputs) and any(path is None for path in selected_outputs):
            parser_cli.error("provide --train-output and --test-output; three-way splits also require --valid-output")
        if expected_outputs == 2 and args.valid_output is not None:
            parser_cli.error("--valid-output is only used for three-way splits")
    if args.db_path and ratios is not None and any(provided_outputs):
        database_path = Path(args.db_path).expanduser().resolve()
        output_paths = {path.expanduser().resolve() for path in provided_outputs if path}
        if any(provided_outputs) and len(output_paths) != len([path for path in provided_outputs if path]):
            parser_cli.error("split output paths must be different")
        if database_path in output_paths:
            parser_cli.error("split output paths must not overwrite the input database")

    if args.text:
        sentences_list = load_sentences(args.text)
        print(f"Loaded {len(sentences_list)} sentences into memory.")
        if sentences_list:
            print("Sample:", sentences_list[0])
        

    # 2. Load or split the shelve DB. Split at source-file level before
    # flattening, so no document can appear in both partitions.
    if args.db_path and ratios is not None:
        documents = load_document_sentences(args.db_path)
        try:
            partitions = split_document_sentences(documents, ratios=ratios, seed=args.seed)
        except ValueError as exc:
            parser_cli.error(str(exc))
        partition_names = ["train", "test"] if len(ratios) == 2 else ["train", "valid", "test"]
        print("Sentences: " + ", ".join(f"{len(part)} {name}" for name, part in zip(partition_names, partitions)))

        output_paths = [args.train_output, args.valid_output, args.test_output] if len(ratios) == 3 else [args.train_output, args.test_output]
        if all(output_paths):
            for name, sentences, output_path in zip(partition_names, partitions, output_paths):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text("".join(sentence + "\n" for sentence in sentences), encoding="utf-8")
                print(f"Wrote {name} sentences to: {display_path(output_path)}")

        if args.verbose:
            for name, sentences in zip(partition_names, partitions):
                if sentences:
                    print(f"{name.title()} sample:", sentences[0])

    # Load the unsplit shelve DB when --split was not requested.
    elif args.db_path:
        db_sentences = load_sentences(args.db_path)
        print(f"Loaded {len(db_sentences)} sentences from DB.")
    
        # # 3. Stream generator (Memory-efficient for multi-gigabyte corpora)
        # sentence_iter = load_sentences('parsed_corpus.db', stream=True)
        # for idx, sentence in enumerate(sentence_iter):
        #     if idx >= 5:  # Inspect first 5 sentences
        #         break
        #     print(f"Sentence {idx + 1}: {sentence}")
