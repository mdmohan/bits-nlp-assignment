import shelve
from pathlib import Path
import random
from typing import Generator, List, Sequence, Union
import argparse


def stream_sentences(source_path: Union[str, Path]) -> Generator[str, None, None]:
    """
    Generator that yields sentences one by one from either a .txt text dump 
    or a shelve database file.
    """
    path = Path(source_path)

    if not path.exists():
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
    if not path.exists():
        raise FileNotFoundError(f"Database does not exist: {path}")

    documents: list[tuple[str, list[str]]] = []
    with shelve.open(str(path), flag="r") as db:
        for document_id in sorted(db.keys()):
            documents.append((document_id, _sentences_in_record(db[document_id])))
    return documents


def split_document_sentences(
    documents: Sequence[tuple[str, list[str]]],
    test_ratio: float,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split whole documents reproducibly, returning train then test sentences.

    ``test_ratio`` is the fraction of documents assigned to test. Documents
    stay intact during assignment; sentences are flattened only afterward.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be greater than 0 and less than 1")
    if len(documents) < 2:
        raise ValueError("At least two source documents are required for a split")

    shuffled_ids = [document_id for document_id, _ in documents]
    random.Random(seed).shuffle(shuffled_ids)
    test_count = round(len(shuffled_ids) * test_ratio)
    test_count = min(max(test_count, 1), len(shuffled_ids) - 1)
    test_ids = set(shuffled_ids[:test_count])

    train_sentences: list[str] = []
    test_sentences: list[str] = []
    # Keep a stable document order in the resulting lists.
    for document_id, sentences in sorted(documents, key=lambda item: item[0]):
        destination = test_sentences if document_id in test_ids else train_sentences
        destination.extend(sentences)
    return train_sentences, test_sentences


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
        help="Test fraction (for example 0.2); splits whole source documents and requires --db-path.",
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

    args = parser_cli.parse_args()

    if args.split is not None and args.db_path is None:
        parser_cli.error("--split requires --db-path so documents remain grouped")
    if (args.train_output is None) != (args.test_output is None):
        parser_cli.error("--train-output and --test-output must be provided together")
    if args.split is None and args.train_output is not None:
        parser_cli.error("--train-output/--test-output require --split")
    if args.train_output is not None and args.train_output.resolve() == args.test_output.resolve():
        parser_cli.error("--train-output and --test-output must be different paths")
    if args.db_path and args.split is not None and args.train_output:
        database_path = Path(args.db_path).expanduser().resolve()
        output_paths = {
            args.train_output.expanduser().resolve(),
            args.test_output.expanduser().resolve(),
        }
        if database_path in output_paths:
            parser_cli.error("split output paths must not overwrite the input database")

    print(args.text, args.db_path)
    if args.text:
        sentences_list = load_sentences(args.text)
        print(f"Loaded {len(sentences_list)} sentences into memory.")
        if sentences_list:
            print("Sample:", sentences_list[0])
        

    # 2. Load or split the shelve DB. Split at source-file level before
    # flattening, so no document can appear in both partitions.
    if args.db_path and args.split is not None:
        documents = load_document_sentences(args.db_path)
        try:
            train_sentences, test_sentences = split_document_sentences(
                documents, test_ratio=args.split, seed=args.seed
            )
        except ValueError as exc:
            parser_cli.error(str(exc))
        test_document_count = min(
            max(round(len(documents) * args.split), 1), len(documents) - 1
        )
        train_document_count = len(documents) - test_document_count
        print(f"Documents: {train_document_count} train, {test_document_count} test")
        print(f"Sentences: {len(train_sentences)} train, {len(test_sentences)} test")

        if args.train_output and args.test_output:
            args.train_output.parent.mkdir(parents=True, exist_ok=True)
            args.test_output.parent.mkdir(parents=True, exist_ok=True)
            args.train_output.write_text(
                "".join(sentence + "\n" for sentence in train_sentences),
                encoding="utf-8",
            )
            args.test_output.write_text(
                "".join(sentence + "\n" for sentence in test_sentences),
                encoding="utf-8",
            )
            print(f"Wrote train sentences to: {args.train_output}")
            print(f"Wrote test sentences to : {args.test_output}")

        if train_sentences:
            print("Train sample:", train_sentences[0])
        if test_sentences:
            print("Test sample :", test_sentences[0])

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
