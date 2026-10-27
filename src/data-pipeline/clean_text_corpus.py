"""Clean a folder of text documents into a new, audited corpus folder.

This is a single-command entry point around the repository's existing
document-quality and block-aware content cleaners. It deliberately writes one
cleaned text file per retained source document; it does not merge documents or
tokenize them for a particular model.

Example (run from the ``llmassign`` repository root)::

    python3 clean_text_corpus.py \\
        --input-dir data/raw/text \\
        --output-dir data/cleaned/rfc-for-nlp \\
        --audit-dir data/reports/rfc-for-nlp
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from collections import Counter
from pathlib import Path

from clean_data import TextCleaner
from training_content_cleaner import TrainingContentCleaner
from log_utils import display_path, log_event


def _read_text(path: Path, encoding: str) -> tuple[str, str]:
    """Read a text file and return its content plus the actual encoding used."""
    raw = path.read_bytes()
    if encoding.casefold() != "auto":
        return raw.decode(encoding), encoding

    try:
        return raw.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        try:
            from charset_normalizer import from_bytes
        except ImportError:
            # Older text corpora often use Windows-1252/Latin-1 punctuation.
            # Keep automatic mode usable without adding a dependency; undefined
            # CP1252 bytes are replaced instead of aborting the whole run.
            return raw.decode("cp1252", errors="replace"), "cp1252"

    match = from_bytes(raw).best()
    if match is None:
        return raw.decode("cp1252", errors="replace"), "cp1252"
    return str(match), match.encoding or "detected"


def _source_files(
    input_dir: Path,
    recursive: bool,
    include_extensionless: bool = False,
    ignored_files: list[str] | None = None,
) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    ignored = {item.replace("\\", "/").strip("/") for item in (ignored_files or [])}
    return sorted(
        path
        for path in iterator
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(input_dir).parts)
        and (path.suffix.casefold() == ".txt" or (include_extensionless and not path.suffix))
        and path.name not in ignored
        and path.relative_to(input_dir).as_posix() not in ignored
    )


def _safe_output_name(source: Path, used_names: set[str]) -> str:
    """Create a stable flat output filename, disambiguating repeated stems."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "document"
    filename = f"{stem}.txt"
    if filename in used_names:
        digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:10]
        filename = f"{stem}_{digest}.txt"
    used_names.add(filename)
    return filename


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load .txt documents from a folder, apply the LLM assignment's "
            "quality and content cleaning, and write retained documents to a "
            "separate folder."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=None,
        help="Directory for document-gate and content-cleaning JSON reports.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Text encoding, or 'auto' to try UTF-8 then charset-normalizer.",
    )
    parser.add_argument(
        "--remove-word",
        action="append",
        default=[],
        metavar="WORD",
        help="Remove this literal whole word (repeat the option for more words).",
    )
    parser.add_argument(
        "--remove-regex",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Remove matches for this regular expression (repeat for more patterns).",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search subfolders for source files.",
    )
    parser.add_argument(
        "--include-extensionless",
        action="store_true",
        help="Also clean extensionless documents (useful for NLTK Reuters).",
    )
    parser.add_argument(
        "--ignore-file",
        action="append",
        default=[],
        metavar="NAME_OR_PATH",
        help="File name or path relative to the input directory to skip; repeat as needed.",
    )

    gates = parser.add_argument_group("document quality gates")
    gates.add_argument("--min-chars", type=int, default=50)
    gates.add_argument("--max-duplicate-paragraph-ratio", type=float, default=0.30)
    gates.add_argument("--target-language", default="en")
    gates.add_argument(
        "--dedup-strategy", choices=("exact", "near", "both"), default="both"
    )
    gates.add_argument("--similarity-threshold", type=float, default=0.85)
    gates.add_argument("--minhash-permutations", type=int, default=128)

    content = parser.add_argument_group("block-aware content cleaning")
    content.add_argument(
        "--content-cleaning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove recognized front matter, TOCs, page furniture, boilerplate, "
            "low-information tables, diagrams, and extraction noise."
        ),
    )
    content.add_argument("--min-content-chars", type=int, default=200)
    content.add_argument("--min-alphabetic-ratio", type=float, default=0.25)
    content.add_argument("--max-cli-lines", type=int, default=160)
    content.add_argument("--max-samples-per-reason", type=int, default=3)
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    audit_dir = (
        args.audit_dir.expanduser().resolve()
        if args.audit_dir
        else output_dir.parent / f"{output_dir.name}_audit"
    )

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError(
            "Output directory must be outside the input directory so generated "
            "files cannot be re-read as source documents."
        )
    if args.min_chars < 0 or args.min_content_chars < 0:
        raise ValueError("Minimum character limits must be zero or greater.")
    if not 0 <= args.max_duplicate_paragraph_ratio <= 1:
        raise ValueError("--max-duplicate-paragraph-ratio must be between 0 and 1.")
    if not 0 < args.similarity_threshold <= 1:
        raise ValueError("--similarity-threshold must be greater than 0 and at most 1.")
    if args.minhash_permutations < 1:
        raise ValueError("--minhash-permutations must be at least 1.")
    if not 0 <= args.min_alphabetic_ratio <= 1:
        raise ValueError("--min-alphabetic-ratio must be between 0 and 1.")
    if args.max_cli_lines < 1 or args.max_samples_per_reason < 0:
        raise ValueError("--max-cli-lines must be positive and audit sample count non-negative.")

    return input_dir, output_dir, audit_dir


def run(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    input_dir, output_dir, audit_dir = _validate_args(args)
    word_patterns = [
        re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
        for word in args.remove_word
    ]
    regex_patterns = [re.compile(pattern) for pattern in args.remove_regex]
    source_files = _source_files(
        input_dir,
        args.recursive,
        include_extensionless=args.include_extensionless,
        ignored_files=args.ignore_file,
    )
    if not source_files:
        description = ".txt or extensionless files" if args.include_extensionless else ".txt files"
        raise ValueError(f"No {description} found under {input_dir}")

    log_event(
        "info",
        f"Cleaning {len(source_files):,} source files; strategy={args.dedup_strategy}, "
        f"content-cleaning={'yes' if args.content_cleaning else 'no'}, "
        f"remove-words={len(args.remove_word)}, remove-regex={len(args.remove_regex)}",
    )

    documents: dict[str, dict] = {}
    encodings: dict[str, str] = {}
    read_errors: list[dict[str, str]] = []
    last_progress_at = time.monotonic()
    for index, source in enumerate(source_files, start=1):
        try:
            text, actual_encoding = _read_text(source, args.encoding)
        except (OSError, UnicodeError, ValueError) as exc:
            read_errors.append({"source_file": str(source), "error": str(exc)})
        else:
            for pattern in (*word_patterns, *regex_patterns):
                text = pattern.sub(" ", text)
            key = str(source)
            documents[key] = {
                "extracted_text": text,
                "output_filename": str(source),
                "status": "loaded_text_input",
            }
            encodings[key] = actual_encoding

        now = time.monotonic()
        if now - last_progress_at >= 10 or index == len(source_files):
            log_event("progress", f"Read {index:,}/{len(source_files):,} source files")
            last_progress_at = now

    if not documents:
        detail = read_errors[0]["error"] if read_errors else "No readable files."
        raise ValueError(f"No readable text files found. First error: {detail}")

    audit_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    document_audit = audit_dir / "document_cleaning_audit.json"
    content_audit = audit_dir / "content_cleaning_audit.json"

    print(f"Loaded {len(documents):,} of {len(source_files):,} text files.")
    if read_errors:
        print(f"Skipped unreadable files: {len(read_errors):,}")

    cleaner = TextCleaner(
        min_chars=args.min_chars,
        max_dup_paragraph_ratio=args.max_duplicate_paragraph_ratio,
        target_lang=args.target_language,
        dedup_strategy=args.dedup_strategy,
        similarity_threshold=args.similarity_threshold,
        num_perm=args.minhash_permutations,
    )
    quality_started_at = time.perf_counter()
    log_event("info", "Quality and duplicate filtering started")
    quality_kept, quality_rejected = cleaner.filter_corpus(
        documents,
        audit_report_path=document_audit,
    )
    log_event(
        "ok",
        f"Quality filtering finished in {time.perf_counter() - quality_started_at:.1f}s; "
        f"kept={len(quality_kept):,}, rejected={len(quality_rejected):,}",
    )

    if args.content_cleaning:
        content_started_at = time.perf_counter()
        log_event("info", "Content cleanup started")
        content_cleaner = TrainingContentCleaner(
            min_content_chars=args.min_content_chars,
            min_alphabetic_ratio=args.min_alphabetic_ratio,
            max_cli_lines=args.max_cli_lines,
            max_samples_per_reason=args.max_samples_per_reason,
        )
        final_documents, content_rejected = content_cleaner.clean_corpus(
            quality_kept,
            audit_report_path=content_audit,
            cleaned_text_dir=output_dir,
            upstream_rejected=quality_rejected,
        )
        log_event(
            "ok",
            f"Content cleanup finished in {time.perf_counter() - content_started_at:.1f}s; "
            f"kept={len(final_documents):,}, rejected={len(content_rejected):,}",
        )
    else:
        final_documents = quality_kept
        content_rejected = {}
        used_names: set[str] = set()
        for source_name, data in final_documents.items():
            source = Path(source_name)
            destination = output_dir / _safe_output_name(source, used_names)
            destination.write_text(
                data["extracted_text"].rstrip() + "\n",
                encoding="utf-8",
            )
            data["cleaned_output_filename"] = str(destination)

    # Add provenance for files that could not be decoded before the cleaner ran.
    if read_errors:
        import json

        manifest_path = audit_dir / "read_errors.json"
        manifest_path.write_text(
            json.dumps({"read_errors": read_errors}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

    print("\nCorpus cleaning complete.")
    print(f"Input documents:       {len(source_files):,}")
    print(f"Readable documents:    {len(documents):,}")
    print(f"Passed quality gates:  {len(quality_kept):,}")
    print(f"Rejected at gates:     {len(quality_rejected):,}")
    print(f"Final cleaned files:   {len(final_documents):,}")
    print(f"Rejected in content:   {len(content_rejected):,}")
    print(f"Clean text folder:     {display_path(output_dir)}")
    print(f"Audit reports folder:  {display_path(audit_dir)}")
    print(
        "Encoding summary:     "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(Counter(encodings.values()).items())
        )
    )
    if args.content_cleaning:
        print(
            "Review the content audit samples, especially removed ASCII diagrams "
            "and RFC boilerplate, before using the corpus."
        )
    log_event("ok", f"Cleaning finished in {time.perf_counter() - started_at:.1f}s")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (ImportError, OSError, RuntimeError, ValueError, re.error) as exc:
        log_event("error", str(exc))
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
