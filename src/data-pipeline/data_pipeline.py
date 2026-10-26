"""Download, clean, load, split, and analyze corpus data."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from data_download import configured_extract_path, load_corpus_config, safe_extract_zip
from database_integrity import database_files, verify_database_sha, write_database_sha
from log_utils import display_path


SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent.parent
KNOWN_PARSERS = {"gutenberg", "brown", "ptb", "reuters", "general"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or run corpus processing stages."
    )
    parser.add_argument("--corpus", required=True, help="Corpus name/type, such as gutenberg or blogs.")
    parser.add_argument(
        "--in",
        "--input-folder",
        dest="input_folder",
        type=Path,
        help="Input folder; for split-only runs this may instead be a DB file.",
    )
    parser.add_argument(
        "--out",
        "--output-folder",
        dest="output_folder",
        type=Path,
        help="Base output folder for processing stages (default: data/processed).",
    )
    parser.add_argument(
        "--download-out",
        type=Path,
        help="Raw download root when downloading and processing in one run (default: data/raw).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the configured corpus; --in must be a data.yaml manifest.",
    )
    parser.add_argument("--clean", action="store_true", help="Clean input text files.")
    parser.add_argument(
        "--include-extensionless",
        action="store_true",
        help="Include extensionless source documents during cleaning.",
    )
    parser.add_argument(
        "--encoding",
        default="auto",
        help="Input encoding for cleaning; 'auto' tries UTF-8, then detects or falls back to CP1252.",
    )
    parser.add_argument("--load", action="store_true", help="Parse files and create the corpus DB.")
    parser.add_argument(
        "--ext",
        help="Optional file extension filter for --load (for example: txt). If omitted, process all files.",
    )
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--split",
        type=float,
        metavar="TEST_RATIO",
        help="Legacy test fraction for train/test, for example --split 0.2.",
    )
    split_group.add_argument(
        "--split-ratios",
        type=float,
        nargs="+",
        metavar="RATIO",
        help="Ratios summing to 1: train test, or train valid test.",
    )
    parser.add_argument("--rare-check", action="store_true", help="Analyze rare bigrams/trigrams.")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible document-split seed.")
    parser.add_argument("--verbose", action="store_true", help="Show per-document and sample details.")
    parser.add_argument(
        "--remove-word",
        action="append",
        default=[],
        metavar="WORD",
        help="Literal word to remove during cleaning; repeat for additional words.",
    )
    parser.add_argument(
        "--remove-regex",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Regex to remove during cleaning; repeat for additional patterns.",
    )
    parser.add_argument(
        "--ignore-file",
        action="append",
        default=[],
        metavar="NAME_OR_PATH",
        help="File name or path to skip during cleaning/loading; repeat for additional files.",
    )
    return parser


def run_stage(command: list[str]) -> None:
    script_name = Path(command[1]).name if len(command) > 1 else "stage"
    stage = {
        "clean_text_corpus.py": "clean",
        "data_process.py": "load",
        "load_processed.py": "split",
        "analyze_rare_ngrams.py": "rare-check",
        "data_download.py": "download",
    }.get(script_name, script_name)
    print(f"\n== {stage} ==", flush=True)
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{stage} stage failed (exit code {exc.returncode})") from exc


def database_from_input(input_path: Path | None, corpus: str) -> Path:
    if input_path is None:
        return Path("data/processed") / corpus / f"{corpus}.db"
    if input_path.is_file() or database_files(input_path):
        return input_path
    if input_path.is_dir():
        candidates = (
            input_path / f"{corpus}.db",
            input_path / corpus / f"{corpus}.db",
        )
        for candidate in candidates:
            if database_files(candidate):
                return candidate
    raise FileNotFoundError(
        f"Could not find {corpus}.db from --in {input_path}; pass the DB file "
        "directly or use a folder containing it."
    )


def sentence_folder_from_input(input_path: Path | None, corpus: str) -> Path:
    if input_path is None:
        return Path("data/processed") / corpus
    if input_path.is_dir():
        return input_path
    if input_path.is_file():
        return input_path.parent
    raise FileNotFoundError(f"Input path does not exist: {display_path(input_path)}")


def downloaded_source_folder(corpus: str, config_path: Path, download_root: Path) -> Path:
    """Resolve the downloaded package/file to a folder consumable by data_process.py."""
    entry = load_corpus_config(config_path, corpus)
    spec = entry["download"]
    corpus_dir = download_root / "corpora" / corpus

    if spec.get("type") == "nltk":
        package = spec.get("package")
        if not isinstance(package, str) or not package:
            raise ValueError(f"Corpus '{corpus}' has NLTK download type but no package")
        package_dir = download_root / configured_extract_path(entry, package)
        archive_path = download_root / "corpora" / f"{package}.zip"
        if not package_dir.is_dir() and zipfile.is_zipfile(archive_path):
            safe_extract_zip(archive_path, package_dir.parent)
        if not package_dir.is_dir():
            raise FileNotFoundError(f"Downloaded NLTK package files were not found: {package_dir}")
        return package_dir

    if spec.get("type") == "url":
        url = spec.get("url")
        filename = spec.get("filename")
        if not filename and isinstance(url, str):
            filename = Path(unquote(urlparse(url).path)).name
        if not filename:
            raise ValueError(f"Corpus '{corpus}' has URL download type but no filename")
        downloaded_file = corpus_dir / filename
        if not downloaded_file.is_file():
            raise FileNotFoundError(f"Downloaded corpus file was not found: {downloaded_file}")
        if zipfile.is_zipfile(downloaded_file):
            relative_extract = entry.get("extract-path")
            if relative_extract:
                extracted_dir = download_root / configured_extract_path(entry)
                safe_extract_zip(downloaded_file, download_root)
                if extracted_dir.is_dir():
                    return extracted_dir
            extracted_dir = corpus_dir / "extracted"
            extracted_dir.mkdir(parents=True, exist_ok=True)
            safe_extract_zip(downloaded_file, extracted_dir)
            return extracted_dir
        return corpus_dir

    raise ValueError(f"Corpus '{corpus}' has unsupported download type {spec.get('type')!r}")


def run(args: argparse.Namespace) -> int:
    split_requested = args.split is not None or args.split_ratios is not None
    processing_selected = any((args.clean, args.load, split_requested, args.rare_check))
    if not args.download and not processing_selected:
        raise ValueError("Choose --download, --clean, --load, --split, or --rare-check.")
    if (args.remove_word or args.remove_regex) and not args.clean:
        raise ValueError("--remove-word and --remove-regex require --clean.")
    if args.download and not args.load and (split_requested or args.rare_check):
        raise ValueError("--download with --split or --rare-check requires --load in the same run.")
    if args.download and args.rare_check and not split_requested:
        raise ValueError("--download --rare-check requires --split so train/test files are created.")

    corpus = args.corpus.strip()
    if not corpus or Path(corpus).name != corpus or corpus in {".", ".."}:
        raise ValueError("--corpus must be a simple folder name.")

    input_arg = args.input_folder.expanduser() if args.input_folder else None
    download_root = None
    input_path: Path | None = None
    if args.download:
        if input_arg is None:
            raise ValueError("--download requires --in to point to a data.yaml manifest.")
        if not input_arg.is_absolute() and not input_arg.exists():
            config_beside_script = SRC_DIR / input_arg
            if config_beside_script.is_file():
                input_arg = config_beside_script
        input_path = input_arg.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"--download expects a YAML file for --in, not: {display_path(input_path)}")
        manifest_entry = load_corpus_config(input_path, corpus)
        manifest_ignored = manifest_entry.get("ignore-files", [])
        if manifest_ignored:
            if not isinstance(manifest_ignored, list) or any(not isinstance(item, str) for item in manifest_ignored):
                raise ValueError(f"Corpus '{corpus}' ignore-files must be a list of file names or relative paths")
            args.ignore_file = list(dict.fromkeys([*args.ignore_file, *manifest_ignored]))
        args.include_extensionless = args.include_extensionless or bool(
            manifest_entry.get("include-extensionless", False)
        )
        download_root = (
            args.download_out.expanduser().resolve()
            if args.download_out
            else (PROJECT_DIR / "data/raw").resolve()
        )
        download_root.mkdir(parents=True, exist_ok=True)
        run_stage(
            [
                sys.executable,
                str(SRC_DIR / "data_download.py"),
                corpus,
                "--conf",
                str(input_path),
                "--out",
                str(download_root),
            ]
        )
        print(f"\nDownloaded corpus '{corpus}' under: {display_path(download_root / 'corpora')}")
        if not processing_selected:
            return 0
        input_path = downloaded_source_folder(corpus, input_path, download_root)

    elif input_arg:
        input_path = input_arg.resolve()
    output_root = (
        args.output_folder.expanduser().resolve()
        if args.output_folder
        else (PROJECT_DIR / "data/processed").resolve()
    )
    corpus_output = output_root / corpus
    cleaned_folder = corpus_output / "cleaned"
    database_path = corpus_output / f"{corpus}.db"
    train_path = corpus_output / "train.txt"
    test_path = corpus_output / "test.txt"

    if (args.clean or args.load) and (input_path is None or not input_path.is_dir()):
        raise ValueError("--clean and --load require --in to be an input folder.")
    if args.split is not None and not 0 < args.split < 1:
        raise ValueError("--split must be greater than 0 and less than 1.")
    ratios = args.split_ratios
    if args.split is not None:
        ratios = [1.0 - args.split, args.split]
    if ratios is not None and len(ratios) not in (2, 3):
        raise ValueError("--split-ratios requires two or three values.")

    parser_type = corpus.casefold() if corpus.casefold() in KNOWN_PARSERS else "general"
    output_root.mkdir(parents=True, exist_ok=True)
    corpus_output.mkdir(parents=True, exist_ok=True)

    if args.clean:
        command = [
            sys.executable,
            str(SRC_DIR / "clean_text_corpus.py"),
            "--input-dir",
            str(input_path),
            "--output-dir",
            str(cleaned_folder),
            "--audit-dir",
            str(corpus_output / "cleaning_audit"),
            "--encoding",
            args.encoding,
        ]
        for word in args.remove_word:
            command.extend(("--remove-word", word))
        for pattern in args.remove_regex:
            command.extend(("--remove-regex", pattern))
        for ignored in args.ignore_file:
            command.extend(("--ignore-file", ignored))
        if args.include_extensionless:
            command.append("--include-extensionless")
        run_stage(command)

    loaded_this_run = False
    if args.load:
        source_folder = cleaned_folder if args.clean else input_path
        # shelve can use one file or several backend-specific files; remove all
        # prior artifacts so a fresh load cannot retain records from old input.
        for artifact in database_files(database_path):
            artifact.unlink()
        Path(str(database_path) + ".sha").unlink(missing_ok=True)
        command = [
            sys.executable,
            str(SRC_DIR / "data_process.py"),
            "--corpus-type",
            parser_type,
            "--db-path",
            str(database_path),
        ]
        if args.ext is not None:
            command.extend(("--ext", args.ext))
        for ignored in args.ignore_file:
            command.extend(("--ignore-file", ignored))
        if args.verbose:
            command.append("--verbose")
        command.append(str(source_folder))
        run_stage(command)
        write_database_sha(database_path)
        loaded_this_run = True

    if split_requested:
        db_path = database_path if loaded_this_run else database_from_input(input_path, corpus)
        if not loaded_this_run:
            verify_database_sha(db_path)
        if len(ratios) == 2:
            (corpus_output / "valid.txt").unlink(missing_ok=True)
        command = [
            sys.executable,
            str(SRC_DIR / "load_processed.py"),
            "--db-path",
            str(db_path),
            "--split-ratios",
            *(str(value) for value in ratios),
            "--seed",
            str(args.seed),
            "--train-output",
            str(train_path),
            "--test-output",
            str(test_path),
        ]
        if len(ratios) == 3:
            command.extend(("--valid-output", str(corpus_output / "valid.txt")))
        if args.verbose:
            command.append("--verbose")
        run_stage(command)

    if args.rare_check:
        if split_requested:
            rare_train = train_path
            rare_tests = [test_path]
            valid_path = corpus_output / "valid.txt"
            if len(ratios) == 3 and valid_path.is_file():
                rare_tests.insert(0, valid_path)
        else:
            sentence_folder = sentence_folder_from_input(input_path, corpus)
            rare_train = sentence_folder / "train.txt"
            rare_tests = [sentence_folder / "test.txt"]
            if not rare_train.is_file() and (sentence_folder / "train_sentences.txt").is_file():
                rare_train = sentence_folder / "train_sentences.txt"
            if not rare_tests[0].is_file() and (sentence_folder / "test_sentences.txt").is_file():
                rare_tests = [sentence_folder / "test_sentences.txt"]
            valid_candidate = sentence_folder / "valid.txt"
            if valid_candidate.is_file():
                rare_tests.insert(0, valid_candidate)

        run_stage(
            [
                sys.executable,
                str(SRC_DIR / "analyze_rare_ngrams.py"),
                "--train",
                str(rare_train),
                "--test",
                *(str(path) for path in rare_tests),
                "--csv",
                str(corpus_output / "rare-ngrams.csv"),
            ]
        )

    print(f"\nPipeline outputs: {display_path(corpus_output)}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except (FileNotFoundError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
