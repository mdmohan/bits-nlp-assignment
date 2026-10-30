"""Run corpus preparation from a small model/data YAML configuration."""

from __future__ import annotations

import argparse
import shelve
import subprocess
import sys
import time
import zipfile
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - environment dependency check
    yaml = None

from data_download import (
    configured_extract_path,
    configured_string_list,
    extracted_source_folder,
    load_corpus_config,
    local_archive_path,
    local_extract_path,
    safe_extract_archive,
    safe_extract_zip,
)
from database_integrity import database_files, verify_database_sha
from log_utils import display_path, log_event


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "data.yaml"


def run_pipeline(command: list[str], stage_summary: str) -> None:
    started_at = time.perf_counter()
    print(f"\n== {stage_summary}: started ==", flush=True)
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        log_event("error", f"{stage_summary} failed after {time.perf_counter() - started_at:.1f}s (exit {exc.returncode})")
        raise RuntimeError(f"Pipeline failed during {stage_summary} (exit code {exc.returncode})") from exc
    log_event("ok", f"{stage_summary} finished in {time.perf_counter() - started_at:.1f}s")


def read_config(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("Reading the pipeline config requires PyYAML; install it with: python -m pip install PyYAML")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file does not exist: {display_path(path)}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config {display_path(path)}: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("Config must contain a top-level 'data' mapping")
    return config["data"]


def split_ratios(data: dict) -> list[float]:
    value = data.get("split_ratio", data.get("split_ratios", [0.8, 0.2]))
    if isinstance(value, dict):
        value = value.get("ratios")
    if not isinstance(value, list):
        raise ValueError("data.split_ratio must be a list, or a mapping containing ratios: [...]")
    if len(value) not in (2, 3) or any(not isinstance(item, (int, float)) for item in value):
        raise ValueError("data.split_ratio must contain two or three numeric ratios")
    return [float(item) for item in value]


def _manifest_source(corpus: str, manifest_path: Path, raw_root: Path) -> Path | None:
    """Find or extract a corpus using its manifest download metadata."""
    entry = load_corpus_config(manifest_path, corpus)
    spec = entry["download"]
    corpora_root = raw_root / "corpora"
    kind = spec.get("type")
    if kind == "nltk":
        package = spec.get("package")
        if not isinstance(package, str) or not package:
            raise ValueError(f"Corpus '{corpus}' has no NLTK package in the manifest")
        package_dir = raw_root / configured_extract_path(entry, package)
        archive_path = corpora_root / f"{package}.zip"
        if not package_dir.is_dir() and zipfile.is_zipfile(archive_path):
            # NLTK package archives contain the package folder at their root.
            safe_extract_zip(archive_path, package_dir.parent)
        return package_dir if package_dir.is_dir() else None
    if kind == "url":
        corpus_dir = corpora_root / corpus
        filename = spec.get("filename")
        url = spec.get("url")
        if not filename and isinstance(url, str):
            from urllib.parse import unquote, urlparse
            filename = Path(unquote(urlparse(url).path)).name
        if not filename:
            raise ValueError(f"Corpus '{corpus}' has no URL filename in the manifest")
        archive_path = corpus_dir / filename
        if not archive_path.is_file():
            return None
        if zipfile.is_zipfile(archive_path):
            if entry.get("extract-path"):
                extracted_path = raw_root / configured_extract_path(entry)
                if not extracted_path.is_dir():
                    safe_extract_zip(archive_path, raw_root)
                if extracted_path.is_dir():
                    return extracted_path
            extracted = corpus_dir / "extracted"
            marker = corpus_dir / f".{filename}.extracted"
            if not marker.is_file() or marker.read_text(encoding="utf-8") != filename:
                extracted.mkdir(parents=True, exist_ok=True)
                safe_extract_zip(archive_path, extracted)
                marker.write_text(filename, encoding="utf-8")
            return extracted
        return corpus_dir
    if kind == "local":
        package = spec.get("package")
        extract_value = entry.get("extract-path")
        if not isinstance(package, str) or not package.strip():
            raise ValueError(f"Corpus '{corpus}' has local download type but no archive package path")
        if not isinstance(extract_value, str) or not extract_value.strip():
            raise ValueError(f"Corpus '{corpus}' has local download type but no extract-path")
        archive_path = local_archive_path(package, PROJECT_DIR, raw_root)
        if not archive_path.is_file():
            return None
        extract_dir = local_extract_path(extract_value, PROJECT_DIR)
        if not extract_dir.is_dir() or not any(extract_dir.iterdir()):
            safe_extract_archive(archive_path, extract_dir)
        return extracted_source_folder(extract_dir)
    raise ValueError(f"Unsupported download type for '{corpus}': {kind!r}")


def command_for_pipeline(
    corpus: str,
    output_root: Path,
    ratios: list[float],
    seed: int,
    input_path: Path | None,
    clean: bool,
    rare_check: bool,
    verbose: bool,
    ignore_files: list[str],
    include_extensionless: bool = False,
    remove_words: list[str] | None = None,
    remove_regex: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "data_pipeline.py"),
        "--corpus", corpus,
        "--out", str(output_root),
    ]
    if input_path is not None:
        command.extend(("--in", str(input_path)))
    if clean:
        command.append("--clean")
        for word in remove_words or []:
            command.extend(("--remove-word", word))
        for pattern in remove_regex or []:
            command.extend(("--remove-regex", pattern))
    if include_extensionless:
        command.append("--include-extensionless")
    for ignored in ignore_files:
        command.extend(("--ignore-file", ignored))
    command.extend(("--load", "--split-ratios", *(str(ratio) for ratio in ratios), "--seed", str(seed)))
    if rare_check:
        command.append("--rare-check")
    if verbose:
        command.append("--verbose")
    return command


def run(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    config_path = args.config.expanduser().resolve()
    data = read_config(config_path)
    corpus = data.get("corpus", data.get("loader"))
    if not isinstance(corpus, str) or not corpus.strip() or Path(corpus).name != corpus:
        raise ValueError("data.corpus must be a simple configured corpus name")
    corpus = corpus.strip()
    if "split_ratio" not in data and "split_ratios" not in data:
        data = {**data, "split_ratio": data.get("params", {}).get("ratios", [0.8, 0.2])}
    ratios = split_ratios(data)
    configured_seed = int(data.get("split_seed", data.get("params", {}).get("split_seed", 42)))
    seed = args.seed if args.seed is not None else configured_seed
    use_cached = bool(data.get("use_cached", True))
    clean = bool(data.get("clean", False))
    configured_rare_check = bool(data.get("rare-check", data.get("rare_check", False)))
    rare_check = configured_rare_check if args.rare_check is None else args.rare_check
    verbose = bool(args.verbose or data.get("verbose", False))
    raw_root = Path(data.get("input", "data/raw")).expanduser()
    output_root = Path(data.get("output", "data/processed")).expanduser()
    if not raw_root.is_absolute():
        raw_root = PROJECT_DIR / raw_root
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root
    raw_root, output_root = raw_root.resolve(), output_root.resolve()
    manifest_value = args.manifest or Path(data.get("manifest", DEFAULT_MANIFEST))
    manifest_path = manifest_value.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_DIR / manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Corpus manifest does not exist: {display_path(manifest_path)}")
    manifest_entry = load_corpus_config(manifest_path, corpus)
    ignore_files = manifest_entry.get("ignore-files", [])
    if not isinstance(ignore_files, list) or any(not isinstance(item, str) for item in ignore_files):
        raise ValueError(f"Corpus '{corpus}' ignore-files must be a list of file names or relative paths")
    include_extensionless = manifest_entry.get("include-extensionless", False)
    if not isinstance(include_extensionless, bool):
        raise ValueError(f"Corpus '{corpus}' include-extensionless must be true or false")
    remove_words = configured_string_list(manifest_entry, "remove-words")
    remove_regex = configured_string_list(manifest_entry, "remove-regex")

    log_event(
        "info",
        f"Preparing {corpus}: ratios={ratios}, seed={seed}, cached={'yes' if use_cached else 'no'}, "
        f"clean={'yes' if clean else 'no'}, rare-check={'yes' if rare_check else 'no'}",
    )
    if args.seed is not None:
        log_event("info", f"Split seed overridden by command line (config seed={configured_seed})")
    log_event(
        "info",
        f"Manifest settings: ignored files={len(ignore_files)}, remove-words={len(remove_words)}, "
        f"remove-regex={len(remove_regex)}",
    )

    corpus_output = output_root / corpus
    database_path = corpus_output / f"{corpus}.db"
    if use_cached and database_files(database_path):
        log_event("info", "Cached database found; checking its checksum before reuse")
        try:
            verify_database_sha(database_path)
            with shelve.open(str(database_path), flag="r") as cached_db:
                cached_documents = len(cached_db)
            log_event("info", f"Cached database contains {cached_documents:,} source documents")
            if cached_documents < len(ratios):
                raise ValueError(
                    f"cached DB has {cached_documents} source documents; "
                    f"{len(ratios)} are required by the configured split"
                )
        except (OSError, ValueError) as exc:
            log_event("warn", f"Cached database is not usable ({exc}); rebuilding from source")
        else:
            command = [
                sys.executable, str(SCRIPT_DIR / "data_pipeline.py"),
                "--corpus", corpus, "--in", str(database_path), "--out", str(output_root),
                "--split-ratios", *(str(ratio) for ratio in ratios), "--seed", str(seed),
            ]
            for ignored in ignore_files:
                command.extend(("--ignore-file", ignored))
            if rare_check:
                command.append("--rare-check")
            if verbose:
                command.append("--verbose")
            run_pipeline(command, f"{corpus}: split cached database")
            log_event("ok", f"Data preparation finished in {time.perf_counter() - started_at:.1f}s")
            return 0
    elif use_cached:
        log_event("info", "No cached database found; resolving a local or downloadable source")
    else:
        log_event("info", "Cache disabled; resolving a local or downloadable source")

    if manifest_entry.get("download", {}).get("type") == "local":
        extract_value = manifest_entry.get("extract-path")
        if isinstance(extract_value, str) and extract_value.strip():
            extract_dir = local_extract_path(extract_value, PROJECT_DIR)
            corpus_output = (output_root / corpus).resolve()
            if (corpus_output == extract_dir or corpus_output.is_relative_to(extract_dir)
                    or extract_dir.is_relative_to(corpus_output)):
                raise ValueError(
                    "Local archive extract-path overlaps the processed output directory. "
                    "Choose an extract-path outside data.output, such as data/raw/extracted/ptb."
                )

    source_path = _manifest_source(corpus, manifest_path, raw_root)
    if source_path is not None:
        command = command_for_pipeline(
            corpus, output_root, ratios, seed, source_path, clean, rare_check, verbose,
            ignore_files, include_extensionless, remove_words, remove_regex,
        )
        print(f"Using raw source: {display_path(source_path)}", flush=True)
        run_pipeline(command, f"{corpus}: prepare local source")
        log_event("ok", f"Data preparation finished in {time.perf_counter() - started_at:.1f}s")
        return 0

    # No local source is available. Let data_pipeline download through the same
    # manifest, extract/resolve the package, then load, split, and analyze it.
    command = [
        sys.executable, str(SCRIPT_DIR / "data_pipeline.py"),
        "--corpus", corpus, "--in", str(manifest_path),
        "--download", "--download-out", str(raw_root), "--out", str(output_root),
    ]
    if clean:
        command.append("--clean")
        for word in remove_words:
            command.extend(("--remove-word", word))
        for pattern in remove_regex:
            command.extend(("--remove-regex", pattern))
    if include_extensionless:
        command.append("--include-extensionless")
    for ignored in ignore_files:
        command.extend(("--ignore-file", ignored))
    command.extend(("--load", "--split-ratios", *(str(ratio) for ratio in ratios), "--seed", str(seed)))
    if rare_check:
        command.append("--rare-check")
    if verbose:
        command.append("--verbose")
    run_pipeline(command, f"{corpus}: download and prepare")
    log_event("ok", f"Data preparation finished in {time.perf_counter() - started_at:.1f}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the configured corpus using cached DB, local raw data, or manifest download.")
    parser.add_argument("--config", type=Path, required=True, help="Model/data YAML with a top-level data: section.")
    parser.add_argument("--manifest", type=Path, default=None, help="Override the registry path; otherwise use data.manifest or src/data-pipeline/data.yaml.")
    parser.add_argument("--seed", "--split-seed", type=int, default=None, help="Override the configured split_seed for this run.")
    rare_group = parser.add_mutually_exclusive_group()
    rare_group.add_argument("--rare-check", dest="rare_check", action="store_true", help="Run rare n-gram analysis even if disabled in the config.")
    rare_group.add_argument("--no-rare-check", dest="rare_check", action="store_false", help="Skip rare n-gram analysis for this run.")
    parser.set_defaults(rare_check=None)
    parser.add_argument("--verbose", action="store_true", help="Show per-document processing and split samples.")
    args = parser.parse_args()
    try:
        return run(args)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        log_event("error", str(exc))
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
