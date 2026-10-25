"""Run corpus preparation from a small model/data YAML configuration."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - environment dependency check
    yaml = None

from data_download import load_corpus_config
from data_pipeline import _safe_extract_zip
from database_integrity import database_files, verify_database_sha
from log_utils import display_path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_MANIFEST = SCRIPT_DIR / "data.yaml"


def run_pipeline(command: list[str], stage_summary: str) -> None:
    print(f"\n== {stage_summary} ==", flush=True)
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Pipeline failed during {stage_summary} (exit code {exc.returncode})") from exc


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
        package_dir = corpora_root / package
        archive_path = corpora_root / f"{package}.zip"
        if not package_dir.is_dir() and zipfile.is_zipfile(archive_path):
            _safe_extract_zip(archive_path, corpora_root)
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
            extracted = corpus_dir / "extracted"
            marker = corpus_dir / f".{filename}.extracted"
            if not marker.is_file() or marker.read_text(encoding="utf-8") != filename:
                extracted.mkdir(parents=True, exist_ok=True)
                _safe_extract_zip(archive_path, extracted)
                marker.write_text(filename, encoding="utf-8")
            return extracted
        return corpus_dir
    raise ValueError(f"Unsupported download type for '{corpus}': {kind!r}")


def command_for_pipeline(corpus: str, output_root: Path, ratios: list[float], seed: int,
                         input_path: Path | None, clean: bool, rare_check: bool,
                         verbose: bool) -> list[str]:
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
    command.extend(("--load", "--split-ratios", *(str(ratio) for ratio in ratios), "--seed", str(seed)))
    if rare_check:
        command.append("--rare-check")
    if verbose:
        command.append("--verbose")
    return command


def run(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    data = read_config(config_path)
    corpus = data.get("corpus", data.get("loader"))
    if not isinstance(corpus, str) or not corpus.strip() or Path(corpus).name != corpus:
        raise ValueError("data.corpus must be a simple configured corpus name")
    corpus = corpus.strip()
    if "split_ratio" not in data and "split_ratios" not in data:
        data = {**data, "split_ratio": data.get("params", {}).get("ratios", [0.8, 0.2])}
    ratios = split_ratios(data)
    seed = int(data.get("split_seed", data.get("params", {}).get("split_seed", 42)))
    use_cached = bool(data.get("use_cached", True))
    clean = bool(data.get("clean", False))
    rare_check = bool(data.get("rare-check", data.get("rare_check", False)))
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

    corpus_output = output_root / corpus
    database_path = corpus_output / f"{corpus}.db"
    if use_cached and database_files(database_path):
        try:
            verify_database_sha(database_path)
        except (OSError, ValueError) as exc:
            print(f"Cached database is not usable ({exc}); rebuilding.")
        else:
            command = [
                sys.executable, str(SCRIPT_DIR / "data_pipeline.py"),
                "--corpus", corpus, "--in", str(database_path), "--out", str(output_root),
                "--split-ratios", *(str(ratio) for ratio in ratios), "--seed", str(seed),
            ]
            if rare_check:
                command.append("--rare-check")
            if verbose:
                command.append("--verbose")
            run_pipeline(command, f"{corpus}: split cached database")
            return 0

    source_path = _manifest_source(corpus, manifest_path, raw_root)
    if source_path is not None:
        command = command_for_pipeline(corpus, output_root, ratios, seed, source_path, clean, rare_check, verbose)
        print(f"Using raw source: {display_path(source_path)}", flush=True)
        run_pipeline(command, f"{corpus}: prepare local source")
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
    command.extend(("--load", "--split-ratios", *(str(ratio) for ratio in ratios), "--seed", str(seed)))
    if rare_check:
        command.append("--rare-check")
    if verbose:
        command.append("--verbose")
    run_pipeline(command, f"{corpus}: download and prepare")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the configured corpus using cached DB, local raw data, or manifest download.")
    parser.add_argument("--config", type=Path, required=True, help="Model/data YAML with a top-level data: section.")
    parser.add_argument("--manifest", type=Path, default=None, help="Override the registry path; otherwise use data.manifest or src/data-pipeline/data.yaml.")
    parser.add_argument("--verbose", action="store_true", help="Show per-document processing and split samples.")
    args = parser.parse_args()
    try:
        return run(args)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
