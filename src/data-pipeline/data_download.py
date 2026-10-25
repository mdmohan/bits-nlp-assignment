"""Download one NLTK corpus or a file from a direct URL."""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
import hashlib
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import nltk
try:
    import yaml
except ImportError:  # pragma: no cover - user environment dependency check
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
from log_utils import display_path


def download_url(
    url: str,
    corpus: str,
    output_root: Path,
    filename: str | None = None,
    sha256: str | None = None,
) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--url must be an http:// or https:// URL")

    filename = filename or Path(unquote(parsed.path)).name or corpus
    corpus_dir = output_root / "corpora" / corpus
    corpus_dir.mkdir(parents=True, exist_ok=True)
    destination = corpus_dir / filename
    partial = destination.with_name(destination.name + ".part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NLP-assignment-corpus-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        if sha256:
            digest = hashlib.sha256(partial.read_bytes()).hexdigest()
            if digest.casefold() != sha256.casefold():
                raise ValueError(
                    f"SHA-256 mismatch for {filename}: expected {sha256}, got {digest}"
                )
        partial.replace(destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    return destination


def load_corpus_config(config_path: Path, corpus: str) -> dict:
    if yaml is None:
        raise RuntimeError("Reading data.yaml requires PyYAML; install it with: python -m pip install PyYAML")
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file does not exist: {config_path}") from None
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config {config_path}: {exc}") from exc

    if not isinstance(config, dict) or not isinstance(config.get("corpora"), dict):
        raise ValueError("Config must contain a top-level 'corpora' mapping")
    if corpus not in config["corpora"]:
        choices = ", ".join(sorted(config["corpora"]))
        raise ValueError(f"Unknown corpus '{corpus}'. Configured corpora: {choices}")

    entry = config["corpora"][corpus]
    if not isinstance(entry, dict) or not isinstance(entry.get("download"), dict):
        raise ValueError(f"Corpus '{corpus}' must define a 'download' mapping")
    return entry


def configured_extract_path(entry: dict, package: str | None = None) -> Path:
    """Return a safe manifest path to the extracted corpus, relative to raw root."""
    value = entry.get("extract-path")
    if value is None:
        value = f"corpora/{package}" if package else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Corpus manifest extract-path must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe corpus extract-path in manifest: {value!r}")
    return path


def safe_extract_zip(archive_path: Path, target_dir: Path) -> None:
    """Extract a ZIP after rejecting members that could escape the target root."""
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename.replace("\\", "/"))
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in corpus archive: {member.filename}")
            destination = (target_root / member_path).resolve()
            if not destination.is_relative_to(target_root):
                raise ValueError(f"Unsafe path in corpus archive: {member.filename}")
        archive.extractall(target_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a named NLTK corpus, or download one file from a direct URL. "
            "The output root defaults to data/raw."
        )
    )
    parser.add_argument(
        "corpus_name",
        nargs="?",
        help="Corpus key from the YAML config (for example: brown).",
    )
    parser.add_argument(
        "--corpus",
        dest="corpus_option",
        help="Corpus key (legacy equivalent of the positional name).",
    )
    parser.add_argument(
        "--conf",
        type=Path,
        default=SCRIPT_DIR / "data.yaml",
        help="YAML corpus registry (default: data.yaml beside this script).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Output root (default: data/raw).",
    )
    parser.add_argument(
        "--url",
        help="Override the configured source with a direct http(s) file URL.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.corpus_name and args.corpus_option and args.corpus_name != args.corpus_option:
        parser.error("positional corpus name and --corpus must match when both are supplied")
    corpus_value = args.corpus_name or args.corpus_option
    if not corpus_value:
        parser.error("provide a corpus name")
    corpus = corpus_value.strip()
    if not corpus or Path(corpus).name != corpus or corpus in {".", ".."}:
        parser.error("--corpus must be a simple name without path separators")

    config_path = args.conf.expanduser()
    if not config_path.is_absolute() and not config_path.exists():
        script_config_path = SCRIPT_DIR / config_path
        if script_config_path.exists():
            config_path = script_config_path
    config_path = config_path.resolve()
    output_root = args.out.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        entry = load_corpus_config(config_path, corpus)
        download = entry["download"]
        source_type = "url" if args.url else download.get("type")
        if source_type == "url":
            url = args.url or download.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError(f"Corpus '{corpus}' has URL download type but no URL")
            destination = download_url(
                url,
                corpus,
                output_root,
                filename=download.get("filename"),
                sha256=download.get("sha256"),
            )
            print(f"Downloaded file to: {display_path(destination)}")
        elif source_type == "nltk":
            package = download.get("package")
            if not isinstance(package, str) or not package:
                raise ValueError(f"Corpus '{corpus}' has NLTK download type but no package")
            package_dir = output_root / configured_extract_path(entry, package)
            package_zip = output_root / "corpora" / f"{package}.zip"
            if not package_dir.is_dir() and zipfile.is_zipfile(package_zip):
                safe_extract_zip(package_zip, output_root)
            if package_dir.is_dir():
                print(f"NLTK corpus package '{package}' is available at: {display_path(package_dir)}")
            else:
                succeeded = nltk.download(package, download_dir=str(output_root))
                if not succeeded:
                    raise ValueError(f"NLTK could not download corpus package: {package}")
                # Some NLTK download states leave the archive without unpacking it.
                if not package_dir.is_dir() and zipfile.is_zipfile(package_zip):
                    safe_extract_zip(package_zip, output_root)
                if not package_dir.is_dir():
                    raise FileNotFoundError(
                        f"NLTK download completed but the configured extract path was not found: {package_dir}"
                    )
                print(f"Downloaded NLTK corpus package '{package}' to: {display_path(package_dir)}")
        else:
            raise ValueError(
                f"Corpus '{corpus}' has unsupported download type {source_type!r}; use 'nltk' or 'url'"
            )
    except (OSError, RuntimeError, urllib.error.URLError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
