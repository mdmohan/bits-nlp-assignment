"""Download one NLTK corpus or a file from a direct URL."""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.request
import hashlib
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import nltk
try:
    import yaml
except ImportError:  # pragma: no cover - user environment dependency check
    yaml = None


SCRIPT_DIR = Path(__file__).resolve().parent
from log_utils import display_path, log_event


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


def configured_string_list(entry: dict, key: str) -> list[str]:
    """Read an optional manifest array containing only non-empty strings."""
    value = entry.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"Corpus manifest field '{key}' must be an array of non-empty strings")
    return value


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


def safe_extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Safely extract ZIP or tar archives, rejecting traversal and links."""
    if zipfile.is_zipfile(archive_path):
        safe_extract_zip(archive_path, target_dir)
        return
    if not tarfile.is_tarfile(archive_path):
        raise ValueError(f"Unsupported archive format (expected ZIP or tar): {archive_path.name}")

    target_root = target_dir.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name.replace("\\", "/"))
            destination = (target_root / member_path).resolve()
            if (member_path.is_absolute() or ".." in member_path.parts
                    or not destination.is_relative_to(target_root)):
                raise ValueError(f"Unsafe path in corpus archive: {member.name}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"Links and device files are not allowed in corpus archives: {member.name}")
        archive.extractall(target_root)


def local_archive_path(package: str, project_root: Path, raw_root: Path | None = None) -> Path:
    """Resolve a local archive path relative to the project (or raw root)."""
    path = Path(package).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_path = (project_root / path).resolve()
    if project_path.is_file() or raw_root is None:
        return project_path
    raw_path = (raw_root / path).resolve()
    return raw_path if raw_path.is_file() else project_path


def local_extract_path(value: str, project_root: Path) -> Path:
    """Resolve local archive extraction path; relative paths use project root."""
    path = Path(value).expanduser()
    if not path.is_absolute() and ".." in path.parts:
        raise ValueError(f"Unsafe corpus extract-path in manifest: {value!r}")
    return (path if path.is_absolute() else project_root / path).resolve()


def extracted_source_folder(extract_dir: Path) -> Path:
    """Return the sole top-level directory when an archive has one, else root."""
    if not extract_dir.is_dir():
        raise FileNotFoundError(f"Archive extraction directory was not created: {extract_dir}")
    entries = list(extract_dir.iterdir())
    directories = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    if len(directories) == 1 and not files:
        return directories[0]
    return extract_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download an NLTK corpus or URL file, or extract a configured local archive. "
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
        started_at = time.perf_counter()
        log_event("info", f"Download stage started: corpus={corpus}, source={source_type}")
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
                # NLTK corpus ZIPs contain a top-level package directory (for
                # example, `reuters/...`), so extract into the configured
                # corpus directory's parent (`raw/corpora`), not `raw`.
                safe_extract_zip(package_zip, package_dir.parent)
            if package_dir.is_dir():
                log_event("skip", f"NLTK package '{package}' already extracted at {display_path(package_dir)}")
            else:
                succeeded = nltk.download(package, download_dir=str(output_root))
                if not succeeded:
                    raise ValueError(f"NLTK could not download corpus package: {package}")
                # Some NLTK download states leave the archive without unpacking it.
                if not package_dir.is_dir() and zipfile.is_zipfile(package_zip):
                    safe_extract_zip(package_zip, package_dir.parent)
                if not package_dir.is_dir():
                    raise FileNotFoundError(
                        f"NLTK download completed but the configured extract path was not found: {package_dir}"
                    )
                print(f"Downloaded NLTK corpus package '{package}' to: {display_path(package_dir)}")
        elif source_type == "local":
            package = download.get("package")
            extract_path = entry.get("extract-path")
            if not isinstance(package, str) or not package.strip():
                raise ValueError(f"Corpus '{corpus}' has local download type but no archive package path")
            if not isinstance(extract_path, str) or not extract_path.strip():
                raise ValueError(f"Corpus '{corpus}' has local download type but no extract-path")
            project_root = SCRIPT_DIR.parent.parent
            archive_path = local_archive_path(package, project_root, output_root)
            if not archive_path.is_file():
                raise FileNotFoundError(f"Local corpus archive was not found: {archive_path}")
            target_dir = local_extract_path(extract_path, project_root)
            safe_extract_archive(archive_path, target_dir)
            source_dir = extracted_source_folder(target_dir)
            print(f"Extracted local corpus archive to: {display_path(source_dir)}")
        else:
            raise ValueError(
                f"Corpus '{corpus}' has unsupported download type {source_type!r}; use 'nltk', 'url', or 'local'"
            )
    except (OSError, RuntimeError, urllib.error.URLError, ValueError) as exc:
        log_event("error", f"Download stage failed: {exc}")
        parser.error(str(exc))
    log_event("ok", f"Download stage finished in {time.perf_counter() - started_at:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
