"""Helpers for identifying and validating shelve database files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from log_utils import display_path, log_event


def database_files(database_path: Path) -> list[Path]:
    """Return the backing file(s) for a shelve database, across dbm backends."""
    if not database_path.parent.exists():
        return []
    candidates = [
        path for path in database_path.parent.iterdir()
        if path.is_file() and (
            path.name == database_path.name
            or path.name.startswith(database_path.name + ".")
        ) and path != Path(str(database_path) + ".sha")
    ]
    return sorted(candidates)


def database_sha256(database_path: Path) -> str:
    files = database_files(database_path)
    if not files:
        raise FileNotFoundError(f"No database files found for: {display_path(database_path)}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_database_sha(database_path: Path) -> Path:
    sidecar = Path(str(database_path) + ".sha")
    checksum = database_sha256(database_path)
    sidecar.write_text(f"{checksum}  {database_path.name}\n", encoding="ascii")
    log_event("ok", f"Wrote database SHA-256 sidecar: {display_path(sidecar)}")
    return sidecar


def verify_database_sha(database_path: Path, require_sidecar: bool = True) -> bool:
    sidecar = Path(str(database_path) + ".sha")
    log_event("check", f"Verifying database SHA-256: {display_path(database_path)}")
    if not sidecar.is_file():
        if require_sidecar:
            log_event("error", f"Database SHA-256 sidecar is missing: {display_path(sidecar)}")
            raise ValueError(f"Database checksum is missing: {display_path(sidecar)}")
        log_event("skip", "No checksum sidecar; verification was optional")
        return False
    fields = sidecar.read_text(encoding="ascii").split()
    if not fields:
        log_event("error", f"Database SHA-256 sidecar is empty: {display_path(sidecar)}")
        raise ValueError(f"Database checksum file is empty: {display_path(sidecar)}")
    expected = fields[0].casefold()
    actual = database_sha256(database_path)
    if expected != actual.casefold():
        log_event("error", f"Database SHA-256 mismatch: {display_path(database_path)}")
        raise ValueError(
            f"Database checksum mismatch for {display_path(database_path)}: "
            f"expected {expected}, got {actual}"
        )
    log_event("ok", f"Database SHA-256 verified: {display_path(database_path)}")
    return True
