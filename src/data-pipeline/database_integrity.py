"""Helpers for identifying and validating shelve database files."""

from __future__ import annotations

import hashlib
from pathlib import Path
from log_utils import display_path


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
    return sidecar


def verify_database_sha(database_path: Path, require_sidecar: bool = True) -> bool:
    sidecar = Path(str(database_path) + ".sha")
    if not sidecar.is_file():
        if require_sidecar:
            raise ValueError(f"Database checksum is missing: {display_path(sidecar)}")
        return False
    fields = sidecar.read_text(encoding="ascii").split()
    if not fields:
        raise ValueError(f"Database checksum file is empty: {display_path(sidecar)}")
    expected = fields[0].casefold()
    actual = database_sha256(database_path)
    if expected != actual.casefold():
        raise ValueError(
            f"Database checksum mismatch for {display_path(database_path)}: "
            f"expected {expected}, got {actual}"
        )
    return True
