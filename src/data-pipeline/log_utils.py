"""Small helpers for concise, project-relative pipeline logs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def display_path(value: str | Path | None) -> str:
    """Prefer a project-relative path; keep external paths to their basename."""
    if value is None:
        return "not requested"
    path = Path(value).expanduser()
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return path.name or str(value)
