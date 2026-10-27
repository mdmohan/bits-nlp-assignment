"""Small helpers for concise, project-relative pipeline logs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def log_event(level: str, message: str) -> None:
    """Print a short, consistently labeled pipeline status line."""
    print(f"[{level.upper()}] {message}", flush=True)


def display_path(value: str | Path | None) -> str:
    """Prefer a project-relative path; keep external paths to their basename."""
    if value is None:
        return "not requested"
    path = Path(value).expanduser()
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except (OSError, ValueError):
        return path.name or str(value)
