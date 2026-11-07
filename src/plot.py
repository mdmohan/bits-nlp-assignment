#!/usr/bin/env python3
"""Regenerate timestamped plots from an existing comparison metrics.csv.

Usage:
    python src/plot.py results/gutenberg_neural
    python src/plot.py results/gutenberg_neural/metrics

The command reads aggregate metrics only. It does not load models, evaluate
text, or overwrite existing plots.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


REQUIRED_COLUMNS = {
    "corpus",
    "percent",
    "model",
    "seed",
    "source",
    "order",
    "accuracy",
    "perplexity",
    "status",
    "prediction_events",
    "word_tokens",
    "correct_predictions",
}


def find_metrics_csv(folder: Path) -> Path:
    """Resolve a result root or metrics directory to one metrics.csv."""
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {folder}")

    candidates = [
        candidate
        for candidate in (folder / "metrics.csv", folder / "metrics" / "metrics.csv")
        if candidate.is_file()
    ]
    if not candidates:
        raise ValueError(
            f"No metrics.csv found at {folder / 'metrics.csv'} "
            f"or {folder / 'metrics' / 'metrics.csv'}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Folder is ambiguous because it contains both a direct and nested metrics.csv: "
            f"{folder}"
        )
    return candidates[0]


def load_metrics(path: Path) -> tuple[list[dict[str, str]], str]:
    """Load and validate one corpus's aggregate comparison metrics."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Metrics file contains no data rows: {path}")
    corpora = sorted({row["corpus"].strip() for row in rows if row["corpus"].strip()})
    if len(corpora) != 1:
        detail = ", ".join(corpora) if corpora else "none"
        raise ValueError(f"Expected exactly one corpus in {path}; found: {detail}")
    return rows, corpora[0]


def choose_output_tag(metrics_dir: Path, now: datetime | None = None) -> str:
    """Return a timestamp tag that does not collide with an existing PNG."""
    base = (now or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    tag = base
    counter = 1
    while any(metrics_dir.glob(f"*_{tag}.png")):
        tag = f"{base}_{counter:02d}"
        counter += 1
    return tag


def regenerate_plots(folder: Path) -> list[Path]:
    """Regenerate plots beside metrics.csv without replacing existing files."""
    metrics_path = find_metrics_csv(folder)
    metrics, corpus = load_metrics(metrics_path)
    metrics_dir = metrics_path.parent
    tag = choose_output_tag(metrics_dir)
    suffix = f"_{tag}"
    chart_path = metrics_dir / f"charts{suffix}.png"

    # Keep the plotting implementation shared with the comparison pipeline.
    # Import lazily so path/CSV validation remains lightweight.
    from model_comparison import plot_metrics, plot_percentage_metrics

    plot_metrics(metrics, chart_path, corpus)
    percentage_paths = plot_percentage_metrics(
        metrics,
        metrics_dir,
        corpus,
        filename_suffix=suffix,
    )
    outputs = [chart_path, *percentage_paths]
    missing_outputs = [path for path in outputs if not path.is_file()]
    if missing_outputs:
        raise RuntimeError(
            "Plot generation did not create: "
            + ", ".join(str(path) for path in missing_outputs)
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        help="result folder containing metrics/metrics.csv, or the metrics folder itself",
    )
    args = parser.parse_args(argv)

    try:
        outputs = regenerate_plots(args.folder)
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        parser.error(str(exc))

    print(f"Generated {len(outputs)} timestamped plot(s); existing files were preserved:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
