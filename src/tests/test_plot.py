from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from model_comparison import format_accuracy_percent
from plot import choose_output_tag, find_metrics_csv, load_metrics, regenerate_plots


FIELDS = [
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
]


def write_metrics(path: Path, corpus: str = "tiny") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({
            "corpus": corpus,
            "percent": "100.0",
            "model": "bigram_backoff",
            "seed": "",
            "source": "test",
            "order": "2",
            "accuracy": "0.127747",
            "perplexity": "123.4",
            "status": "finite",
            "prediction_events": "110",
            "word_tokens": "100",
            "correct_predictions": "13",
        })


class PlotInputTests(unittest.TestCase):
    def test_resolves_result_root_and_metrics_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            metrics_path = root / "metrics" / "metrics.csv"
            write_metrics(metrics_path)

            self.assertEqual(find_metrics_csv(root), metrics_path.resolve())
            self.assertEqual(find_metrics_csv(metrics_path.parent), metrics_path.resolve())

    def test_rejects_missing_metrics_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No metrics.csv found"):
                find_metrics_csv(Path(directory))

    def test_loads_one_valid_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            write_metrics(path, corpus="tiny")
            rows, corpus = load_metrics(path)

            self.assertEqual(corpus, "tiny")
            self.assertEqual(len(rows), 1)

    def test_timestamp_collision_gets_numeric_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_dir = Path(directory)
            when = datetime(2026, 11, 7, 12, 34, 56)
            self.assertEqual(choose_output_tag(metrics_dir, when), "20261107_123456")

            (metrics_dir / "charts_20261107_123456.png").touch()
            self.assertEqual(choose_output_tag(metrics_dir, when), "20261107_123456_01")

    def test_accuracy_formatter_uses_percentage_units(self):
        self.assertEqual(format_accuracy_percent(0.127747), "12.8%")

    def test_regeneration_preserves_existing_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            metrics_dir = root / "metrics"
            write_metrics(metrics_dir / "metrics.csv")
            existing_chart = metrics_dir / "charts.png"
            existing_percent = metrics_dir / "plot_100p.png"
            existing_chart.write_bytes(b"original chart")
            existing_percent.write_bytes(b"original percentage plot")

            outputs = regenerate_plots(root)

            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertRegex(outputs[0].name, r"^charts_\d{8}_\d{6}(?:_\d{2})?\.png$")
            self.assertRegex(outputs[1].name, r"^plot_100p_\d{8}_\d{6}(?:_\d{2})?\.png$")
            self.assertEqual(existing_chart.read_bytes(), b"original chart")
            self.assertEqual(existing_percent.read_bytes(), b"original percentage plot")


if __name__ == "__main__":
    unittest.main()
