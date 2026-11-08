from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from plot_paper import (
    Bundle,
    CORPUS_SPECS,
    MODEL_ORDER,
    PERCENTAGES,
    accuracy_summary,
    choose_tag,
    complete_training_summary,
    generate,
    mean_sd,
)


METRICS_FIELDS = [
    "corpus",
    "percent",
    "model",
    "seed",
    "source",
    "accuracy",
    "perplexity",
    "status",
    "zero_probability_events",
    "word_tokens",
    "correct_predictions",
    "oov_tokens",
    "oov_rate",
    "evaluation_seconds",
    "train_seconds",
]

PREDICTION_FIELDS = [
    "percent",
    "model",
    "seed",
    "source",
    "context",
    "expected",
    "prediction",
    "correctness",
    "weight",
    "rare_file",
]


def metric_row(
    corpus: str,
    model: str,
    *,
    percent: int = 100,
    seed: str = "",
    correct: int = 4,
    words: int = 10,
    oov: int = 4,
    perplexity: str = "100",
    status: str = "finite",
    zero_probability_events: str = "0",
    evaluation_seconds: str = "2",
    train_seconds: str = "3",
) -> dict[str, str]:
    return {
        "corpus": corpus,
        "percent": str(float(percent)),
        "model": model,
        "seed": seed,
        "source": "test",
        "accuracy": str(correct / words),
        "perplexity": perplexity,
        "status": status,
        "zero_probability_events": zero_probability_events,
        "word_tokens": str(words),
        "correct_predictions": str(correct),
        "oov_tokens": str(oov),
        "oov_rate": str(oov / words),
        "evaluation_seconds": evaluation_seconds,
        "train_seconds": train_seconds,
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_bundle_fixture(results_root: Path, result_name: str, corpus: str) -> None:
    root = results_root / result_name
    rows: list[dict[str, str]] = []
    for percent in PERCENTAGES:
        for index, model in enumerate(MODEL_ORDER):
            seeds = ("13",) if model.startswith("ffnn_") or model == "lstm" else ("",)
            for seed in seeds:
                if model.endswith("_mle"):
                    ppl = ""
                    status = "infinite_zero_probability"
                    zeros = "2"
                else:
                    ppl = str(50 + percent + index)
                    status = "finite"
                    zeros = "0"
                rows.append(
                    metric_row(
                        corpus,
                        model,
                        percent=percent,
                        seed=seed,
                        perplexity=ppl,
                        status=status,
                        zero_probability_events=zeros,
                        evaluation_seconds=str(1 + index / 10),
                        train_seconds=str(2 + index / 10),
                    )
                )
    write_csv(root / "metrics" / "metrics.csv", METRICS_FIELDS, rows)

    for percent in PERCENTAGES:
        stats = {
            "vocab_size": 1_000 + percent,
            "valid_oov_rate": 0.20 - percent / 1_000,
            "test_oov_rate": 0.25 - percent / 1_000,
            "train_percent": float(percent),
        }
        path = root / f"train_{percent}pct" / "data_stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats), encoding="utf-8")


class PaperPlotAggregationTests(unittest.TestCase):
    def test_mean_sd_uses_sample_standard_deviation(self):
        mean, sample_sd = mean_sd([100.0, 200.0, 300.0])
        self.assertEqual(mean, 200.0)
        self.assertEqual(sample_sd, 100.0)

        mean, sample_sd = mean_sd([42.0])
        self.assertEqual(mean, 42.0)
        self.assertIsNone(sample_sd)

    def test_retained_accuracy_streams_weighted_correct_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tiny-result"
            metrics_path = root / "metrics" / "metrics.csv"
            metrics: list[dict[str, str]] = []
            for model in MODEL_ORDER:
                seed = "13" if model.startswith("ffnn_") or model == "lstm" else ""
                metrics.append(
                    metric_row(
                        "tiny",
                        model,
                        seed=seed,
                        correct=8 if model == "bigram_mle" else 4,
                    )
                )
            write_csv(metrics_path, METRICS_FIELDS, metrics)

            predictions = [
                {
                    "percent": "100.0",
                    "model": "bigram_mle",
                    "seed": "",
                    "source": "test",
                    "context": "quoted, context",
                    "expected": "<unk>",
                    "prediction": "<unk>",
                    "correctness": "correct",
                    "weight": "2",
                    "rare_file": "",
                },
                {
                    "percent": "100",
                    "model": "bigram_mle",
                    "seed": "",
                    "source": "test",
                    "context": "x",
                    "expected": "<unk>",
                    "prediction": "<unk>",
                    "correctness": "correct",
                    "weight": "1",
                    "rare_file": "",
                },
                # These rows must not contribute to full-budget test OOV accuracy.
                {
                    "percent": "100",
                    "model": "bigram_mle",
                    "seed": "",
                    "source": "valid",
                    "context": "x",
                    "expected": "<unk>",
                    "prediction": "<unk>",
                    "correctness": "correct",
                    "weight": "99",
                    "rare_file": "",
                },
                {
                    "percent": "100",
                    "model": "bigram_mle",
                    "seed": "",
                    "source": "rare",
                    "context": "x",
                    "expected": "<unk>",
                    "prediction": "<unk>",
                    "correctness": "correct",
                    "weight": "99",
                    "rare_file": "rare.csv",
                },
                {
                    "percent": "20",
                    "model": "bigram_mle",
                    "seed": "",
                    "source": "test",
                    "context": "x",
                    "expected": "<unk>",
                    "prediction": "<unk>",
                    "correctness": "correct",
                    "weight": "99",
                    "rare_file": "",
                },
                {
                    "percent": "100",
                    "model": "bigram_mle",
                    "seed": "",
                    "source": "test",
                    "context": "x",
                    "expected": "</s>",
                    "prediction": "</s>",
                    "correctness": "correct",
                    "weight": "99",
                    "rare_file": "",
                },
            ]
            write_csv(
                root / "metrics" / "predictions_correct.csv",
                PREDICTION_FIELDS,
                predictions,
            )
            bundle = Bundle(
                display_name="Tiny",
                result_name="tiny-result",
                root=root,
                metrics_path=metrics_path,
                metrics=tuple(metrics),
                stats={},
            )

            summary, rows_read = accuracy_summary(bundle, progress_seconds=0)

            self.assertEqual(rows_read, len(predictions))
            self.assertAlmostEqual(summary["always_unk_baseline"], 0.4)
            bigram = summary["models"]["bigram_mle"]
            self.assertAlmostEqual(bigram["overall_mean"], 0.8)
            self.assertAlmostEqual(bigram["retained_mean"], 5 / 6)
            self.assertEqual(bigram["seeds"][0]["correct_oov"], 3)
            self.assertEqual(bigram["seeds"][0]["correct_retained"], 5)

    def test_complete_training_summary_retains_infinite_mle_and_neural_sd(self):
        metrics: list[dict[str, str]] = []
        for model in MODEL_ORDER:
            if model.endswith("_mle"):
                metrics.append(
                    metric_row(
                        "tiny",
                        model,
                        perplexity="",
                        status="infinite_zero_probability",
                        zero_probability_events="2",
                    )
                )
                continue
            seeds = ("13", "42", "2026") if model == "lstm" else ("",)
            for index, seed in enumerate(seeds, start=1):
                metrics.append(
                    metric_row(
                        "tiny",
                        model,
                        seed=seed,
                        correct=index * 2,
                        perplexity=str(index * 100),
                    )
                )

        bundle = Bundle(
            display_name="Tiny",
            result_name="tiny-result",
            root=Path("tiny-result"),
            metrics_path=Path("tiny-result/metrics/metrics.csv"),
            metrics=tuple(metrics),
            stats={},
        )
        summary = complete_training_summary(bundle)

        self.assertTrue(summary["models"]["bigram_mle"]["perplexity_infinite"])
        self.assertIsNone(summary["models"]["bigram_mle"]["perplexity_mean"])
        self.assertAlmostEqual(summary["models"]["lstm"]["accuracy_mean"], 0.4)
        self.assertAlmostEqual(
            summary["models"]["lstm"]["accuracy_sample_sd"], 0.2
        )
        self.assertAlmostEqual(summary["models"]["lstm"]["perplexity_mean"], 200.0)
        self.assertAlmostEqual(
            summary["models"]["lstm"]["perplexity_sample_sd"], 100.0
        )

    def test_choose_tag_refuses_to_overwrite_requested_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self.assertEqual(choose_tag(output_dir, "fixed"), "fixed")
            existing = output_dir / "paper_test_perplexity_fixed.png"
            existing.write_bytes(b"original")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                choose_tag(output_dir, "fixed")
            self.assertEqual(existing.read_bytes(), b"original")


class PaperPlotIntegrationTests(unittest.TestCase):
    def test_skip_accuracy_generates_five_compact_figures_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            results_root = temporary / "results"
            output_dir = temporary / "figures"
            for display, result_name in CORPUS_SPECS:
                corpus = display.lower().replace("stories", "-stories")
                write_bundle_fixture(results_root, result_name, corpus)

            old_mplconfigdir = os.environ.get("MPLCONFIGDIR")
            os.environ["MPLCONFIGDIR"] = str(temporary / "mplconfig")
            try:
                outputs = generate(
                    results_root,
                    output_dir,
                    tag="fixture",
                    skip_accuracy=True,
                    progress_seconds=0,
                )
            finally:
                if old_mplconfigdir is None:
                    os.environ.pop("MPLCONFIGDIR", None)
                else:
                    os.environ["MPLCONFIGDIR"] = old_mplconfigdir

            self.assertEqual(len(outputs), 11)
            pngs = sorted(path for path in outputs if path.suffix == ".png")
            pdfs = sorted(path for path in outputs if path.suffix == ".pdf")
            records = [path for path in outputs if path.suffix == ".json"]
            self.assertEqual(len(pngs), 5)
            self.assertEqual(len(pdfs), 5)
            self.assertEqual(len(records), 1)
            for path in pngs:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            for path in pdfs:
                self.assertEqual(path.read_bytes()[:4], b"%PDF")

            record = json.loads(records[0].read_text(encoding="utf-8"))
            self.assertTrue(record["accuracy_skipped"])
            self.assertEqual(record["tag"], "fixture")
            self.assertEqual(len(record["corpus_order"]), 6)
            self.assertNotIn("accuracy", record["aggregates"])
            self.assertEqual(
                set(record["aggregates"]),
                {
                    "complete_training",
                    "event_categories",
                    "test_perplexity",
                    "vocabulary_oov",
                    "runtime",
                },
            )
            self.assertEqual(
                record["selected_plots"],
                [
                    "complete-training",
                    "event-categories",
                    "test-perplexity",
                    "vocabulary-oov",
                    "runtime",
                ],
            )
            self.assertEqual(
                record["plot_functions"]["complete-training"],
                "plot_complete_training",
            )
            self.assertEqual(
                record["plot_functions"]["event-categories"],
                "plot_event_category_contributions",
            )
            self.assertEqual(
                record["aggregates"]["event_categories"]["corpora"]["WebText"]["total"],
                0.42605,
            )


if __name__ == "__main__":
    unittest.main()
