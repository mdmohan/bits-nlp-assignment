from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from model_comparison import (
    METRIC_FIELDS,
    PREDICTION_FIELDS,
    add_prediction_stats,
    checkpoint_identity,
    empty_prediction_stats,
    load_resumable_metrics,
    merge_prediction_exports,
    metric_row_valid,
    prediction_group_valid,
    prediction_digest,
    prediction_key,
    rewrite_prediction_journal,
    select_prediction_source,
)
from run import comparison_command


def metric_row(model: str, source: str, order: str, correct: int = 1) -> dict:
    row = {field: "" for field in METRIC_FIELDS}
    row.update({
        "corpus": "tiny",
        "percent": "20.0",
        "model": model,
        "family": "ngram" if "gram" in model else "neural",
        "order": order,
        "seed": "" if "gram" in model else "13",
        "source": source,
        "sentences": "1",
        "word_tokens": "2",
        "prediction_events": "3",
        "correct_predictions": str(correct),
        "accuracy": str(correct / 2),
        "perplexity": "4.0",
        "status": "finite",
        "zero_probability_events": "0",
        "oov_tokens": "0",
        "oov_rate": "0.0",
        "evaluation_seconds": "1.0",
        "train_seconds": "0.5",
        "prediction_rows": "1",
        "prediction_weight": "1",
        "prediction_digest": "0" * 64,
    })
    return row


def prediction_row(
    model: str,
    source: str,
    correctness: str,
    *,
    seed: str = "",
    rare_file: str = "",
) -> dict:
    return {
        "percent": "20.0",
        "model": model,
        "seed": seed,
        "source": source,
        "context": "a",
        "expected": "b",
        "prediction": "b" if correctness == "correct" else "c",
        "correctness": correctness,
        "weight": "1",
        "rare_file": rare_file,
    }


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def attach_prediction_stats(row: dict, predictions: list[dict]) -> None:
    stats = empty_prediction_stats()
    for prediction in predictions:
        add_prediction_stats(stats, prediction, max(1, int(prediction.get("weight") or 1)))
    row["prediction_rows"] = str(stats["rows"])
    row["prediction_weight"] = str(stats["weight"])
    row["prediction_digest"] = prediction_digest(stats)


class ModelComparisonResumeTests(unittest.TestCase):
    def test_duplicate_lstm_folder_keeps_distinct_label(self):
        label, seed, order = checkpoint_identity(Path("models/lstm_2/lstm_seed13.pt"))
        self.assertEqual((label, seed, order), ("lstm_2", "13", ""))

    def test_prediction_key_distinguishes_lstm_rare_orders(self):
        bigram = prediction_row(
            "lstm", "rare", "correct", seed="13", rare_file="rare-seq-bigram.csv"
        )
        trigram = prediction_row(
            "lstm", "rare", "correct", seed="13", rare_file="rare-seq-trigram.csv"
        )
        self.assertEqual(prediction_key(bigram)[-1], "2")
        self.assertEqual(prediction_key(trigram)[-1], "3")
        self.assertNotEqual(prediction_key(bigram), prediction_key(trigram))

    def test_largest_full_prediction_backup_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rows(
                root / "predictions.csv", PREDICTION_FIELDS,
                [prediction_row("bigram_mle", "test", "correct")],
            )
            write_rows(
                root / "predictions-002.csv", PREDICTION_FIELDS,
                [
                    prediction_row("bigram_mle", "test", "correct"),
                    prediction_row("bigram_mle", "test", "incorrect"),
                ],
            )
            write_rows(
                root / "predictions_incorrect-003.csv", PREDICTION_FIELDS,
                [prediction_row("bigram_mle", "test", "incorrect")] * 10,
            )
            self.assertEqual(select_prediction_source(root).name, "predictions-002.csv")

    def test_only_count_valid_metric_rows_are_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            valid = metric_row("bigram_mle", "test", "2")
            invalid = metric_row("bigram_mle", "valid", "2")
            invalid["word_tokens"] = "999"
            write_rows(path, METRIC_FIELDS, [valid, invalid])
            expected = {
                (20.0, "bigram_mle", "", "test", "2"):
                    {"sentences": 1, "word_tokens": 2, "prediction_events": 3},
                (20.0, "bigram_mle", "", "valid", "2"):
                    {"sentences": 1, "word_tokens": 2, "prediction_events": 3},
            }
            rows, origins, issues = load_resumable_metrics(
                [("canonical", path)], expected
            )
            self.assertEqual(set(rows), {(20.0, "bigram_mle", "", "test", "2")})
            self.assertEqual(set(origins.values()), {"canonical"})
            self.assertEqual(issues, [])

    def test_changed_evaluation_signature_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            row = metric_row("bigram_mle", "test", "2")
            row["evaluation_signature"] = "a" * 64
            write_rows(path, METRIC_FIELDS, [row])
            key = (20.0, "bigram_mle", "", "test", "2")
            totals = {key: {"sentences": 1, "word_tokens": 2, "prediction_events": 3}}
            rows, origins, issues = load_resumable_metrics(
                [("canonical", path)], totals, {key: "b" * 64}
            )
            self.assertEqual((rows, origins, issues), ({}, {}, []))

    def test_prediction_journal_drops_uncommitted_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".predictions.resume.csv"
            committed = prediction_row("bigram_mle", "test", "correct")
            partial = prediction_row("trigram_laplace", "valid", "incorrect")
            write_rows(path, PREDICTION_FIELDS, [committed, partial])
            rewrite_prediction_journal(path, {prediction_key(committed)})
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [committed])

    def test_partial_zero_correct_group_is_not_accepted(self):
        row = metric_row("bigram_mle", "test", "2", correct=0)
        row["prediction_rows"] = "10"
        row["prediction_weight"] = "10"
        partial = {"rows": 1, "weight": 1, "correct": 0, "invalid": 0}
        self.assertFalse(prediction_group_valid(row, partial))

    def test_same_aggregate_but_different_prediction_content_is_rejected(self):
        expected_prediction = prediction_row("bigram_mle", "test", "correct")
        row = metric_row("bigram_mle", "test", "2", correct=1)
        attach_prediction_stats(row, [expected_prediction])
        changed_prediction = dict(expected_prediction, context="different context")
        observed = empty_prediction_stats()
        add_prediction_stats(observed, changed_prediction, 1)
        self.assertFalse(prediction_group_valid(row, observed))

    def test_train_metric_allows_correct_predictions_without_export_rows(self):
        row = metric_row("bigram_mle", "train", "2", correct=1)
        row["prediction_rows"] = "0"
        row["prediction_weight"] = "0"
        expected = {"sentences": 1, "word_tokens": 2, "prediction_events": 3}
        self.assertTrue(metric_row_valid(row, expected))

    def test_legitimate_empty_prediction_group_validates(self):
        row = metric_row("bigram_mle", "test", "2", correct=0)
        attach_prediction_stats(row, [])
        self.assertTrue(prediction_group_valid(row, None))

    def test_merge_reuses_base_and_journal_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "predictions-002.csv"
            journal = root / ".predictions.resume.csv"
            base_row = prediction_row("bigram_mle", "test", "correct")
            journal_row = prediction_row("trigram_laplace", "valid", "incorrect")
            write_rows(base, PREDICTION_FIELDS, [base_row])
            write_rows(journal, PREDICTION_FIELDS, [journal_row])
            output = {
                "predictions": root / "predictions.csv",
                "correct": root / "predictions_correct.csv",
                "incorrect": root / "predictions_incorrect.csv",
            }
            base_key = prediction_key(base_row)
            journal_key = prediction_key(journal_row)
            rows = {
                base_key: metric_row("bigram_mle", "test", "2", correct=1),
                journal_key: metric_row("trigram_laplace", "valid", "3", correct=0),
            }
            attach_prediction_stats(rows[base_key], [base_row])
            attach_prediction_stats(rows[journal_key], [journal_row])
            merge_prediction_exports(
                base, {base_key}, journal, {journal_key}, output, rows
            )
            with output["predictions"].open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with output["correct"].open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)
            with output["incorrect"].open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 1)

    def test_failed_merge_preserves_canonical_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "predictions-002.csv"
            journal = root / ".predictions.resume.csv"
            base_row = prediction_row("bigram_mle", "test", "incorrect")
            write_rows(base, PREDICTION_FIELDS, [base_row])
            write_rows(journal, PREDICTION_FIELDS, [])
            output = {
                "predictions": root / "predictions.csv",
                "correct": root / "predictions_correct.csv",
                "incorrect": root / "predictions_incorrect.csv",
            }
            for path in output.values():
                path.write_text("surviving evidence\n", encoding="utf-8")
            key = prediction_key(base_row)
            expected = metric_row("bigram_mle", "test", "2", correct=0)
            expected["prediction_rows"] = "2"
            expected["prediction_weight"] = "2"
            with self.assertRaises(ValueError):
                merge_prediction_exports(base, {key}, journal, set(), output, {key: expected})
            for path in output.values():
                self.assertEqual(path.read_text(encoding="utf-8"), "surviving evidence\n")
                self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_full_resume_passes_flag_to_comparison(self):
        comparison = Path("src/model_comparison.py")
        self.assertEqual(comparison_command(comparison, "reuters", False)[-1], "reuters")
        self.assertEqual(comparison_command(comparison, "reuters", True)[-1], "--resume")
        config = Path("src/configs/reuters.yaml")
        self.assertEqual(
            comparison_command(comparison, "reuters", False, config)[-2:],
            ["--config", str(config)],
        )
        self.assertEqual(
            comparison_command(comparison, "reuters", True, config)[-3:],
            ["--config", str(config), "--resume"],
        )


if __name__ == "__main__":
    unittest.main()
