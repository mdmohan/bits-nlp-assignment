#!/usr/bin/env python3
"""Audit whether ``<unk>`` matches inflate saved LM accuracy results.

This script reads existing ``metrics.csv`` and ``predictions.csv`` files. It
does not load models or run inference, and it uses only the Python standard
library so that corpus-sized prediction files can be streamed.

Examples:

    python src/analyze_unk_accuracy.py results/reuters_neural
    python src/analyze_unk_accuracy.py results/reuters_neural --source test rare
    python src/analyze_unk_accuracy.py results/reuters_neural --all-models
    python src/analyze_unk_accuracy.py results/reuters_neural --output unk-audit.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


UNK = "<unk>"
UNSEEN = "<unseen>"
DEFAULT_SOURCES = ("valid", "test", "rare")


@dataclass(frozen=True, order=True)
class Key:
    percent: float
    model: str
    seed: str
    source: str
    order: str

    @property
    def dataset(self) -> str:
        if self.source != "rare":
            return self.source
        suffix = {"2": "bigram", "3": "trigram"}.get(self.order, "")
        return f"rare:{suffix}" if suffix else "rare"


@dataclass
class Metric:
    word_tokens: int
    correct: int
    oov_tokens: int
    accuracy: float | None
    zero_events: int
    prediction_events: int


@dataclass
class PredictionCount:
    rows: int = 0
    events: int = 0
    correct: int = 0
    unk_correct: int = 0
    predicted_unk: int = 0
    iv_predicted_unk: int = 0
    predicted_unseen: int = 0
    correctness_mismatches: int = 0

    def add(self, expected: str, predicted: str, correctness: str, weight: int) -> None:
        matches = expected == predicted
        self.rows += 1
        self.events += weight
        self.correct += weight * int(matches)
        self.unk_correct += weight * int(expected == UNK and predicted == UNK)
        self.predicted_unk += weight * int(predicted == UNK)
        self.iv_predicted_unk += weight * int(expected != UNK and predicted == UNK)
        self.predicted_unseen += weight * int(predicted == UNSEEN)
        self.correctness_mismatches += int(
            (correctness.strip().casefold() == "correct") != matches
        )


def integer(value: str, field: str, line: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} at line {line}: {value!r}") from exc


def decimal(value: str, field: str, line: int) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field} at line {line}: {value!r}") from exc


def order_text(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(int(float(value)))
    except ValueError:
        return value.strip()


def infer_order(model: str, source: str, rare_file: str) -> str:
    model, rare_file = model.casefold(), rare_file.casefold()
    if source == "rare":
        if "bigram" in rare_file:
            return "2"
        if "trigram" in rare_file:
            return "3"
    if model.startswith("bigram_") or model.startswith("ffnn_ctx1"):
        return "2"
    if model.startswith("trigram_") or model.startswith("ffnn_ctx2"):
        return "3"
    return ""


def resolve_files(value: Path) -> tuple[Path, Path]:
    path = value.expanduser().resolve()
    if path.is_file():
        candidates = [(path, path.parent / "metrics.csv")]
    else:
        candidates = [
            (path / "metrics" / "predictions.csv", path / "metrics" / "metrics.csv"),
            (path / "predictions.csv", path / "metrics.csv"),
        ]
    for predictions, metrics in candidates:
        if predictions.is_file() and metrics.is_file():
            return predictions, metrics
    raise FileNotFoundError(
        f"Could not find predictions.csv and metrics.csv from {path}. Pass a "
        "results/<run> folder, its metrics folder, or predictions.csv."
    )


def wanted(percent: float, source: str, percentages: set[float] | None,
           sources: set[str]) -> bool:
    return source in sources and (percentages is None or percent in percentages)


def load_metrics(path: Path, sources: set[str], percentages: set[float] | None
                 ) -> tuple[dict[Key, Metric], list[Key]]:
    result: dict[Key, Metric] = {}
    duplicates = []
    required = {
        "percent", "model", "seed", "source", "order", "word_tokens",
        "correct_predictions", "accuracy", "oov_tokens",
        "zero_probability_events", "prediction_events",
    }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            percent = decimal(row["percent"], "percent", line)
            source = row["source"].strip()
            if not wanted(percent, source, percentages, sources):
                continue
            key = Key(
                percent, row["model"].strip(), row["seed"].strip(), source,
                order_text(row["order"]),
            )
            accuracy = row["accuracy"].strip()
            metric = Metric(
                word_tokens=integer(row["word_tokens"], "word_tokens", line),
                correct=integer(row["correct_predictions"], "correct_predictions", line),
                oov_tokens=integer(row["oov_tokens"], "oov_tokens", line),
                accuracy=float(accuracy) if accuracy else None,
                zero_events=integer(
                    row["zero_probability_events"], "zero_probability_events", line
                ),
                prediction_events=integer(row["prediction_events"], "prediction_events", line),
            )
            if key in result:
                duplicates.append(key)
            result[key] = metric
    return result, duplicates


def scan_predictions(path: Path, sources: set[str], percentages: set[float] | None,
                     progress_seconds: float) -> tuple[dict[Key, PredictionCount], int]:
    result: dict[Key, PredictionCount] = defaultdict(PredictionCount)
    required = {
        "percent", "model", "seed", "source", "expected", "prediction",
        "correctness", "weight", "rare_file",
    }
    scanned = 0
    started = last_progress = time.monotonic()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            scanned += 1
            percent = decimal(row["percent"], "percent", line)
            source = row["source"].strip()
            if wanted(percent, source, percentages, sources):
                model = row["model"].strip()
                key = Key(
                    percent, model, row["seed"].strip(), source,
                    infer_order(model, source, row["rare_file"]),
                )
                weight = integer(row["weight"] or "1", "weight", line)
                if weight < 1:
                    raise ValueError(f"Weight must be positive at line {line}")
                result[key].add(
                    row["expected"].strip(), row["prediction"].strip(),
                    row["correctness"], weight,
                )
            now = time.monotonic()
            if progress_seconds > 0 and now - last_progress >= progress_seconds:
                print(
                    f"  scanned {scanned:,} rows in {now - started:.0f}s",
                    file=sys.stderr, flush=True,
                )
                last_progress = now
    return dict(result), scanned


def divide(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def audit(key: Key, metric: Metric | None, predictions: PredictionCount | None
          ) -> dict[str, object]:
    prediction_rows_present = predictions is not None
    predictions = predictions or PredictionCount()
    if metric is None:
        total, correct = predictions.events, predictions.correct
        oov = 0
        accuracy = divide(correct, total)
        zero_rate = None
        status = "PARTIAL/no metrics row"
    else:
        total, correct, oov = metric.word_tokens, metric.correct, metric.oov_tokens
        accuracy = metric.accuracy if metric.accuracy is not None else divide(correct, total)
        zero_rate = divide(metric.zero_events, metric.prediction_events)
        if not prediction_rows_present and total:
            status = "PARTIAL predictions"
        elif predictions.correctness_mismatches:
            status = "INVALID correctness column"
        elif predictions.correct != correct:
            # Every correct ordinary-word prediction is retained in predictions.csv,
            # even though incorrect predictions of </s> are filtered out.
            status = "PARTIAL predictions"
        else:
            status = "complete"
    iv_total = max(0, total - oov)
    iv_correct = max(0, correct - predictions.unk_correct)
    return {
        "percent": key.percent,
        "dataset": key.dataset,
        "model": key.model,
        "seed": key.seed,
        "order": key.order,
        "status": status,
        "word_tokens": total,
        "correct": correct,
        "accuracy": accuracy,
        "iv_tokens": iv_total,
        "iv_correct": iv_correct,
        "iv_accuracy": divide(iv_correct, iv_total),
        "oov_tokens": oov,
        "oov_rate": divide(oov, total),
        "oov_accuracy": divide(predictions.unk_correct, oov),
        "predicted_unk": predictions.predicted_unk,
        "predicted_unk_rate": divide(predictions.predicted_unk, total),
        "iv_predicted_unk": predictions.iv_predicted_unk,
        "iv_false_unk_rate": divide(predictions.iv_predicted_unk, iv_total),
        "unk_correct": predictions.unk_correct,
        "unk_accuracy_points": divide(predictions.unk_correct, total),
        "unk_share_of_correct": divide(predictions.unk_correct, correct),
        "accuracy_treating_unk_wrong": divide(iv_correct, total),
        "zero_probability_rate": zero_rate,
        "prediction_csv_rows": predictions.rows,
        "prediction_csv_weighted_events": predictions.events,
        "predicted_unseen": predictions.predicted_unseen,
    }


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    if not rows:
        print("  No matching rows.")
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def flag(row: dict[str, object]) -> str:
    if row["status"] != "complete":
        return "INCOMPLETE"
    share = row["unk_share_of_correct"]
    points = row["unk_accuracy_points"]
    false_unk = row["iv_false_unk_rate"]
    if ((share is not None and share >= 0.50)
            or (points is not None and points >= 0.02)
            or (false_unk is not None and false_unk >= 0.10)):
        return "HIGH"
    if ((share is not None and share >= 0.20)
            or (points is not None and points >= 0.005)
            or (false_unk is not None and false_unk >= 0.03)):
        return "CHECK"
    return "low"


def ngram_pairs(audits: dict[Key, dict[str, object]]) -> list[dict[str, object]]:
    output = []
    dimensions = sorted(
        {(key.percent, key.source, key.order) for key in audits if key.order in {"2", "3"}}
    )
    for pct, source, order in dimensions:
        prefix = "bigram" if order == "2" else "trigram"
        keys = {
            method: Key(pct, f"{prefix}_{method}", "", source, order)
            for method in ("mle", "laplace", "backoff")
        }
        if not all(key in audits for key in keys.values()):
            continue
        mle, lap, backoff = (audits[keys[name]] for name in ("mle", "laplace", "backoff"))
        if not all(row["status"] == "complete" for row in (mle, lap, backoff)):
            continue
        if any(row["accuracy"] is None for row in (mle, lap, backoff)):
            continue
        raw_gain = lap["accuracy"] - mle["accuracy"]
        iv_gain = (
            lap["iv_accuracy"] - mle["iv_accuracy"]
            if lap["iv_accuracy"] is not None and mle["iv_accuracy"] is not None else None
        )
        extra_unk_points = divide(lap["unk_correct"] - mle["unk_correct"], lap["word_tokens"])
        output.append({
            "percent": pct,
            "dataset": lap["dataset"],
            "order": order,
            "mle_accuracy": mle["accuracy"],
            "laplace_accuracy": lap["accuracy"],
            "backoff_accuracy": backoff["accuracy"],
            "laplace_iv_accuracy": lap["iv_accuracy"],
            "laplace_gain": raw_gain,
            "laplace_iv_gain": iv_gain,
            "extra_unk_points": extra_unk_points,
            "gain_explained_by_extra_unk": (
                divide(extra_unk_points, raw_gain) if raw_gain > 0 else None
            ),
            "mle_zero_rate": mle["zero_probability_rate"],
        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["percent", "dataset", "model", "status"]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {destination}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="results/<run>, metrics folder, or predictions.csv")
    parser.add_argument(
        "--source", nargs="+", choices=DEFAULT_SOURCES, default=list(DEFAULT_SOURCES),
        help="sources to include (default: valid test rare)",
    )
    parser.add_argument("--percent", nargs="+", type=float, help="training percentages to include")
    parser.add_argument(
        "--all-models", action="store_true",
        help="show detailed rows for every model; default focuses on Laplace",
    )
    parser.add_argument("--output", type=Path, help="optional full-audit CSV destination")
    parser.add_argument(
        "--progress-seconds", type=float, default=15.0,
        help="progress interval while scanning; 0 disables it (default: 15)",
    )
    args = parser.parse_args()

    predictions_path, metrics_path = resolve_files(args.results)
    sources = set(args.source)
    percentages = set(args.percent) if args.percent else None
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print("Streaming existing CSVs; no model loading or inference.")

    metrics, duplicates = load_metrics(metrics_path, sources, percentages)
    counts, scanned = scan_predictions(
        predictions_path, sources, percentages, args.progress_seconds,
    )
    keys = sorted(set(metrics) | set(counts))
    audits = {key: audit(key, metrics.get(key), counts.get(key)) for key in keys}
    print(f"Scanned {scanned:,} prediction rows; found {len(audits):,} groups.")
    if duplicates:
        print(f"WARNING: metrics.csv has {len(duplicates)} duplicate group row(s).")

    detailed = list(audits.values())
    if not args.all_models:
        detailed = [row for row in detailed if "laplace" in str(row["model"]).casefold()]
    detailed.sort(key=lambda row: (
        float(row["percent"]), str(row["dataset"]), str(row["model"]), str(row["seed"])
    ))

    print("\nUNK audit")
    print_table(
        [
            "%", "data", "model", "all acc", "IV acc", "OOV rate", "OOV acc",
            "pred UNK", "acc from UNK", "correct from UNK", "flag",
        ],
        [[
            f"{float(row['percent']):g}", str(row["dataset"]),
            str(row["model"]) + (f"/s{row['seed']}" if row["seed"] else ""),
            percent(row["accuracy"]), percent(row["iv_accuracy"]),
            percent(row["oov_rate"]), percent(row["oov_accuracy"]),
            percent(row["predicted_unk_rate"]), percent(row["unk_accuracy_points"]),
            percent(row["unk_share_of_correct"]), flag(row),
        ] for row in detailed],
    )

    print("\nMatched n-gram comparison")
    print_table(
        [
            "%", "data", "n", "MLE acc", "Lap acc", "Backoff acc", "Lap IV acc",
            "Lap-MLE", "IV gain", "extra UNK", "gain via UNK", "MLE zero",
        ],
        [[
            f"{float(row['percent']):g}", str(row["dataset"]), str(row["order"]),
            percent(row["mle_accuracy"]), percent(row["laplace_accuracy"]),
            percent(row["backoff_accuracy"]), percent(row["laplace_iv_accuracy"]),
            percent(row["laplace_gain"]), percent(row["laplace_iv_gain"]),
            percent(row["extra_unk_points"]),
            percent(row["gain_explained_by_extra_unk"]), percent(row["mle_zero_rate"]),
        ] for row in ngram_pairs(audits)],
    )

    incomplete = sum(row["status"] != "complete" for row in audits.values())
    print("\nInterpretation")
    print("  IV acc excludes OOV targets that were mapped to <unk>.")
    print("  acc from UNK is the headline-accuracy contribution from <unk> -> <unk> matches.")
    print("  gain via UNK estimates how much of Laplace's raw gain over MLE came from extra UNK matches.")
    print("  A high value diagnoses possible UNK inflation; it does not by itself prove unseen-context ties.")
    print("  Train rows and sentence endings are not present in predictions.csv.")
    if incomplete:
        print(f"  WARNING: {incomplete} group(s) are partial or lack a completed metrics row.")
    if args.output:
        write_csv(args.output, [audits[key] for key in sorted(audits)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
