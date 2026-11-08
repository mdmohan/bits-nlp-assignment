#!/usr/bin/env python3
"""Generate compact, cross-corpus figures for the research paper.

The comparison pipeline writes one metrics bundle per corpus.  This command
combines the six current bundles into the small-multiple figures used by the
paper without loading a model or re-running evaluation.

Examples:
    python src/plot_paper.py
    python src/plot_paper.py --output-dir docs/figures
    python src/plot_paper.py --plots complete-training --tag final-100pct
    python src/plot_paper.py --skip-accuracy

Existing figures are never replaced.  Each invocation writes timestamped PNG
and vector-PDF files plus a JSON record of the inputs and plotted aggregates.
The accuracy figure streams ``predictions_correct.csv`` because some exports
are several gigabytes; it never loads those files into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PERCENTAGES = (20, 40, 60, 80, 100)

PLOT_ORDER = (
    "complete-training",
    "event-categories",
    "test-perplexity",
    "vocabulary-oov",
    "accuracy",
    "runtime",
)

PLOT_FUNCTIONS = {
    "complete-training": "plot_complete_training",
    "event-categories": "plot_event_category_contributions",
    "test-perplexity": "plot_test_perplexity",
    "vocabulary-oov": "plot_vocabulary_oov",
    "accuracy": "plot_accuracy",
    "runtime": "plot_runtime",
}

CORPUS_SPECS = (
    ("Brown", "brown_press_neural"),
    ("Gutenberg", "gutenberg_neural"),
    ("Inaugural", "inaugural_neural"),
    ("Reuters", "reuters_neural"),
    ("ROCStories", "roc-stories"),
    ("WebText", "webtext-nn"),
)

MODEL_ORDER = (
    "bigram_mle",
    "trigram_mle",
    "bigram_laplace",
    "trigram_laplace",
    "bigram_backoff",
    "trigram_backoff",
    "ffnn_ctx1",
    "ffnn_ctx2",
    "lstm",
)

FINITE_MODEL_ORDER = tuple(model for model in MODEL_ORDER if not model.endswith("_mle"))

MODEL_LABELS = {
    "bigram_mle": "B-MLE",
    "trigram_mle": "T-MLE",
    "bigram_laplace": "B-Laplace",
    "trigram_laplace": "T-Laplace",
    "bigram_backoff": "B-backoff",
    "trigram_backoff": "T-backoff",
    "ffnn_ctx1": "FFNN-1",
    "ffnn_ctx2": "FFNN-2",
    "lstm": "LSTM",
}

MODEL_STYLES = {
    "bigram_laplace": dict(color="#E69F00", marker="o", linestyle="-"),
    "trigram_laplace": dict(color="#D55E00", marker="o", linestyle="-"),
    "bigram_backoff": dict(color="#009E73", marker="s", linestyle="--"),
    "trigram_backoff": dict(color="#56B4E9", marker="s", linestyle="--"),
    "ffnn_ctx1": dict(color="#0072B2", marker="D", linestyle="-"),
    "ffnn_ctx2": dict(color="#CC79A7", marker="D", linestyle="-"),
    "lstm": dict(color="#8B1A1A", marker="^", linestyle="-"),
}

# Corrected complete-training aggregates reported in Table 4 of the paper.
# Each value is an additive contribution to LSTM-minus-backoff NLL in nats per
# total test event.  These retained aggregates are used because the revised
# WebText bundle does not contain the per-event archive needed for rescoring.
EVENT_CATEGORY_ORDER = (
    "seen_trigram",
    "unseen_trigram_seen_bigram",
    "unseen_bigram_retained_target",
    "oov_target",
)

EVENT_CATEGORY_LABELS = {
    "seen_trigram": "Seen trigram",
    "unseen_trigram_seen_bigram": "Unseen trigram, seen bigram",
    "unseen_bigram_retained_target": "Unseen bigram, retained target",
    "oov_target": "OOV target",
}

EVENT_CATEGORY_COLORS = {
    "seen_trigram": "#0072B2",
    "unseen_trigram_seen_bigram": "#56B4E9",
    "unseen_bigram_retained_target": "#009E73",
    "oov_target": "#D55E00",
}

EVENT_CATEGORY_CONTRIBUTIONS = {
    "Brown": {
        "seen_trigram": -0.06934,
        "unseen_trigram_seen_bigram": -0.04864,
        "unseen_bigram_retained_target": -0.11281,
        "oov_target": 0.00192,
        "total": -0.22887,
        "comparator": "bigram backoff",
    },
    "Gutenberg": {
        "seen_trigram": 0.02797,
        "unseen_trigram_seen_bigram": 0.10623,
        "unseen_bigram_retained_target": -0.18319,
        "oov_target": 0.19093,
        "total": 0.14194,
        "comparator": "bigram backoff",
    },
    "Inaugural": {
        "seen_trigram": 0.13304,
        "unseen_trigram_seen_bigram": 0.03651,
        "unseen_bigram_retained_target": -0.22356,
        "oov_target": 0.17724,
        "total": 0.12323,
        "comparator": "trigram backoff",
    },
    "Reuters": {
        "seen_trigram": 0.18257,
        "unseen_trigram_seen_bigram": -0.04131,
        "unseen_bigram_retained_target": -0.27837,
        "oov_target": 0.19332,
        "total": 0.05622,
        "comparator": "trigram backoff",
    },
    "ROCStories": {
        "seen_trigram": -0.06731,
        "unseen_trigram_seen_bigram": -0.14337,
        "unseen_bigram_retained_target": -0.19584,
        "oov_target": 0.06182,
        "total": -0.34469,
        "comparator": "trigram backoff",
    },
    "WebText": {
        "seen_trigram": 0.05337,
        "unseen_trigram_seen_bigram": 0.22513,
        "unseen_bigram_retained_target": -0.10223,
        "oov_target": 0.24978,
        "total": 0.42605,
        "comparator": "bigram backoff",
    },
}

METRICS_REQUIRED = {
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
}

STATS_REQUIRED = {
    "vocab_size",
    "valid_oov_rate",
    "test_oov_rate",
    "train_percent",
}


@dataclass(frozen=True)
class Bundle:
    """One corpus's retained comparison artifacts."""

    display_name: str
    result_name: str
    root: Path
    metrics_path: Path
    metrics: tuple[dict[str, str], ...]
    stats: dict[int, dict[str, object]]

    @property
    def corpus(self) -> str:
        return self.metrics[0]["corpus"].strip()


def _float(value: object, *, field: str, context: str) -> float:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"Missing {field} in {context}")
    try:
        result = float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}={text!r} in {context}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field}={text!r} in {context}")
    return result


def _optional_float(value: object) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _percent(value: object, *, context: str) -> int:
    number = _float(value, field="percent", context=context)
    rounded = int(round(number))
    if not math.isclose(number, rounded, abs_tol=1e-8):
        raise ValueError(f"Non-integral percent={number!r} in {context}")
    return rounded


def load_bundle(results_root: Path, display_name: str, result_name: str) -> Bundle:
    root = (results_root / result_name).resolve()
    metrics_path = root / "metrics" / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(METRICS_REQUIRED - columns)
        if missing:
            raise ValueError(
                f"{metrics_path} is missing required column(s): {', '.join(missing)}"
            )
        metrics = tuple(reader)

    if not metrics:
        raise ValueError(f"Metrics file contains no data rows: {metrics_path}")
    corpora = {row["corpus"].strip() for row in metrics}
    if len(corpora) != 1:
        raise ValueError(
            f"Expected one corpus in {metrics_path}; found {sorted(corpora)!r}"
        )

    stats: dict[int, dict[str, object]] = {}
    for percent in PERCENTAGES:
        path = root / f"train_{percent}pct" / "data_stats.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing budget statistics: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
        missing = sorted(STATS_REQUIRED - set(payload))
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(missing)}")
        recorded = int(round(_float(payload["train_percent"], field="train_percent", context=str(path))))
        if recorded != percent:
            raise ValueError(
                f"Budget mismatch in {path}: expected {percent}, found {recorded}"
            )
        stats[percent] = payload

    return Bundle(display_name, result_name, root, metrics_path, metrics, stats)


def load_bundles(results_root: Path) -> list[Bundle]:
    return [load_bundle(results_root, display, folder) for display, folder in CORPUS_SPECS]


def metric_rows(
    bundle: Bundle,
    model: str,
    percent: int,
    *,
    source: str = "test",
) -> list[dict[str, str]]:
    rows = [
        row
        for row in bundle.metrics
        if row["model"] == model
        and row["source"] == source
        and _percent(row["percent"], context=str(bundle.metrics_path)) == percent
    ]
    if not rows:
        raise ValueError(
            f"No {source} rows for {bundle.display_name}, {model}, {percent}%"
        )
    if model.startswith(("bigram_", "trigram_")) and len(rows) != 1:
        raise ValueError(
            f"Expected one deterministic row for {bundle.display_name}, {model}, "
            f"{percent}%; found {len(rows)}"
        )
    return rows


def mean_sd(values: Sequence[float]) -> tuple[float, float | None]:
    if not values:
        raise ValueError("Cannot summarize an empty value sequence")
    mean = statistics.fmean(values)
    return mean, statistics.stdev(values) if len(values) > 1 else None


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _save_figure(fig, output_base: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix, options in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
    ):
        path = output_base.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing figure: {path}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **options)
        outputs.append(path)
    return outputs


def complete_training_summary(bundle: Bundle) -> dict[str, object]:
    """Summarize the primary 100%-training test comparison from metrics.csv."""
    models: dict[str, object] = {}
    for model in MODEL_ORDER:
        rows = metric_rows(bundle, model, 100)
        accuracies = [
            _float(
                row["accuracy"],
                field="accuracy",
                context=f"{bundle.display_name}, {model}, seed {row['seed'] or '-'}",
            )
            for row in rows
        ]
        if any(not 0.0 <= value <= 1.0 for value in accuracies):
            raise ValueError(
                f"Accuracy outside [0, 1] for {bundle.display_name}, {model}"
            )
        accuracy_mean, accuracy_sd = mean_sd(accuracies)

        perplexities = [
            value
            for row in rows
            if (value := _optional_float(row["perplexity"])) is not None
        ]
        infinite = not perplexities and all(
            int(float(row["zero_probability_events"] or 0)) > 0
            and "infinite" in row["status"].strip().lower()
            for row in rows
        )
        if not infinite and len(perplexities) != len(rows):
            raise ValueError(
                f"Mixed finite/missing PPL for {bundle.display_name}, {model}"
            )

        perplexity_mean: float | None = None
        perplexity_sd: float | None = None
        if perplexities:
            perplexity_mean, perplexity_sd = mean_sd(perplexities)

        models[model] = {
            "accuracy_mean": accuracy_mean,
            "accuracy_sample_sd": accuracy_sd,
            "perplexity_mean": perplexity_mean,
            "perplexity_sample_sd": perplexity_sd,
            "perplexity_infinite": infinite,
            "seed_count": len(rows),
        }
    return {"percent": 100, "source": "test", "models": models}


def plot_complete_training(
    bundles: Sequence[Bundle], output_base: Path
) -> tuple[list[Path], dict[str, object]]:
    """Plot the primary complete-training accuracy/PPL result for every corpus."""
    summaries = {
        bundle.corpus: complete_training_summary(bundle) for bundle in bundles
    }
    maximum_accuracy = max(
        item["accuracy_mean"] * 100.0
        for corpus in summaries.values()
        for item in corpus["models"].values()
    )
    accuracy_limit = max(25.0, math.ceil(maximum_accuracy / 5.0) * 5.0)

    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.7), constrained_layout=False)
    y = list(range(len(MODEL_ORDER)))

    for ax, bundle in zip(axes.flat, bundles):
        corpus_summary = summaries[bundle.corpus]
        models = corpus_summary["models"]
        accuracy = [models[model]["accuracy_mean"] * 100.0 for model in MODEL_ORDER]
        accuracy_sd = [
            (models[model]["accuracy_sample_sd"] or 0.0) * 100.0
            for model in MODEL_ORDER
        ]

        ax.barh(
            y,
            accuracy,
            height=0.56,
            color="#3B6FB6",
            alpha=0.72,
            zorder=2,
        )
        error_positions = [
            position for position, deviation in enumerate(accuracy_sd) if deviation > 0
        ]
        if error_positions:
            ax.errorbar(
                [accuracy[position] for position in error_positions],
                error_positions,
                xerr=[accuracy_sd[position] for position in error_positions],
                fmt="none",
                ecolor="#1F1F1F",
                capsize=2,
                linewidth=0.8,
                zorder=3,
            )
        ax.set_xlim(0.0, accuracy_limit)
        ax.set_yticks(y, [MODEL_LABELS[model] for model in MODEL_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("Word top-1 accuracy (%)", color="#244E86")
        ax.tick_params(axis="x", labelcolor="#244E86")
        ax.grid(axis="x", color="#e4e4e4", linewidth=0.7, zorder=0)
        ax.set_title(bundle.display_name, pad=28, fontweight="semibold")

        ppl_ax = ax.twiny()
        finite_positions: list[int] = []
        finite_means: list[float] = []
        finite_sd: list[float] = []
        for position, model in enumerate(MODEL_ORDER):
            item = models[model]
            if item["perplexity_mean"] is None:
                if item["perplexity_infinite"]:
                    ppl_ax.text(
                        1.015,
                        position,
                        "PPL = ∞",
                        transform=ppl_ax.get_yaxis_transform(),
                        ha="left",
                        va="center",
                        color="#A84300",
                        fontsize=7,
                        fontweight="semibold",
                        clip_on=False,
                    )
                continue
            finite_positions.append(position)
            finite_means.append(item["perplexity_mean"])
            finite_sd.append(item["perplexity_sample_sd"] or 0.0)

        ppl_ax.scatter(
            finite_means,
            finite_positions,
            marker="D",
            color="#D55E00",
            s=25,
            zorder=4,
        )
        ppl_error_positions = [
            index for index, deviation in enumerate(finite_sd) if deviation > 0
        ]
        if ppl_error_positions:
            ppl_ax.errorbar(
                [finite_means[index] for index in ppl_error_positions],
                [finite_positions[index] for index in ppl_error_positions],
                xerr=[finite_sd[index] for index in ppl_error_positions],
                fmt="none",
                ecolor="#8C3A00",
                capsize=2,
                linewidth=0.8,
                zorder=3,
            )
        ppl_ax.set_xscale("log")
        ppl_ax.set_xlabel(
            "Test perplexity (log scale)",
            color="#A84300",
            fontsize=7,
            labelpad=3,
        )
        ppl_ax.tick_params(axis="x", labelcolor="#A84300", labelsize=7)
        ppl_ax.spines["top"].set_visible(True)
        ppl_ax.spines["top"].set_color("#A84300")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#3B6FB6", alpha=0.72, label="Word top-1 accuracy"),
        Line2D(
            [],
            [],
            color="#D55E00",
            marker="D",
            linestyle="none",
            markersize=5,
            label="Test perplexity",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        "Complete-training test performance by corpus",
        y=0.99,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.052,
        "Blue bottom axes: word accuracy. Orange top axes: panel-specific log PPL. "
        "PPL = ∞ is shown beyond the finite axis for zero-probability MLE. "
        "Neural values show mean ± sample SD where replicated.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.91,
        bottom=0.135,
        hspace=0.52,
        wspace=0.38,
    )
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs, summaries


def plot_event_category_contributions(
    bundles: Sequence[Bundle], output_base: Path
) -> tuple[list[Path], dict[str, object]]:
    """Plot corrected complete-training LSTM-minus-backoff NLL contributions."""
    corpus_names = [bundle.display_name for bundle in bundles]
    missing = [
        corpus for corpus in corpus_names if corpus not in EVENT_CATEGORY_CONTRIBUTIONS
    ]
    if missing:
        raise ValueError(
            "Missing retained event-category contribution(s): " + ", ".join(missing)
        )

    summaries: dict[str, object] = {}
    for corpus in corpus_names:
        retained = EVENT_CATEGORY_CONTRIBUTIONS[corpus]
        calculated_total = sum(retained[category] for category in EVENT_CATEGORY_ORDER)
        recorded_total = float(retained["total"])
        if not math.isclose(calculated_total, recorded_total, abs_tol=2e-5):
            raise ValueError(
                f"Event-category contributions do not add to the retained total for "
                f"{corpus}: {calculated_total:.5f} versus {recorded_total:.5f}"
            )
        summaries[corpus] = {
            "categories": {
                category: retained[category] for category in EVENT_CATEGORY_ORDER
            },
            "total": recorded_total,
            "sum_of_displayed_rounded_categories": calculated_total,
            "comparator": retained["comparator"],
        }

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.2, 5.25), constrained_layout=False)
    y = list(range(len(corpus_names)))
    positive_left = [0.0] * len(corpus_names)
    negative_left = [0.0] * len(corpus_names)

    for category in EVENT_CATEGORY_ORDER:
        values = [
            float(EVENT_CATEGORY_CONTRIBUTIONS[corpus][category])
            for corpus in corpus_names
        ]
        left: list[float] = []
        for index, value in enumerate(values):
            if value >= 0:
                left.append(positive_left[index])
                positive_left[index] += value
            else:
                left.append(negative_left[index])
                negative_left[index] += value
        ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=EVENT_CATEGORY_COLORS[category],
            label=EVENT_CATEGORY_LABELS[category],
            zorder=2,
        )

    totals = [
        float(EVENT_CATEGORY_CONTRIBUTIONS[corpus]["total"])
        for corpus in corpus_names
    ]
    ax.scatter(
        totals,
        y,
        marker="D",
        s=31,
        color="#1F1F1F",
        edgecolor="white",
        linewidth=0.55,
        label="Total",
        zorder=4,
    )
    for position, total in zip(y, totals):
        offset = 0.014 if total >= 0 else -0.014
        ax.text(
            total + offset,
            position,
            f"{total:+.3f}",
            ha="left" if total >= 0 else "right",
            va="center",
            fontsize=7.5,
            fontweight="semibold",
            color="#1F1F1F",
            zorder=5,
        )

    maximum = max(
        max(positive_left),
        abs(min(negative_left)),
        max(abs(value) for value in totals),
    )
    limit = math.ceil((maximum + 0.05) * 10.0) / 10.0
    ax.set_xlim(-limit, limit)
    ax.axvline(0.0, color="#4D4D4D", linewidth=0.8, zorder=3)
    ax.set_yticks(y, corpus_names)
    ax.invert_yaxis()
    ax.set_xlabel("Contribution to LSTM − backoff NLL (nats per test event)")
    ax.set_title(
        "Complete-training category contributions",
        pad=25,
        fontweight="semibold",
    )
    ax.grid(axis="x", color="#e4e4e4", linewidth=0.7, zorder=0)
    ax.text(
        0.01,
        1.015,
        "← LSTM lower loss",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.text(
        0.99,
        1.015,
        "Backoff lower loss →",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.text(
        0.5,
        0.015,
        "Stacked segments show signed category contributions; black diamonds show "
        "the net corpus total.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.17, right=0.97, top=0.88, bottom=0.25)
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs, {
        "source": "Corrected complete-training Table 4 aggregates",
        "units": "nats per total test event",
        "difference": "LSTM minus fixed backoff NLL",
        "corpora": summaries,
    }


def plot_test_perplexity(
    bundles: Sequence[Bundle], output_base: Path
) -> tuple[list[Path], dict[str, object]]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), constrained_layout=False)
    summary: dict[str, object] = {}

    for ax, bundle in zip(axes.flat, bundles):
        corpus_summary: dict[str, object] = {}
        for model in FINITE_MODEL_ORDER:
            means: list[float] = []
            deviations: list[float] = []
            counts: list[int] = []
            for percent in PERCENTAGES:
                rows = metric_rows(bundle, model, percent)
                values = [
                    value
                    for row in rows
                    if (value := _optional_float(row["perplexity"])) is not None
                ]
                if len(values) != len(rows):
                    raise ValueError(
                        f"Expected finite PPL for {bundle.display_name}, {model}, "
                        f"{percent}%"
                    )
                mean, deviation = mean_sd(values)
                means.append(mean)
                deviations.append(0.0 if deviation is None else deviation)
                counts.append(len(values))

            style = MODEL_STYLES[model]
            kwargs = dict(
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.5,
                markersize=4,
                capsize=2,
                label=MODEL_LABELS[model],
            )
            yerr = deviations if max(deviations, default=0.0) > 0 else None
            ax.errorbar(PERCENTAGES, means, yerr=yerr, **kwargs)
            corpus_summary[model] = {
                "mean": dict(zip(map(str, PERCENTAGES), means)),
                "sample_sd": dict(zip(map(str, PERCENTAGES), deviations)),
                "seed_count": dict(zip(map(str, PERCENTAGES), counts)),
            }

        mle_infinite = True
        for model in ("bigram_mle", "trigram_mle"):
            for percent in PERCENTAGES:
                row = metric_rows(bundle, model, percent)[0]
                zero = int(float(row["zero_probability_events"] or 0))
                status = row["status"].strip().lower()
                ppl = _optional_float(row["perplexity"])
                if not (zero > 0 and (status == "infinite" or ppl is None)):
                    mle_infinite = False
        note = "B/T MLE: ∞\nat every budget" if mle_infinite else "MLE omitted where PPL = ∞"
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82, edgecolor="none"),
        )
        ax.set_title(bundle.display_name)
        ax.set_yscale("log")
        ax.set_xticks(PERCENTAGES)
        ax.set_xlabel("Training-sentence budget (%)")
        ax.set_ylabel("Test perplexity (words + EOS)")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.grid(axis="y", which="major", color="#eeeeee", linewidth=0.6)
        corpus_summary["mle_infinite_at_all_budgets"] = mle_infinite
        summary[bundle.corpus] = corpus_summary

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("Test perplexity across six corpora and five training budgets", y=0.985, fontweight="semibold")
    fig.text(
        0.5,
        0.105,
        "Panel-specific logarithmic scales; neural points show mean ± sample SD where multiple seeds exist.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.20, hspace=0.42, wspace=0.34)
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs, summary


def plot_vocabulary_oov(
    bundles: Sequence[Bundle], output_base: Path
) -> tuple[list[Path], dict[str, object]]:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), constrained_layout=False)
    summary: dict[str, object] = {}
    legend_handles = None

    for ax, bundle in zip(axes.flat, bundles):
        vocab = [float(bundle.stats[p]["vocab_size"]) / 1000.0 for p in PERCENTAGES]
        valid_oov = [float(bundle.stats[p]["valid_oov_rate"]) * 100.0 for p in PERCENTAGES]
        test_oov = [float(bundle.stats[p]["test_oov_rate"]) * 100.0 for p in PERCENTAGES]
        twin = ax.twinx()

        vocabulary_line = ax.plot(
            PERCENTAGES,
            vocab,
            color="#009E73",
            marker="o",
            linewidth=1.7,
            markersize=4,
            label="Vocabulary",
        )[0]
        test_line = twin.plot(
            PERCENTAGES,
            test_oov,
            color="#D55E00",
            marker="s",
            linewidth=1.7,
            markersize=4,
            label="Test OOV",
        )[0]
        valid_line = twin.plot(
            PERCENTAGES,
            valid_oov,
            color="#0072B2",
            linestyle="--",
            linewidth=1.5,
            label="Validation OOV",
        )[0]
        if legend_handles is None:
            legend_handles = [vocabulary_line, test_line, valid_line]

        ax.set_title(bundle.display_name)
        ax.set_xticks(PERCENTAGES)
        ax.set_xlabel("Training-sentence budget (%)")
        ax.set_ylabel("Vocabulary (thousands)", color="#007A58")
        twin.set_ylabel("OOV word tokens (%)", color="#A84300")
        ax.tick_params(axis="y", labelcolor="#007A58")
        twin.tick_params(axis="y", labelcolor="#A84300")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
        twin.spines["right"].set_visible(True)

        summary[bundle.corpus] = {
            "vocabulary": dict(zip(map(str, PERCENTAGES), [int(v * 1000) for v in vocab])),
            "validation_oov_percent": dict(zip(map(str, PERCENTAGES), valid_oov)),
            "test_oov_percent": dict(zip(map(str, PERCENTAGES), test_oov)),
        }

    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Training vocabulary and held-out OOV rate across budgets", y=0.985, fontweight="semibold")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.91, bottom=0.14, hspace=0.42, wspace=0.48)
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs, summary


def stream_oov_correct_counts(
    path: Path,
    *,
    progress_seconds: float = 10.0,
) -> tuple[dict[tuple[str, str], int], int]:
    """Return full-budget test correct-UNK counts keyed by model and seed."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing correct-prediction export: {path}")
    counts: dict[tuple[str, str], int] = {}
    rows_read = 0
    last_report = time.monotonic()
    started = last_report

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"percent", "model", "seed", "source", "expected", "weight"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing: {', '.join(missing)}")
        for row in reader:
            rows_read += 1
            if (
                row["source"] == "test"
                and row["expected"] == "<unk>"
                and math.isclose(float(row["percent"]), 100.0, abs_tol=1e-8)
            ):
                key = (row["model"], row["seed"].strip())
                counts[key] = counts.get(key, 0) + int(float(row["weight"] or 1))

            if progress_seconds > 0 and time.monotonic() - last_report >= progress_seconds:
                elapsed = time.monotonic() - started
                print(
                    f"  {path.parent.parent.name}: streamed {rows_read:,} correct rows "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
                last_report = time.monotonic()

    elapsed = time.monotonic() - started
    print(
        f"  {path.parent.parent.name}: retained-accuracy scan complete; "
        f"{rows_read:,} correct rows in {elapsed:.1f}s",
        flush=True,
    )
    return counts, rows_read


def accuracy_summary(
    bundle: Bundle,
    *,
    progress_seconds: float,
) -> tuple[dict[str, object], int]:
    correct_path = bundle.root / "metrics" / "predictions_correct.csv"
    oov_correct, rows_read = stream_oov_correct_counts(
        correct_path, progress_seconds=progress_seconds
    )

    models: dict[str, object] = {}
    baseline_values: set[float] = set()
    for model in MODEL_ORDER:
        rows = metric_rows(bundle, model, 100)
        overall_values: list[float] = []
        retained_values: list[float] = []
        seed_values: list[dict[str, object]] = []
        for row in rows:
            seed = row["seed"].strip()
            words = int(float(row["word_tokens"]))
            oov = int(float(row["oov_tokens"]))
            correct = int(float(row["correct_predictions"]))
            retained_denominator = words - oov
            correct_oov = oov_correct.get((model, seed), 0)
            retained_correct = correct - correct_oov
            if words <= 0 or retained_denominator <= 0:
                raise ValueError(
                    f"Invalid accuracy denominator for {bundle.display_name}, {model}, seed {seed or '-'}"
                )
            if not 0 <= correct_oov <= correct or not 0 <= retained_correct <= retained_denominator:
                raise ValueError(
                    f"Inconsistent correct counts for {bundle.display_name}, {model}, seed {seed or '-'}: "
                    f"correct={correct}, correct_oov={correct_oov}, words={words}, oov={oov}"
                )
            overall = correct / words
            retained = retained_correct / retained_denominator
            overall_values.append(overall)
            retained_values.append(retained)
            baseline_values.add(oov / words)
            seed_values.append({
                "seed": seed or None,
                "overall_accuracy": overall,
                "retained_accuracy": retained,
                "correct_oov": correct_oov,
                "correct_retained": retained_correct,
                "word_tokens": words,
                "oov_tokens": oov,
            })

        overall_mean, overall_sd = mean_sd(overall_values)
        retained_mean, retained_sd = mean_sd(retained_values)
        models[model] = {
            "overall_mean": overall_mean,
            "overall_sample_sd": overall_sd,
            "retained_mean": retained_mean,
            "retained_sample_sd": retained_sd,
            "seeds": seed_values,
        }

    if len({round(value, 12) for value in baseline_values}) != 1:
        raise ValueError(
            f"Full-budget test OOV rate is inconsistent across models for {bundle.display_name}"
        )
    return {
        "always_unk_baseline": next(iter(baseline_values)),
        "models": models,
        "source": str(correct_path),
    }, rows_read


def plot_accuracy(
    bundles: Sequence[Bundle],
    output_base: Path,
    *,
    progress_seconds: float,
) -> tuple[list[Path], dict[str, object]]:
    summaries: dict[str, object] = {}
    print("Scanning correct-prediction exports for retained-word accuracy...", flush=True)
    for bundle in bundles:
        summary, rows_read = accuracy_summary(
            bundle, progress_seconds=progress_seconds
        )
        summary["rows_streamed"] = rows_read
        summaries[bundle.corpus] = summary

    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.2), constrained_layout=False)
    y = list(range(len(MODEL_ORDER)))
    width = 0.36

    for ax, bundle in zip(axes.flat, bundles):
        corpus_summary = summaries[bundle.corpus]
        models = corpus_summary["models"]
        overall = [models[model]["overall_mean"] * 100.0 for model in MODEL_ORDER]
        retained = [models[model]["retained_mean"] * 100.0 for model in MODEL_ORDER]
        overall_sd = [
            (models[model]["overall_sample_sd"] or 0.0) * 100.0
            for model in MODEL_ORDER
        ]
        retained_sd = [
            (models[model]["retained_sample_sd"] or 0.0) * 100.0
            for model in MODEL_ORDER
        ]
        baseline = corpus_summary["always_unk_baseline"] * 100.0

        ax.barh(
            [position - width / 2 for position in y],
            overall,
            height=width,
            color="#3B6FB6",
            label="Overall words",
            xerr=overall_sd,
            error_kw=dict(ecolor="black", capsize=2, linewidth=0.8),
        )
        ax.barh(
            [position + width / 2 for position in y],
            retained,
            height=width,
            color="#009E73",
            label="In-vocabulary words",
            xerr=retained_sd,
            error_kw=dict(ecolor="black", capsize=2, linewidth=0.8),
        )
        ax.axvline(
            baseline,
            color="#D55E00",
            linestyle="--",
            linewidth=1.3,
            label="Always <unk>",
        )
        ax.text(
            baseline,
            1.01,
            f"{baseline:.1f}%",
            transform=ax.get_xaxis_transform(),
            color="#A84300",
            ha="center",
            va="bottom",
            fontsize=7,
        )
        ax.set_title(bundle.display_name)
        ax.set_yticks(y, [MODEL_LABELS[model] for model in MODEL_ORDER])
        ax.invert_yaxis()
        ax.set_xlabel("Word top-1 accuracy (%)")
        ax.grid(axis="x", color="#e4e4e4", linewidth=0.7)
        upper = max(overall + retained + [baseline]) * 1.15
        ax.set_xlim(0, max(upper, 1.0))

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Full-budget accuracy and the mapped-unknown baseline", y=0.985, fontweight="semibold")
    fig.text(
        0.5,
        0.046,
        "Accuracy uses word targets only; error bars show sample SD where multiple neural seeds exist.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.14, hspace=0.42, wspace=0.34)
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs, summaries


def _load_runs(bundle: Bundle) -> dict[tuple[str, str], dict[str, str]]:
    path = bundle.root / "train_100pct" / "runs.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        variant = (row.get("variant") or "").strip()
        seed = (row.get("seed") or "").strip()
        if variant:
            result[(variant, seed)] = row
    return result


def runtime_summary(bundle: Bundle) -> dict[str, object]:
    runs = _load_runs(bundle)
    ngram_rows = [
        row
        for model in MODEL_ORDER[:6]
        for row in metric_rows(bundle, model, 100)
    ]
    shared_values = [
        value
        for row in ngram_rows
        if (value := _optional_float(row["train_seconds"])) is not None
    ]
    if not shared_values:
        raise ValueError(f"No shared count-fit time for {bundle.display_name}")
    shared_count_fit = statistics.median(shared_values)

    models: dict[str, object] = {}
    for model in MODEL_ORDER:
        rows = metric_rows(bundle, model, 100)
        test_values: list[float] = []
        train_values: list[float] = []
        for row in rows:
            evaluation = _optional_float(row["evaluation_seconds"])
            if evaluation is None:
                raise ValueError(
                    f"No test-scoring time for {bundle.display_name}, {model}"
                )
            test_values.append(evaluation)
            if model.startswith("ffnn_") or model == "lstm":
                training = _optional_float(row["train_seconds"])
                if training is None:
                    fallback = runs.get((model, row["seed"].strip()), {})
                    training = _optional_float(fallback.get("train_seconds"))
                if training is None:
                    raise ValueError(
                        f"No training time for {bundle.display_name}, {model}, seed {row['seed'] or '-'}"
                    )
                train_values.append(training)

        test_mean, test_sd = mean_sd(test_values)
        train_mean, train_sd = (None, None)
        if train_values:
            train_mean, train_sd = mean_sd(train_values)
        models[model] = {
            "test_mean_seconds": test_mean,
            "test_sample_sd_seconds": test_sd,
            "train_mean_seconds": train_mean,
            "train_sample_sd_seconds": train_sd,
            "seed_count": len(rows),
        }

    return {"shared_count_fit_seconds": shared_count_fit, "models": models}


def plot_runtime(
    bundles: Sequence[Bundle], output_base: Path
) -> tuple[list[Path], dict[str, object]]:
    summaries = {bundle.corpus: runtime_summary(bundle) for bundle in bundles}
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 8.0), constrained_layout=False)
    labels = ["Shared counts", *[MODEL_LABELS[model] for model in MODEL_ORDER]]
    y = list(range(len(labels)))

    for ax, bundle in zip(axes.flat, bundles):
        summary = summaries[bundle.corpus]
        train_x: list[float] = [summary["shared_count_fit_seconds"]]
        train_y: list[int] = [0]
        train_err: list[float] = [0.0]
        test_x: list[float] = []
        test_y: list[int] = []
        test_err: list[float] = []

        for index, model in enumerate(MODEL_ORDER, start=1):
            item = summary["models"][model]
            test_x.append(item["test_mean_seconds"])
            test_y.append(index)
            test_err.append(item["test_sample_sd_seconds"] or 0.0)
            if item["train_mean_seconds"] is not None:
                train_x.append(item["train_mean_seconds"])
                train_y.append(index)
                train_err.append(item["train_sample_sd_seconds"] or 0.0)

        ax.errorbar(
            train_x,
            train_y,
            xerr=train_err,
            fmt="o",
            color="#009E73",
            ecolor="#00664D",
            capsize=2,
            markersize=5,
            label="Fit/train",
        )
        ax.errorbar(
            test_x,
            test_y,
            xerr=test_err,
            fmt="s",
            color="#0072B2",
            ecolor="#004B75",
            capsize=2,
            markersize=4,
            label="Test scoring",
        )
        ax.set_xscale("log")
        ax.set_title(bundle.display_name)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Recorded wall time (seconds, log scale)")
        ax.grid(axis="x", which="major", color="#e4e4e4", linewidth=0.7)

    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Recorded full-budget fitting and test-scoring time", y=0.985, fontweight="semibold")
    fig.text(
        0.5,
        0.047,
        "Shared statistical counting is shown once; execution environments are not hardware-matched.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.92, bottom=0.14, hspace=0.43, wspace=0.34)
    outputs = _save_figure(fig, output_base)
    plt.close(fig)
    return outputs, summaries


def _safe_tag(tag: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag):
        raise ValueError(
            "--tag must start with an alphanumeric character and contain only "
            "letters, digits, dots, underscores or hyphens"
        )
    return tag


def choose_tag(output_dir: Path, requested: str | None) -> str:
    if requested:
        tag = _safe_tag(requested)
        if any(output_dir.glob(f"paper_*_{tag}.*")):
            raise FileExistsError(
                f"Output tag {tag!r} already exists under {output_dir}; choose another tag"
            )
        return tag

    base = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    tag = base
    counter = 1
    while any(output_dir.glob(f"paper_*_{tag}.*")):
        tag = f"{base}_{counter:02d}"
        counter += 1
    return tag


def input_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def generate(
    results_root: Path,
    output_dir: Path,
    *,
    tag: str | None = None,
    plots: Sequence[str] | None = None,
    skip_accuracy: bool = False,
    progress_seconds: float = 10.0,
) -> list[Path]:
    results_root = results_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results root is not a directory: {results_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tag = choose_tag(output_dir, tag)
    bundles = load_bundles(results_root)

    requested = set(PLOT_ORDER if plots is None else plots)
    unknown = sorted(requested - set(PLOT_ORDER))
    if unknown:
        raise ValueError(f"Unknown plot selection(s): {', '.join(unknown)}")
    if skip_accuracy:
        requested.discard("accuracy")
    selected_plots = tuple(name for name in PLOT_ORDER if name in requested)
    if not selected_plots:
        raise ValueError("No plots selected")

    outputs: list[Path] = []
    aggregates: dict[str, object] = {}
    plot_outputs: dict[str, list[str]] = {}

    if "complete-training" in requested:
        print("Generating complete-training accuracy/PPL figure...", flush=True)
        created, data = plot_complete_training(
            bundles, output_dir / f"paper_complete_training_{output_tag}"
        )
        outputs.extend(created)
        plot_outputs["complete-training"] = [str(path) for path in created]
        aggregates["complete_training"] = data

    if "event-categories" in requested:
        print("Generating complete-training event-category figure...", flush=True)
        created, data = plot_event_category_contributions(
            bundles, output_dir / f"paper_event_categories_{output_tag}"
        )
        outputs.extend(created)
        plot_outputs["event-categories"] = [str(path) for path in created]
        aggregates["event_categories"] = data

    if "test-perplexity" in requested:
        print("Generating compact test-perplexity figure...", flush=True)
        created, data = plot_test_perplexity(
            bundles, output_dir / f"paper_test_perplexity_{output_tag}"
        )
        outputs.extend(created)
        plot_outputs["test-perplexity"] = [str(path) for path in created]
        aggregates["test_perplexity"] = data

    if "vocabulary-oov" in requested:
        print("Generating compact vocabulary/OOV figure...", flush=True)
        created, data = plot_vocabulary_oov(
            bundles, output_dir / f"paper_vocabulary_oov_{output_tag}"
        )
        outputs.extend(created)
        plot_outputs["vocabulary-oov"] = [str(path) for path in created]
        aggregates["vocabulary_oov"] = data

    if "accuracy" in requested:
        created, data = plot_accuracy(
            bundles,
            output_dir / f"paper_accuracy_{output_tag}",
            progress_seconds=progress_seconds,
        )
        outputs.extend(created)
        plot_outputs["accuracy"] = [str(path) for path in created]
        aggregates["accuracy"] = data

    if "runtime" in requested:
        print("Generating compact runtime figure...", flush=True)
        created, data = plot_runtime(
            bundles, output_dir / f"paper_runtime_{output_tag}"
        )
        outputs.extend(created)
        plot_outputs["runtime"] = [str(path) for path in created]
        aggregates["runtime"] = data

    inputs: list[dict[str, object]] = []
    for bundle in bundles:
        inputs.append(input_record(bundle.metrics_path))
        if "vocabulary-oov" in requested:
            for percent in PERCENTAGES:
                inputs.append(input_record(bundle.root / f"train_{percent}pct" / "data_stats.json"))
        if "runtime" in requested:
            runs = bundle.root / "train_100pct" / "runs.csv"
            if runs.is_file():
                inputs.append(input_record(runs))
        if "accuracy" in requested:
            inputs.append(input_record(bundle.root / "metrics" / "predictions_correct.csv"))

    record_path = output_dir / f"paper_plot_data_{output_tag}.json"
    if record_path.exists():
        raise FileExistsError(f"Refusing to overwrite provenance record: {record_path}")
    record = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "tag": output_tag,
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "corpus_order": [bundle.corpus for bundle in bundles],
        "display_names": {bundle.corpus: bundle.display_name for bundle in bundles},
        "selected_plots": list(selected_plots),
        "plot_functions": {
            name: PLOT_FUNCTIONS[name] for name in selected_plots
        },
        "plot_outputs": plot_outputs,
        "inputs": inputs,
        "outputs": [str(path) for path in outputs],
        "accuracy_skipped": "accuracy" not in requested,
        "aggregates": aggregates,
        "not_generated": {
            "event_category_shares": (
                "Revised WebText scoring-event archive is not present locally."
            ),
        },
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs.append(record_path)
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="folder containing the six result bundles (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="destination for timestamped plots (default: docs/figures)",
    )
    parser.add_argument(
        "--tag",
        help="explicit non-overwriting filename tag; default is a local timestamp",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=PLOT_ORDER,
        metavar="NAME",
        help=(
            "generate only the named plot families; choices: "
            + ", ".join(PLOT_ORDER)
            + " (default: all)"
        ),
    )
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="skip the retained-word accuracy plot and its large CSV scans",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        default=10.0,
        help="progress interval while streaming correct predictions; 0 disables updates",
    )
    args = parser.parse_args(argv)
    if args.progress_seconds < 0:
        parser.error("--progress-seconds must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        outputs = generate(
            args.results_root,
            args.output_dir,
            tag=args.tag,
            plots=args.plots,
            skip_accuracy=args.skip_accuracy,
            progress_seconds=args.progress_seconds,
        )
    except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Generated {len(outputs)} new file(s); existing figures were preserved:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
