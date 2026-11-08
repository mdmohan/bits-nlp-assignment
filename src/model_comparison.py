#!/usr/bin/env python3
"""Evaluate saved neural and count-based LMs across available train percentages.

Usage: python src/model_comparison.py <corpus> [--check | --resume]

The script discovers ``results/<run-name>/train_*pct`` folders (or a legacy
single results folder), evaluates each saved neural checkpoint and six n-gram
baselines on the exact saved training subset plus valid/test, then writes
metrics, perplexity summaries, prediction details, and plots under
``results/<run-name>/metrics/``. Evaluation events are streamed to CSV to avoid
keeping corpus-sized prediction tables in memory.

Normal runs write into a resume journal and replace the canonical CSV files
only after every evaluation unit succeeds. ``--resume`` validates completed
metric rows, removes any uncommitted prediction rows, and continues with only
the missing units. Existing suffixed prediction exports can be used as the
preserved prediction source without loading them into memory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import yaml


def find_project_root(start: Path = Path.cwd()) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "src/lm/cli.py").is_file() and (candidate / "src/configs").is_dir():
            return candidate
    raise FileNotFoundError("Run this script from the repository or one of its subdirectories.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from lm.cli import CountNGram, LanguageModelCLI, iter_sentences
from lm.data import BOS, EOS, PAD, UNK, Vocab
from lm.evaluation import evaluate_model, neural_events
from lm.tokenization import canonical_tokenizer_name, get_tokenizer


PREDICTION_FIELDS = [
    "percent", "model", "seed", "source", "context", "expected", "prediction",
    "correctness", "weight", "rare_file",
]
BASE_METRIC_FIELDS = [
    "corpus", "percent", "model", "family", "order", "seed", "source",
    "sentences", "word_tokens", "prediction_events", "correct_predictions",
    "accuracy", "perplexity", "status", "zero_probability_events", "oov_tokens",
    "oov_rate", "evaluation_seconds", "train_seconds",
]
METRIC_FIELDS = BASE_METRIC_FIELDS + [
    "prediction_rows", "prediction_weight", "prediction_digest",
    "evaluation_signature",
]

NGRAM_SPECS = [
    {"name": "bigram_mle", "order": 2, "smoothing": "mle"},
    {"name": "bigram_laplace", "order": 2, "smoothing": "laplace", "k": 1.0},
    {"name": "bigram_backoff", "order": 2, "smoothing": "backoff", "discount": 0.75},
    {"name": "trigram_mle", "order": 3, "smoothing": "mle"},
    {"name": "trigram_laplace", "order": 3, "smoothing": "laplace", "k": 1.0},
    {"name": "trigram_backoff", "order": 3, "smoothing": "backoff", "discount": 0.75},
]
SPECIAL_OUTPUT = {PAD, BOS, EOS}


class NullCSVWriter:
    """CSV-writer-compatible sink used for derivable resume subsets."""

    @staticmethod
    def writerow(_row: dict) -> None:
        return None


def empty_prediction_stats() -> dict:
    return {
        "rows": 0,
        "weight": 0,
        "correct": 0,
        "invalid": 0,
        "digest": hashlib.sha256(),
    }


def add_prediction_stats(stats: dict, row: dict, weight: int) -> None:
    """Add one normalized export row to a per-unit validation digest."""
    values = [str(row.get(field, "")) for field in PREDICTION_FIELDS]
    payload = "\x1f".join(values).encode("utf-8")
    stats["digest"].update(len(payload).to_bytes(8, "big"))
    stats["digest"].update(payload)
    stats["rows"] += 1
    stats["weight"] += weight
    correctness = row.get("correctness")
    if correctness == "correct":
        stats["correct"] += weight
    elif correctness != "incorrect":
        stats["invalid"] += 1


def prediction_digest(stats: dict | None) -> str:
    if not stats:
        return hashlib.sha256().hexdigest()
    digest = stats.get("digest")
    return digest.hexdigest() if hasattr(digest, "hexdigest") else str(digest or "")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_signature(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def project_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def display_path(path: Path) -> Path:
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT)
    return path


def load_corpus_config(
    corpus: str, explicit_path: str | Path | None = None
) -> tuple[dict, Path]:
    if explicit_path is not None:
        path = project_path(explicit_path)
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if (config.get("data") or {}).get("corpus") != corpus:
            raise ValueError(f"{path} does not configure data.corpus={corpus!r}")
        return config, path
    config_dir = PROJECT_ROOT / "src/configs"
    direct = config_dir / f"{corpus}.yaml"
    candidates = [direct] if direct.is_file() else []
    if not candidates:
        for path in sorted(config_dir.glob("*.yaml")):
            try:
                candidate = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(candidate, dict) and (candidate.get("data") or {}).get("corpus") == corpus:
                candidates.append(path)
    if len(candidates) != 1:
        available = []
        for candidate_path in config_dir.glob("*.yaml"):
            try:
                candidate_config = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(candidate_config, dict) and "run" not in candidate_config:
                available.append(candidate_path.stem)
        available.sort()
        raise FileNotFoundError(
            f"Expected one corpus config for {corpus!r}; found {len(candidates)}. "
            f"Available config names: {available}"
        )
    path = candidates[0]
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if (config.get("data") or {}).get("corpus") != corpus:
        raise ValueError(f"{path.name} does not configure data.corpus={corpus!r}")
    return config, path


def discover_percent_folders(run_base: Path, config: dict) -> list[tuple[float, Path]]:
    found = []
    pattern = re.compile(r"^train_(\d+(?:\.\d+)?)pct$")
    if run_base.is_dir():
        for path in run_base.iterdir():
            match = pattern.fullmatch(path.name)
            if path.is_dir() and match:
                found.append((float(match.group(1)), path))
    if found:
        return sorted(found, key=lambda item: item[0])
    if run_base.is_dir() and (run_base / "models").is_dir():
        percent = float(config.get("train_percent", 100))
        return [(percent, run_base)]
    raise FileNotFoundError(f"No percentage result folders or legacy models directory under {run_base}")


def checkpoint_paths(percent_dir: Path) -> list[Path]:
    model_root = percent_dir / "models"
    if not model_root.is_dir():
        model_root = percent_dir
    paths = sorted(
        p for p in model_root.rglob("*_seed*.pt")
        if p.is_file() and "checkpoints" not in p.parts
    )
    return paths


def model_label(checkpoint: Path, loaded_name: str) -> str:
    folder = checkpoint.parent.name
    if re.fullmatch(r"(?:ffnn_ctx\d+|lstm)(?:_\d+)?", folder):
        return folder
    return loaded_name


def make_ngram_variants(train_path: Path, vocab: Vocab, lowercase: bool,
                        tokenizer: str, train_file_percent: float
                        ) -> tuple[dict[str, dict], dict[int, float]]:
    """Fit one trigram table, then expose both orders and smoothing variants."""
    variants: dict[str, dict] = {}
    import copy

    started = time.perf_counter()
    base = CountNGram(
        order=3,
        vocab=vocab.itos,
        sentences=iter_sentences(
            train_path, lowercase, train_file_percent, tokenizer_name=tokenizer
        ),
        smoothing="mle",
        allowed_target_ids=vocab.allowed_target_ids,
    )
    fit_seconds = time.perf_counter() - started
    train_seconds = {2: fit_seconds, 3: fit_seconds}
    for spec in NGRAM_SPECS:
        order = spec["order"]
        view = copy.copy(base)
        view.order = order
        view.counts = view.model.ngram_counts[order]
        view._best_followers = {}
        for gram, count in view.counts.items():
            context, token = gram[:-1], gram[-1]
            current = view._best_followers.get(context)
            if current is None or count > current[0] or (count == current[0] and token < current[1]):
                view._best_followers[context] = (count, token)
        view.smoothing = spec["smoothing"]
        view.k = float(spec.get("k", 1.0))
        view.discount = float(spec.get("discount", 0.75))
        view._backoff_prediction_cache = {}
        view.model.smoothing = view.smoothing
        view.model.k = view.k
        view.model.d = view.discount
        view.model._alpha_cache.clear()
        variants[spec["name"]] = {
            "model": view,
            "family": "ngram",
            "order": order,
            "seed": "",
            "train_seconds": fit_seconds,
            "spec": spec,
        }
    return variants, train_seconds


def metric_row(corpus: str, percent: float, model_name: str, info: dict,
               source: str, result: dict, elapsed: float,
               prediction_stats: dict | None = None) -> dict:
    prediction_stats = prediction_stats or {}
    return {
        "corpus": corpus,
        "percent": percent,
        "model": model_name,
        "family": info["family"],
        "order": info.get("order", ""),
        "seed": info.get("seed", ""),
        "source": source,
        "sentences": result.get("sentences", ""),
        "word_tokens": result.get("word_tokens", result.get("tokens", "")),
        "prediction_events": result.get("tokens", ""),
        "correct_predictions": result.get("correct", ""),
        "accuracy": result.get("accuracy"),
        "perplexity": result.get("perplexity"),
        "status": result.get("status", ""),
        "zero_probability_events": result.get("zero_probability_events", 0),
        "oov_tokens": result.get("oov_tokens", ""),
        "oov_rate": result.get("oov_rate", ""),
        "evaluation_seconds": elapsed,
        "train_seconds": info.get("train_seconds", ""),
        "prediction_rows": prediction_stats.get("rows", 0),
        "prediction_weight": prediction_stats.get("weight", 0),
        "prediction_digest": prediction_digest(prediction_stats),
        "evaluation_signature": info.get("evaluation_signature", ""),
    }


def report_progress(label: str, source: str, started: float, state: dict) -> None:
    now = time.monotonic()
    if now - state["last"] >= 20:
        print(f"  {label} / {source}: {state['events']:,} events ({now - started:.0f}s)", flush=True)
        state["last"] = now


def write_full_event(handle, event: dict, vocab: Vocab, metadata: dict) -> None:
    position = event["position"]
    raw_target = EOS if event["eos"] else event["words"][position]
    log_probability = event["log_probability"]
    zero_probability = log_probability == -math.inf
    if not zero_probability and not math.isfinite(log_probability):
        raise ValueError("Invalid non-finite token log probability")
    record = {
        **metadata,
        "sentence_index": event["sentence_index"],
        "position": position,
        "raw_target": raw_target,
        "target_id": event["target"],
        "mapped_target": vocab.itos[event["target"]],
        "prediction_id": event["prediction"],
        "correct": event["target"] == event["prediction"],
        "eos": event["eos"],
        "oov": not event["eos"] and raw_target not in vocab,
        "context_covered": event.get("covered"),
        "log_probability": None if zero_probability else log_probability,
        "zero_probability": zero_probability,
        "weight": 1,
    }
    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def verify_scoring_events(metrics_path: Path, events_path: Path) -> dict:
    with metrics_path.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["source"] in {"train", "valid", "test"}
        ]

    def key(row):
        return (float(row["percent"]), row["model"], str(row["seed"]), row["source"])

    expected = {key(row): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate base metric conditions")
    aggregates = {}
    with gzip.open(events_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            condition = key(event)
            if condition not in expected:
                raise ValueError(f"Unexpected scoring event condition: {condition}")
            totals = aggregates.setdefault(
                condition,
                dict(events=0, words=0, correct=0, oov=0, zeros=0,
                     nll=0.0, sentences=0, position=0),
            )
            if (event["sentence_index"] != totals["sentences"]
                    or event["position"] != totals["position"]):
                raise ValueError(f"Missing or repeated scoring occurrence: {condition}")
            totals["events"] += 1
            totals["zeros"] += int(event["zero_probability"])
            if not event["zero_probability"]:
                totals["nll"] -= event["log_probability"]
            if event["eos"]:
                totals["sentences"] += 1
                totals["position"] = 0
            else:
                totals["position"] += 1
                totals["words"] += 1
                totals["correct"] += int(event["correct"])
                totals["oov"] += int(event["oov"])
    if set(expected) != set(aggregates):
        raise ValueError("Scoring events do not cover the complete metric matrix")
    for condition, row in expected.items():
        totals = aggregates[condition]
        if totals["position"]:
            raise ValueError(f"Scoring stream is missing its final EOS: {condition}")
        for metric, field in (
            ("prediction_events", "events"), ("word_tokens", "words"),
            ("correct_predictions", "correct"), ("oov_tokens", "oov"),
            ("zero_probability_events", "zeros"), ("sentences", "sentences"),
        ):
            if int(row[metric]) != totals[field]:
                raise ValueError(f"Event reconstruction mismatch for {condition}: {metric}")
        accuracy = totals["correct"] / totals["words"]
        if not math.isclose(float(row["accuracy"]), accuracy, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(f"Accuracy reconstruction mismatch: {condition}")
        if totals["zeros"]:
            if row["status"] != "infinite_zero_probability":
                raise ValueError(f"Zero-probability status mismatch: {condition}")
        elif row["status"] == "finite":
            mean_nll = totals["nll"] / totals["events"]
            if not math.isclose(
                math.log(float(row["perplexity"])), mean_nll,
                rel_tol=1e-6, abs_tol=1e-6,
            ):
                raise ValueError(f"Perplexity reconstruction mismatch: {condition}")
    return {
        "passed": True,
        "conditions": len(expected),
        "events": sum(value["events"] for value in aggregates.values()),
        "accuracy_denominator": "word_tokens",
        "perplexity_denominator": "word_tokens_plus_eos",
        "sentence_index_origin": 0,
        "position_origin": 0,
    }


def evaluate_split(corpus: str, percent: float, label: str, info: dict, model,
                   vocab: Vocab, path: Path, source: str, cli: LanguageModelCLI,
                   writers: tuple[csv.DictWriter, csv.DictWriter, csv.DictWriter],
                   event_handle=None) -> dict:
    all_writer, correct_writer, incorrect_writer = writers
    started = time.perf_counter()
    progress_started = time.monotonic()
    state = {"events": 0, "last": progress_started}
    prediction_stats = empty_prediction_stats()

    def sink(event: dict) -> None:
        state["events"] += 1
        report_progress(label, source, progress_started, state)
        if event_handle is not None:
            write_full_event(
                event_handle,
                event,
                vocab,
                {
                    "corpus": corpus,
                    "percent": percent,
                    "model": label,
                    "seed": info.get("seed", ""),
                    "source": source,
                },
            )
        # Keep train split evaluation for aggregate metrics, but only emit
        # per-token prediction rows for held-out and rare examples.
        if source == "train":
            return
        if event.get("eos"):
            return
        # The evaluation event doesn't include n-gram boundary padding in words;
        # output only ordinary word targets/predictions and omit special matches.
        target_id, prediction_id = event["target"], event["prediction"]
        expected = vocab.itos[target_id]
        predicted = vocab.itos[prediction_id] if prediction_id is not None else "<unseen>"
        if expected in SPECIAL_OUTPUT or predicted in SPECIAL_OUTPUT:
            return
        if info["family"] == "ngram":
            context = " ".join(
                event["words"][:event["position"]][-(info["order"] - 1):]
            )
        else:
            context = " ".join(event["words"][:event["position"]])
        row = {
            "percent": percent,
            "model": label,
            "seed": info.get("seed", ""),
            "source": source,
            "context": context,
            "expected": expected,
            "prediction": predicted,
            "correctness": "correct" if target_id == prediction_id else "incorrect",
            "weight": 1,
            "rare_file": "",
        }
        add_prediction_stats(prediction_stats, row, 1)
        all_writer.writerow(row)
        (correct_writer if row["correctness"] == "correct" else incorrect_writer).writerow(row)

    result = evaluate_model(
        model=model,
        vocab=vocab,
        sentences=iter_sentences(
            path,
            info["lowercase"],
            info.get("train_file_percent", 100) if source == "train" else 100,
            tokenizer_name=info["tokenizer"],
        ),
        family=info["family"],
        device=cli.device if info["family"] == "neural" else None,
        allowed_target_ids=info.get("allowed_target_ids", "vocab"),
        event_sink=sink,
        case_limit=0,
    )
    elapsed = time.perf_counter() - started
    row = metric_row(
        corpus, percent, label, info, source, result, elapsed, prediction_stats
    )
    print(
        f"  {label} / {source}: {result['word_tokens']:,} words, "
        f"accuracy={format_number(result.get('accuracy'))}, "
        f"ppl={format_number(result.get('perplexity'))}, {elapsed:.1f}s",
        flush=True,
    )
    return row


def format_number(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isinf(value):
        return "∞"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.5g}" if isinstance(value, float) else str(value)


def rare_rows(path: Path, order: int, tokenizer: str):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"order", "sequence"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Rare CSV {path} must have order and sequence columns")
        for row in reader:
            try:
                if int(row["order"]) != order:
                    continue
                if canonical_tokenizer_name(row.get("tokenizer") or "space") != tokenizer:
                    continue
                tokens = row["sequence"].split()
                if len(tokens) != order:
                    continue
                weight = max(1, int(row.get("test_count") or 1))
            except (TypeError, ValueError):
                continue
            yield tokens[:-1], tokens[-1], weight


def rare_metric_accumulator():
    return {"events": 0, "words": 0, "correct": 0, "word_correct": 0, "nll": 0.0,
            "zeros": 0, "oov": 0, "sentences": 0}


def add_rare_score(totals: dict, log_probability: float, target_id: int,
                   prediction_id: int | None, weight: int, is_eos: bool,
                   is_oov: bool) -> None:
    totals["events"] += weight
    if is_eos:
        totals["sentences"] += weight
    else:
        totals["words"] += weight
        totals["oov"] += weight * int(is_oov)
        totals["word_correct"] += weight * int(prediction_id == target_id)
    totals["correct"] += weight * int(prediction_id == target_id)
    if not math.isfinite(log_probability):
        totals["zeros"] += weight
    else:
        totals["nll"] -= weight * log_probability


def finish_rare_metrics(totals: dict) -> dict:
    events = totals["events"]
    zeros = totals["zeros"]
    status = "infinite_zero_probability" if zeros else ("finite" if events else "empty")
    if zeros:
        ppl = math.inf
    elif events:
        mean_nll = totals["nll"] / events
        if mean_nll >= math.log(float.fromhex("0x1.fffffffffffffp+1023")):
            ppl, status = math.inf, "overflow"
        else:
            ppl = math.exp(mean_nll)
    else:
        ppl = None
    return {
        "sentences": totals["sentences"],
        "word_tokens": totals["words"],
        "tokens": events,
        "correct": totals["word_correct"],
        "accuracy": totals["word_correct"] / totals["words"] if totals["words"] else None,
        "perplexity": ppl,
        "status": status,
        "zero_probability_events": zeros,
        "oov_tokens": totals["oov"],
        "oov_rate": totals["oov"] / totals["words"] if totals["words"] else 0.0,
    }


def rare_prediction_writer(event: dict, vocab: Vocab, model_name: str,
                           info: dict, percent: float, path: Path,
                           writers: tuple[csv.DictWriter, csv.DictWriter, csv.DictWriter],
                           expected_id: int, weight: int, context_words: list[str],
                           prediction_stats: dict) -> None:
    all_writer, correct_writer, incorrect_writer = writers
    prediction_id = event["prediction"]
    expected = vocab.itos[expected_id]
    prediction = vocab.itos[prediction_id] if prediction_id is not None else "<unseen>"
    if expected in SPECIAL_OUTPUT or prediction in SPECIAL_OUTPUT:
        return
    context = " ".join(word for word in context_words if word not in {BOS, EOS, PAD})
    row = {
        "percent": percent,
        "model": model_name,
        "seed": info.get("seed", ""),
        "source": "rare",
        "context": context,
        "expected": expected,
        "prediction": prediction,
        "correctness": "correct" if prediction_id == expected_id else "incorrect",
        "weight": weight,
        "rare_file": path.name,
    }
    add_prediction_stats(prediction_stats, row, weight)
    all_writer.writerow(row)
    (correct_writer if row["correctness"] == "correct" else incorrect_writer).writerow(row)


def evaluate_rare_file(corpus: str, percent: float, label: str, info: dict,
                       model, vocab: Vocab, path: Path, order: int,
                       cli: LanguageModelCLI,
                       writers: tuple[csv.DictWriter, csv.DictWriter, csv.DictWriter]) -> dict:
    started = time.perf_counter()
    totals = rare_metric_accumulator()
    stoi, unk_id = vocab.stoi, vocab.unk_id
    batch = []
    row_count = 0
    progress_started = time.monotonic()
    progress_state = {"events": 0, "last": progress_started}
    prediction_stats = empty_prediction_stats()

    def score_batch(records):
        if info["family"] == "ngram":
            for context_words, target_word, weight in records:
                words = [word.lower() for word in context_words + [target_word]] if info["lowercase"] else context_words + [target_word]
                mapped_context_words = list(words[:-1])
                context_ids = [stoi.get(word, unk_id) for word in mapped_context_words]
                context_ids = [stoi[BOS]] * max(0, order - 1 - len(context_ids)) + context_ids[-(order - 1):]
                target_id = stoi.get(words[-1], unk_id)
                target_oov = words[-1] not in stoi and words[-1] not in {BOS, EOS, PAD}
                probability, prediction_id = model.score(tuple(context_ids), target_id)
                log_probability = math.log(probability) if probability > 0 else -math.inf
                add_rare_score(totals, log_probability, target_id, prediction_id, weight,
                               words[-1] == EOS, target_oov)
                event = {"prediction": prediction_id}
                rare_prediction_writer(event, vocab, label, info, percent, path, writers,
                                       target_id, weight, mapped_context_words,
                                       prediction_stats)
        else:
            # Model input sentences recreate the same prefix. The evaluator adds
            # BOS/EOS itself; removing stored BOS padding here preserves the prefix.
            examples = []
            targets = []
            contexts = []
            weights = []
            oov_flags = []
            for context_words, target_word, weight in records:
                words = [word.lower() for word in context_words + [target_word]] if info["lowercase"] else context_words + [target_word]
                context = [w for w in words[:-1] if w not in {BOS, EOS, PAD}]
                target = words[-1]
                examples.append(context + [target])
                targets.append(stoi.get(target, unk_id))
                contexts.append(context_words)
                weights.append(weight)
                oov_flags.append(target not in stoi and target not in {BOS, EOS, PAD})
            final_positions = [len(sentence) - 1 for sentence in examples]
            selected = {}
            for event in neural_events(
                model, vocab, examples, cli.device,
                batch_size=max(1, min(32, len(examples))), max_padded_tokens=2048,
                allowed_target_ids=info.get("allowed_target_ids", "vocab"),
            ):
                if event["position"] == final_positions[event["sentence_index"]]:
                    selected[event["sentence_index"]] = event
            for index, event in selected.items():
                target_id = targets[index]
                target_word = vocab.itos[target_id]
                add_rare_score(
                    totals, event["log_probability"], target_id, event["prediction"],
                    weights[index], target_word == EOS,
                    oov_flags[index],
                )
                rare_prediction_writer(
                    event, vocab, label, info, percent, path, writers,
                    target_id, weights[index], contexts[index], prediction_stats,
                )
        progress_state["events"] += sum(item[2] for item in records)
        report_progress(label, "rare", progress_started, progress_state)

    for context_words, target_word, weight in rare_rows(path, order, info["tokenizer"]):
        row_count += 1
        batch.append((context_words, target_word, weight))
        if len(batch) >= 128:
            score_batch(batch)
            batch.clear()
    if batch:
        score_batch(batch)
    elapsed = time.perf_counter() - started
    result = finish_rare_metrics(totals)
    print(
        f"  {label} / rare ({path.name}): {row_count:,} unique rows, "
        f"{totals['events']:,} weighted events, ppl={format_number(result['perplexity'])}, {elapsed:.1f}s",
        flush=True,
    )
    row = metric_row(
        corpus, percent, label, info, "rare", result, elapsed, prediction_stats
    )
    row["order"] = order
    return row


def comparison_percentages(config_path: Path, run_base: Path) -> list[float]:
    """Use mega-run percentages when configured; otherwise inspect result folders."""
    found = re.compile(r"^train_(\d+(?:\.\d+)?)pct$")
    percentages = [
        float(match.group(1))
        for path in run_base.iterdir() if path.is_dir()
        if (match := found.fullmatch(path.name))
    ] if run_base.is_dir() else []
    if not percentages and (run_base / "models").is_dir():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return [float(config.get("train_percent", 100))]

    planned: list[float] = []
    manifest_path = PROJECT_ROOT / "src/configs/mega-run.yaml"
    if manifest_path.is_file():
        try:
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            configured_runs = manifest.get("run", manifest.get("configs", []))
            configured_names = {Path(value).name for value in configured_runs if isinstance(value, str)}
            if config_path.name in configured_names:
                values = manifest.get("train_percentages", manifest.get("percentages", [20, 40, 60, 80, 100]))
                if isinstance(values, list) and values:
                    planned = [float(value) for value in values]
        except (OSError, TypeError, ValueError, yaml.YAMLError):
            pass
    if not planned and not percentages:
        planned = [100.0]
    return sorted(set(planned) | set(percentages))


def expected_neural_runs(config: dict) -> list[tuple[str, str, str, int | None]]:
    """Return (folder label, model name, seed, FFNN context size) entries."""
    expanded = []
    context_sizes = config.get("context_sizes")
    for model in config.get("models", []):
        name = str(model.get("name", "model"))
        params = dict(model.get("params", {}))
        if name.casefold() == "ffnn" and context_sizes:
            for context_size in context_sizes:
                expanded.append((name, {**params, "context_size": context_size}))
        else:
            expanded.append((name, params))

    counts: dict[str, int] = {}
    runs = []
    for name, params in expanded:
        context_size = params.get("context_size", 3) if name.casefold() == "ffnn" else None
        label = f"ffnn_ctx{context_size}" if context_size is not None else name
        counts[label] = counts.get(label, 0) + 1
        folder = label if counts[label] == 1 else f"{label}_{counts[label]}"
        for seed in config.get("seeds", [42]):
            runs.append((folder, name, str(seed), context_size))
    return runs


def count_split_targets(path: Path, lowercase: bool, tokenizer: str,
                        percent: float = 100) -> dict[str, int] | None:
    if not path.is_file():
        return None
    sentences = words = 0
    for sentence in iter_sentences(path, lowercase, percent, tokenizer):
        sentences += 1
        words += len(sentence)
    return {"sentences": sentences, "word_tokens": words, "prediction_events": words + sentences}


def count_rare_targets(path: Path, order: int, tokenizer: str) -> dict[str, int]:
    sentences = words = events = 0
    for _context, target, weight in rare_rows(path, order, tokenizer):
        events += weight
        if target == EOS:
            sentences += weight
        else:
            words += weight
    return {"sentences": sentences, "word_tokens": words, "prediction_events": events}


def metric_key(row: dict) -> tuple[float, str, str, str, str]:
    percent = float(row.get("percent") or 0)
    try:
        order = str(int(float(row.get("order") or 0))) if row.get("order") else ""
    except (TypeError, ValueError):
        order = str(row.get("order") or "")
    return (
        round(percent, 8), str(row.get("model") or ""), str(row.get("seed") or ""),
        str(row.get("source") or ""), order,
    )


def model_order(label: str) -> str:
    """Return the matched n-gram order encoded by a model label."""
    if label.startswith("bigram_") or re.match(r"^ffnn_ctx1(?:_|$)", label):
        return "2"
    if label.startswith("trigram_") or re.match(r"^ffnn_ctx2(?:_|$)", label):
        return "3"
    return ""


def prediction_key(row: dict) -> tuple[float, str, str, str, str]:
    """Map one prediction export row to the metric unit that produced it."""
    label = str(row.get("model") or "")
    source = str(row.get("source") or "")
    order = model_order(label)
    if source == "rare":
        rare_name = str(row.get("rare_file") or "").casefold()
        if "bigram" in rare_name:
            order = "2"
        elif "trigram" in rare_name:
            order = "3"
    return (
        round(float(row.get("percent") or 0), 8), label,
        str(row.get("seed") or ""), source, order,
    )


def checkpoint_identity(checkpoint: Path) -> tuple[str, str, str]:
    """Return (model label, seed, matched order) without loading a checkpoint."""
    match = re.search(r"_seed(-?\d+)$", checkpoint.stem)
    if not match:
        raise ValueError(f"Checkpoint name does not contain a seed: {checkpoint}")
    folder = checkpoint.parent.name
    if re.fullmatch(r"(?:ffnn_ctx\d+|lstm)(?:_\d+)?", folder):
        label = folder
    else:
        label = checkpoint.stem[:match.start()]
    return label, match.group(1), model_order(label)


def metric_row_valid(row: dict, expected: dict[str, int]) -> bool:
    """Validate the counts and basic invariants of one resumable metric row."""
    try:
        observed = {
            field: int(float(row[field]))
            for field in ("sentences", "word_tokens", "prediction_events")
        }
        correct = int(float(row["correct_predictions"]))
        zeros = int(float(row.get("zero_probability_events") or 0))
        oov = int(float(row.get("oov_tokens") or 0))
        accuracy = float(row["accuracy"]) if row.get("accuracy") not in (None, "") else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if observed != expected:
        return False
    if not (0 <= correct <= observed["word_tokens"]):
        return False
    if not (0 <= zeros <= observed["prediction_events"]):
        return False
    if not (0 <= oov <= observed["word_tokens"]):
        return False
    prediction_rows = row.get("prediction_rows")
    prediction_weight = row.get("prediction_weight")
    if prediction_rows not in (None, "") or prediction_weight not in (None, ""):
        try:
            emitted_rows = int(float(prediction_rows))
            emitted_weight = int(float(prediction_weight))
        except (TypeError, ValueError, OverflowError):
            return False
        if str(row.get("source") or "") == "train":
            if emitted_rows != 0 or emitted_weight != 0:
                return False
        elif not (
            0 <= emitted_rows <= emitted_weight <= observed["word_tokens"]
            and correct <= emitted_weight
        ):
            return False
        digest = str(row.get("prediction_digest") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            return False
    signature = str(row.get("evaluation_signature") or "")
    if signature and not re.fullmatch(r"[0-9a-f]{64}", signature):
        return False
    if observed["word_tokens"]:
        if accuracy is None or not math.isclose(
            accuracy, correct / observed["word_tokens"], rel_tol=1e-9, abs_tol=1e-12
        ):
            return False
    status = str(row.get("status") or "")
    if status not in {"finite", "infinite_zero_probability", "overflow", "empty"}:
        return False
    if status == "finite":
        try:
            perplexity = float(row["perplexity"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(perplexity) or perplexity <= 0:
            return False
    return True


def metric_core(row: dict) -> tuple[str, ...]:
    """Comparable metric content excluding machine-dependent timing columns."""
    return tuple(str(row.get(field, "")) for field in BASE_METRIC_FIELDS
                 if field not in {"evaluation_seconds", "train_seconds"})


def load_resumable_metrics(
    sources: list[tuple[str, Path]],
    expected_totals: dict[tuple[float, str, str, str, str], dict[str, int]],
    expected_signatures: dict[tuple[float, str, str, str, str], str] | None = None,
) -> tuple[dict[tuple[float, str, str, str, str], dict], dict, list[str]]:
    """Load unique, count-valid metric rows from canonical and journal CSVs."""
    candidates: dict[tuple[float, str, str, str, str], list[tuple[str, dict]]] = defaultdict(list)
    issues: list[str] = []
    for origin, path in sources:
        if not path.is_file() or not path.stat().st_size:
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or not set(BASE_METRIC_FIELDS).issubset(reader.fieldnames):
                    issues.append(f"ignored {path.name}: incompatible metric header")
                    continue
                for row in reader:
                    try:
                        key = metric_key(row)
                    except (TypeError, ValueError):
                        continue
                    expected = expected_totals.get(key)
                    signature_matches = (
                        expected_signatures is None
                        or row.get("evaluation_signature") == expected_signatures.get(key)
                    )
                    if (
                        expected is not None
                        and signature_matches
                        and metric_row_valid(row, expected)
                    ):
                        candidates[key].append((origin, row))
        except (OSError, csv.Error) as exc:
            issues.append(f"ignored {path.name}: {exc}")

    rows: dict[tuple[float, str, str, str, str], dict] = {}
    origins: dict[tuple[float, str, str, str, str], str] = {}
    for key, items in candidates.items():
        cores = {metric_core(row) for _origin, row in items}
        if len(cores) != 1:
            issues.append(
                f"will recompute conflicting metric rows: {key[0]:g}% {key[1]} "
                f"seed={key[2]} {key[3]} order={key[4]}"
            )
            continue
        # Prefer the newer row carrying exact prediction-emission counts. A
        # resume journal is newer than an equivalent canonical row and may be
        # paired with a newly regenerated prediction group.
        origin, row = max(
            items,
            key=lambda item: (
                bool(item[1].get("prediction_rows") not in (None, "")),
                item[0] == "journal",
            ),
        )
        rows[key] = row
        origins[key] = origin
    return rows, origins, issues


def valid_csv_header(path: Path, required: list[str]) -> bool:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return set(required).issubset(next(csv.reader(handle), []))
    except (OSError, csv.Error, StopIteration):
        return False


def prediction_candidates(report_dir: Path) -> list[Path]:
    """Return full prediction exports, excluding subsets and resume/temp files."""
    candidates = []
    for path in report_dir.glob("predictions*.csv"):
        name = path.name
        if (not path.is_file() or not path.stat().st_size
                or name.startswith("predictions_correct")
                or name.startswith("predictions_incorrect")
                or ".resume" in name or ".tmp" in name):
            continue
        if valid_csv_header(path, PREDICTION_FIELDS):
            candidates.append(path)
    return candidates


def select_prediction_source(report_dir: Path, explicit: Path | None = None) -> Path | None:
    """Select an existing full prediction export without modifying it."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file() or not valid_csv_header(path, PREDICTION_FIELDS):
            raise ValueError(f"Prediction source is missing or has an incompatible header: {path}")
        return path
    candidates = prediction_candidates(report_dir)
    return max(candidates, key=lambda path: path.stat().st_size) if candidates else None


def scan_prediction_stats(path: Path | None, label: str) -> dict:
    """Stream a prediction export and collect small per-unit validation counts."""
    stats = defaultdict(empty_prediction_stats)
    if path is None or not path.is_file() or not path.stat().st_size:
        return stats
    started = time.monotonic()
    last = started
    total_rows = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(PREDICTION_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Prediction CSV has an incompatible header: {path}")
        for row in reader:
            try:
                key = prediction_key(row)
                weight = max(1, int(row.get("weight") or 1))
            except (TypeError, ValueError):
                continue
            add_prediction_stats(stats[key], row, weight)
            total_rows += 1
            now = time.monotonic()
            if now - last >= 20:
                print(
                    f"  resume scan {label}: {total_rows:,} prediction rows "
                    f"({now - started:.0f}s)", flush=True,
                )
                last = now
    print(f"  resume scan {label}: {total_rows:,} prediction rows complete", flush=True)
    return stats


def prediction_group_valid(row: dict, stats: dict | None) -> bool:
    """Check that a preserved prediction group agrees with its completed metric."""
    if str(row.get("source") or "") == "train":
        return True  # Training metrics intentionally have no per-token export.
    try:
        expected_rows = int(float(row["prediction_rows"]))
        expected_weight = int(float(row["prediction_weight"]))
        expected_correct = int(float(row["correct_predictions"]))
        observed = stats or empty_prediction_stats()
        return (
            not observed.get("invalid")
            and observed["rows"] == expected_rows
            and observed["weight"] == expected_weight
            and observed["correct"] == expected_correct
            and prediction_digest(observed) == str(row["prediction_digest"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def write_metric_csv(path: Path, rows: list[dict]) -> None:
    """Atomically write a metric CSV."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rewrite_prediction_journal(path: Path, allowed_keys: set) -> None:
    """Drop interrupted prediction groups from a resume journal atomically."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        if path.is_file() and path.stat().st_size:
            with path.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames and set(PREDICTION_FIELDS).issubset(reader.fieldnames):
                    for row in reader:
                        try:
                            if prediction_key(row) in allowed_keys:
                                writer.writerow(row)
                        except (TypeError, ValueError):
                            continue
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def merge_prediction_exports(
    base_source: Path | None,
    base_keys: set,
    journal_source: Path,
    journal_keys: set,
    output_files: dict[str, Path],
    metric_rows: dict,
) -> None:
    """Build the three canonical prediction exports and replace them atomically."""
    destinations = {
        name: path.with_name(path.name + ".tmp")
        for name, path in (
            ("predictions", output_files["predictions"]),
            ("correct", output_files["correct"]),
            ("incorrect", output_files["incorrect"]),
        )
    }
    handles = {
        name: path.open("w", encoding="utf-8", newline="")
        for name, path in destinations.items()
    }
    writers = {
        name: csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        for name, handle in handles.items()
    }
    for writer in writers.values():
        writer.writeheader()
    merged_stats = defaultdict(empty_prediction_stats)

    def copy_rows(source_path: Path | None, allowed: set) -> None:
        if source_path is None or not source_path.is_file() or not source_path.stat().st_size:
            return
        with source_path.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not set(PREDICTION_FIELDS).issubset(reader.fieldnames):
                raise ValueError(f"Prediction CSV has an incompatible header: {source_path}")
            for row in reader:
                try:
                    key = prediction_key(row)
                    if key not in allowed:
                        continue
                    weight = max(1, int(row.get("weight") or 1))
                except (TypeError, ValueError):
                    continue
                writers["predictions"].writerow(row)
                correctness = row.get("correctness")
                if correctness in {"correct", "incorrect"}:
                    writers[correctness].writerow(row)
                add_prediction_stats(merged_stats[key], row, weight)

    try:
        try:
            copy_rows(base_source, base_keys)
            copy_rows(journal_source, journal_keys)
            bad = [
                key for key, row in metric_rows.items()
                if row.get("source") != "train"
                and not prediction_group_valid(row, merged_stats.get(key))
            ]
            if bad:
                sample = ", ".join(
                    f"{key[0]:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}"
                    for key in bad[:5]
                )
                raise ValueError(
                    f"Prediction exports do not match {len(bad)} completed metric unit(s): {sample}"
                )
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            for handle in handles.values():
                handle.close()

        for name in ("predictions", "correct", "incorrect"):
            os.replace(destinations[name], output_files[name])
    except BaseException:
        for temporary in destinations.values():
            temporary.unlink(missing_ok=True)
        raise


def check_comparison(corpus: str, expected_percentages: list[float] | None = None) -> dict:
    """Read existing comparison CSVs and report completed versus pending evaluations."""
    config, config_path = load_corpus_config(corpus)
    data = config.get("data", {})
    corpus_root = project_path(data.get("output", "data/processed")) / corpus
    run_base = project_path(config.get("output_dir", "results")) / config.get("name", corpus)
    metrics_path = run_base / "metrics" / "metrics.csv"
    metric_sources = [metrics_path, run_base / "metrics" / ".metrics.resume.csv"]
    percentages = (
        [float(value) for value in expected_percentages]
        if expected_percentages is not None
        else comparison_percentages(config_path, run_base)
    )
    lowercase = bool(data.get("lowercase", True))
    tokenizer = canonical_tokenizer_name(data.get("tokenizer", "space"))
    rare_paths = {
        2: corpus_root / "rare-seq-bigram.csv",
        3: corpus_root / "rare-seq-trigram.csv",
    }
    rare_paths = {order: path for order, path in rare_paths.items() if path.is_file()}

    saved_rows: dict[tuple[float, str, str, str, str], list[dict]] = defaultdict(list)
    csv_row_count = 0
    for metric_source in metric_sources:
        if not metric_source.is_file() or not metric_source.stat().st_size:
            continue
        try:
            with metric_source.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    csv_row_count += 1
                    try:
                        saved_rows[metric_key(row)].append(row)
                    except (TypeError, ValueError):
                        continue
        except (OSError, csv.Error) as exc:
            print(f"Could not read {metric_source}: {exc}")
            return {
                "complete": 0,
                "expected": 0,
                "reports_complete": False,
                "missing_reports": [metric_source.name],
                "all_complete": False,
            }

    print("Read-only model-comparison check (no model inference or report writes).")
    print(f"Corpus: {corpus} | metrics rows read: {csv_row_count:,}")
    print(f"Metrics CSV: {metrics_path}")
    if metric_sources[1].is_file() and metric_sources[1].stat().st_size:
        print(f"Resume journal: {metric_sources[1]}")
    if not rare_paths and data.get("rare_check", data.get("rare-check", False)):
        print("Rare inputs: expected by config but no rare-seq CSVs are present.")
    elif rare_paths:
        print("Rare inputs: " + ", ".join(path.name for path in rare_paths.values()))

    models = expected_neural_runs(config)
    if not models:
        print("No neural models are configured; cannot determine the comparison plan.")
        return {"complete": 0, "expected": 0, "reports_complete": False}

    overall_complete = overall_expected = 0
    issues: list[str] = []
    split_count_cache: dict[tuple[str, bool, str, float], dict[str, int] | None] = {}
    rare_count_cache: dict[tuple[str, int, str], dict[str, int]] = {}

    def split_totals_for(path: Path, percent: float = 100) -> dict[str, int] | None:
        cache_key = (str(path.resolve()), lowercase, tokenizer, float(percent))
        if cache_key not in split_count_cache:
            split_count_cache[cache_key] = count_split_targets(path, lowercase, tokenizer, percent)
        return split_count_cache[cache_key]

    def rare_totals_for(path: Path, order: int) -> dict[str, int]:
        cache_key = (str(path.resolve()), order, tokenizer)
        if cache_key not in rare_count_cache:
            rare_count_cache[cache_key] = count_rare_targets(path, order, tokenizer)
        return rare_count_cache[cache_key]

    rare_requested = bool(data.get("rare_check", data.get("rare-check", False)))
    rare_orders_expected = sorted(set(rare_paths) | ({2, 3} if rare_requested else set()))
    if rare_requested:
        for order in (2, 3):
            if order not in rare_paths:
                issues.append(f"rare-seq-{ 'bigram' if order == 2 else 'trigram' }.csv missing from {corpus_root}")

    for percent in percentages:
        percent_dir = run_base / f"train_{percent:g}pct"
        if not percent_dir.is_dir() and (run_base / "models").is_dir():
            percent_dir = run_base
        snapshot = percent_dir / "train.txt"
        train_source = snapshot if snapshot.is_file() else corpus_root / "train.txt"
        train_percent = 100.0 if snapshot.is_file() else percent
        split_totals = {
            "train": split_totals_for(train_source, train_percent),
            "valid": split_totals_for(corpus_root / "valid.txt"),
            "test": split_totals_for(corpus_root / "test.txt"),
        }
        for source, totals in split_totals.items():
            if totals is None:
                issues.append(f"{percent:g}% {source} input file unavailable; cannot verify row totals")
        if not snapshot.is_file():
            issues.append(f"{percent:g}%: exact training snapshot missing ({snapshot})")

        expected_units: list[tuple[tuple[float, str, str, str, str], dict[str, int] | None]] = []
        # Statistical baselines are evaluated at orders 2 and 3, with each
        # matching rare-sequence file included when present.
        for spec in NGRAM_SPECS:
            label, order = spec["name"], spec["order"]
            for source, totals in split_totals.items():
                expected_units.append(((round(percent, 8), label, "", source, str(order)), totals))
            if order in rare_orders_expected:
                expected_units.append((
                    (round(percent, 8), label, "", "rare", str(order)),
                    rare_totals_for(rare_paths[order], order) if order in rare_paths else None,
                ))

        # Neural evaluation rows use the saved model label and seed.
        for folder, model_name, seed, context_size in models:
            label = folder
            for source, totals in split_totals.items():
                neural_order = 2 if label.startswith("ffnn_ctx1") else 3 if label.startswith("ffnn_ctx2") else ""
                expected_units.append((
                    (round(percent, 8), label, seed, source, str(neural_order) if neural_order else ""),
                    totals,
                ))
            if label.startswith("ffnn_ctx1"):
                rare_orders = [2]
            elif label.startswith("ffnn_ctx2"):
                rare_orders = [3]
            else:
                rare_orders = [2, 3]
            for order in rare_orders:
                if order in rare_orders_expected:
                    expected_units.append((
                        (round(percent, 8), label, seed, "rare", str(order)),
                        rare_totals_for(rare_paths[order], order) if order in rare_paths else None,
                    ))

        done = mismatched = 0
        for key, expected_totals in expected_units:
            rows = saved_rows.get(key, [])
            if not rows:
                issues.append(f"{percent:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}: metric row missing")
                continue
            if expected_totals is None:
                mismatched += 1
                issues.append(f"{percent:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}: source totals unavailable")
                continue
            valid_rows = [row for row in rows if metric_row_valid(row, expected_totals)]
            cores = {metric_core(row) for row in valid_rows}
            if len(cores) != 1:
                mismatched += 1
                detail = "conflicting metric rows" if valid_rows else "invalid metric row"
                issues.append(
                    f"{percent:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}: {detail}"
                )
                continue
            row = valid_rows[0]
            observed = {}
            try:
                observed = {
                    field: int(float(row[field]))
                    for field in ("sentences", "word_tokens", "prediction_events")
                }
            except (KeyError, TypeError, ValueError):
                pass
            if observed == expected_totals:
                done += 1
            else:
                mismatched += 1
                issues.append(
                    f"{percent:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}: "
                    f"counts saved={observed or 'invalid'}, expected={expected_totals}"
                )

        expected_count = len(expected_units)
        overall_complete += done
        overall_expected += expected_count
        pending = expected_count - done
        print(f"  {percent:g}%: {done}/{expected_count} evaluation rows complete; {pending} pending/mismatched")
        for folder, model_name, seed, _context in models:
            checkpoint = percent_dir / "models" / folder / f"{model_name}_seed{seed}.pt"
            if not checkpoint.is_file():
                issues.append(f"{percent:g}% {folder} seed={seed}: model checkpoint missing")

    print(f"Overall metrics progress: {overall_complete}/{overall_expected} complete; "
          f"{overall_expected - overall_complete} pending/mismatched")
    if issues:
        print("Pending or inconsistent items:")
        for issue in issues[:50]:
            print(f"  - {issue}")
        if len(issues) > 50:
            print(f"  ... plus {len(issues) - 50} more")

    report_paths = [
        run_base / "metrics" / name for name in (
            "metrics.csv", "perplexity_by_percentage.csv", "predictions.csv",
            "predictions_correct.csv", "predictions_incorrect.csv", "charts.png",
        )
    ]
    for percent in percentages:
        plot_percent = f"{percent:g}".replace(".", "_")
        report_paths.append(run_base / "metrics" / f"plot_{plot_percent}p.png")
    missing_reports = [path.name for path in report_paths if not path.is_file() or path.stat().st_size == 0]
    print("Final report files: " + ("all present" if not missing_reports else "missing/empty: " + ", ".join(missing_reports)))
    resume_artifacts = [
        run_base / "metrics" / name
        for name in (".metrics.resume.csv", ".predictions.resume.csv")
        if (run_base / "metrics" / name).is_file()
    ]
    if resume_artifacts:
        print(
            "Finalization pending: resume journal(s) still present: "
            + ", ".join(path.name for path in resume_artifacts)
        )
    complete = (
        overall_complete == overall_expected
        and not missing_reports
        and not resume_artifacts
    )
    print(f"Comparison status: {'COMPLETE' if complete else 'PENDING'}")
    return {
        "complete": overall_complete,
        "expected": overall_expected,
        "reports_complete": not missing_reports,
        "missing_reports": missing_reports,
        "all_complete": complete,
    }


def plot_metrics(metrics: list[dict], output_path: Path, corpus: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(metrics)
    frame["perplexity_numeric"] = pd.to_numeric(frame["perplexity"], errors="coerce")
    frame["percent_numeric"] = pd.to_numeric(frame["percent"], errors="coerce")
    sources = [source for source in ("train", "valid", "test", "rare") if source in set(frame["source"])]
    if not sources:
        print("No metrics available for chart generation.")
        return
    fig, axes = plt.subplots(len(sources), 2, figsize=(14, 3.6 * len(sources)), squeeze=False)
    groups = [
        (2, "Bigram models", {"bigram_mle", "bigram_laplace", "bigram_backoff", "ffnn_ctx1", "lstm"}),
        (3, "Trigram models", {"trigram_mle", "trigram_laplace", "trigram_backoff", "ffnn_ctx2", "lstm"}),
    ]
    def belongs_to_order(name: str, order: int, allowed: set[str]) -> bool:
        if name in allowed:
            return True
        return name.startswith("ffnn_ctx1_") if order == 2 else name.startswith("ffnn_ctx2_")

    for row_index, source in enumerate(sources):
        for column_index, (order, title, allowed) in enumerate(groups):
            ax = axes[row_index][column_index]
            relevant = frame["model"].astype(str).map(lambda name: belongs_to_order(name, order, allowed))
            subset = frame[(frame["source"] == source) & relevant].copy()
            if source == "rare":
                subset = subset[pd.to_numeric(subset["order"], errors="coerce") == order]
            if subset.empty:
                ax.text(0.5, 0.5, "No results", ha="center", va="center", transform=ax.transAxes)
            else:
                for model_name, model_rows in subset.groupby("model", sort=True):
                    grouped = model_rows.groupby("percent_numeric", dropna=True)["perplexity_numeric"].mean()
                    finite = grouped[(grouped > 0) & grouped.map(math.isfinite)]
                    if not finite.empty:
                        ax.plot(finite.index, finite.values, marker="o", label=model_name)
                    infinite_points = model_rows[model_rows["status"] == "infinite_zero_probability"]
                    for _, item in infinite_points.iterrows():
                        ax.annotate("∞", (item["percent_numeric"], ax.get_ylim()[1] or 1),
                                    ha="center", va="bottom", fontsize=9)
            ax.set_title(f"{title} · {source}")
            ax.set_xlabel("Training data (%)")
            ax.set_ylabel("Perplexity (log scale)")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.25)
            if not subset.empty:
                ax.legend(fontsize=8)
    fig.suptitle(f"{corpus}: perplexity by model, split, and training percentage", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def format_accuracy_percent(value: float) -> str:
    """Format an accuracy stored as a 0-1 fraction for display."""
    return "" if math.isnan(value) else f"{value:.1%}"


def plot_percentage_metrics(
    metrics: list[dict],
    output_dir: Path,
    corpus: str,
    filename_suffix: str = "",
) -> list[Path]:
    """Write fixed-percentage accuracy/perplexity comparisons like the notebook."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame(metrics)
    if frame.empty:
        return []
    for column in ("percent", "accuracy", "perplexity", "prediction_events",
                   "word_tokens", "correct_predictions"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    colors = {
        "train": "#6f9e75",
        "valid": "#8ab6d6",
        "test": "#e39a63",
        "rare": "#b08bbb",
    }
    written = []

    def split_values(rows: "pd.DataFrame", source: str) -> tuple[float, float]:
        """Average seeds; combine rare bigram/trigram rows by their event counts."""
        source_rows = rows[rows["source"] == source]
        if source_rows.empty:
            return np.nan, np.nan
        if source != "rare":
            accuracies = source_rows["accuracy"].dropna()
            perplexities = source_rows["perplexity"].dropna().astype(float).tolist()
            accuracy = float(accuracies.mean()) if not accuracies.empty else np.nan
            if any(math.isinf(value) for value in perplexities):
                perplexity = math.inf
            else:
                perplexity = float(np.mean(perplexities)) if perplexities else np.nan
            return accuracy, perplexity

        # LSTM can have a rare row for each n-gram order. Combine those rows
        # within each seed before averaging seeds so longer rare lists carry
        # their proper weight.
        accuracies, perplexities = [], []
        for _, seed_rows in source_rows.groupby("seed", dropna=False):
            word_count = seed_rows["word_tokens"].fillna(0).sum()
            correct = seed_rows["correct_predictions"].fillna(0).sum()
            if word_count:
                accuracies.append(float(correct / word_count))

            event_count = seed_rows["prediction_events"].fillna(0).sum()
            ppl_values = seed_rows["perplexity"].dropna().astype(float).tolist()
            if any(math.isinf(value) for value in ppl_values):
                perplexities.append(math.inf)
            elif event_count and ppl_values:
                log_total = sum(
                    float(row["prediction_events"]) * math.log(float(row["perplexity"]))
                    for _, row in seed_rows.iterrows()
                    if row["prediction_events"] > 0 and pd.notna(row["perplexity"])
                )
                mean_nll = log_total / event_count
                perplexities.append(
                    math.inf if mean_nll >= math.log(float.fromhex("0x1.fffffffffffffp+1023"))
                    else math.exp(mean_nll)
                )
        accuracy = float(np.mean(accuracies)) if accuracies else np.nan
        if any(math.isinf(value) for value in perplexities):
            perplexity = math.inf
        else:
            perplexity = float(np.mean(perplexities)) if perplexities else np.nan
        return accuracy, perplexity

    for percent in sorted(frame["percent"].dropna().unique()):
        percent_rows = frame[frame["percent"] == percent]
        sources = [name for name in ("train", "valid", "test", "rare")
                   if name in set(percent_rows["source"])]
        models = sorted(percent_rows["model"].dropna().astype(str).unique())
        if not sources or not models:
            continue

        values = {
            (model, source): split_values(
                percent_rows[percent_rows["model"].astype(str) == model], source
            )
            for model in models for source in sources
        }
        positions = np.arange(len(models))
        width = 0.82 / len(sources)
        fig_width = max(14, len(models) * 1.45)
        fig, (accuracy_ax, perplexity_ax) = plt.subplots(1, 2, figsize=(fig_width, 6))
        finite_perplexities = [
            ppl for _, ppl in values.values()
            if not math.isnan(ppl) and math.isfinite(ppl) and ppl > 0
        ]
        infinity_y = max(finite_perplexities, default=1.0) * 1.5

        for split_index, source in enumerate(sources):
            offsets = positions + (split_index - (len(sources) - 1) / 2) * width
            accuracies = [values[(model, source)][0] for model in models]
            accuracy_bars = accuracy_ax.bar(
                offsets,
                [value * 100 if not math.isnan(value) else np.nan for value in accuracies],
                width,
                color=colors[source],
                edgecolor="white",
                label=source,
            )
            accuracy_ax.bar_label(
                accuracy_bars,
                labels=[format_accuracy_percent(value) for value in accuracies],
                padding=2,
                fontsize=7,
                rotation=90,
            )

            finite_x, finite_y, infinite_x = [], [], []
            for offset, model in zip(offsets, models):
                perplexity = values[(model, source)][1]
                if math.isinf(perplexity):
                    infinite_x.append(offset)
                elif not math.isnan(perplexity) and perplexity > 0:
                    finite_x.append(offset)
                    finite_y.append(perplexity)
            if finite_x:
                perplexity_ax.bar(
                    finite_x, finite_y, width, color=colors[source], edgecolor="white"
                )
            for offset in infinite_x:
                perplexity_ax.scatter(
                    [offset], [infinity_y], marker="^", s=50,
                    color=colors[source], zorder=4,
                )
                perplexity_ax.annotate(
                    "∞", (offset, infinity_y), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9,
                )

        accuracy_ax.set_title("Top-1 accuracy")
        accuracy_ax.set_ylabel("Correct word targets (%)")
        max_accuracy = percent_rows["accuracy"].max(skipna=True)
        accuracy_ax.set_ylim(0, min(100, max(5, float(max_accuracy) * 120
                                            if pd.notna(max_accuracy) else 5)))
        accuracy_ax.set_xticks(positions, models, rotation=35, ha="right")
        accuracy_ax.grid(axis="y", alpha=0.25)

        perplexity_ax.set_title("Perplexity")
        perplexity_ax.set_ylabel("Perplexity (log scale; ∞ marks zero-probability cases)")
        perplexity_ax.set_yscale("log")
        perplexity_ax.set_ylim(bottom=1, top=max(10, infinity_y * 3))
        perplexity_ax.set_xticks(positions, models, rotation=35, ha="right")
        perplexity_ax.grid(axis="y", which="both", alpha=0.25)

        legend = [Patch(facecolor=colors[source], edgecolor="white", label=source)
                  for source in sources]
        fig.legend(handles=legend, title="Split", loc="upper center",
                   ncol=len(sources), bbox_to_anchor=(0.5, 0.98))
        fig.suptitle(
            f"{corpus}: model comparison at {percent:g}% training data",
            fontsize=14,
            y=1.04,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        percent_label = f"{percent:g}".replace(".", "_")
        output_path = output_dir / f"plot_{percent_label}p{filename_suffix}.png"
        fig.savefig(output_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        written.append(output_path)
    return written


def make_perplexity_summary(metrics: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in metrics:
        key = (row["percent"], row["model"], row["seed"])
        grouped[key][row["source"]].append(row)
    rows = []
    for (percent, model, seed), sources in sorted(grouped.items()):
        fields = {}
        for source, source_rows in sources.items():
            if source != "rare" or len(source_rows) == 1:
                fields[source + "_perplexity"] = source_rows[0].get("perplexity", "")
                fields[source + "_status"] = source_rows[0].get("status", "")
                continue
            events = sum(int(item.get("prediction_events") or 0) for item in source_rows)
            zeros = sum(int(item.get("zero_probability_events") or 0) for item in source_rows)
            log_sum = sum(
                int(item.get("prediction_events") or 0) * math.log(float(item["perplexity"]))
                for item in source_rows
                if item.get("perplexity") is not None and math.isfinite(float(item["perplexity"]))
            )
            overflow = any(item.get("status") == "overflow" for item in source_rows)
            if zeros:
                fields["rare_perplexity"] = math.inf
                fields["rare_status"] = "infinite_zero_probability"
            elif overflow:
                fields["rare_perplexity"] = math.inf
                fields["rare_status"] = "overflow"
            else:
                fields["rare_perplexity"] = math.exp(log_sum / events) if events else ""
                fields["rare_status"] = "finite"
        rows.append({
            "percent": percent,
            "model": model,
            "seed": seed,
            "train_perplexity": fields.get("train_perplexity", ""),
            "valid_perplexity": fields.get("valid_perplexity", ""),
            "test_perplexity": fields.get("test_perplexity", ""),
            "rare_perplexity": fields.get("rare_perplexity", ""),
            "train_status": fields.get("train_status", ""),
            "valid_status": fields.get("valid_status", ""),
            "test_status": fields.get("test_status", ""),
            "rare_status": fields.get("rare_status", ""),
        })
    return rows


def run(
    corpus: str,
    device_override: str | None = None,
    resume: bool = False,
    prediction_source: Path | None = None,
    config_override: str | Path | None = None,
) -> int:
    config, config_path = load_corpus_config(corpus, config_override)
    if config.get("execution", {}).get("colab_only"):
        if not os.environ.get("COLAB_RELEASE_TAG") or not torch.cuda.is_available():
            raise RuntimeError("This experiment may only fit count tables in a Colab GPU runtime")
    if config.get("data", {}).get("prepared_manifest"):
        from run import verify_prepared_inputs
        verify_prepared_inputs(config)
    data = config.get("data", {})
    data_root = project_path(data.get("output", "data/processed"))
    corpus_dir = data_root / corpus
    valid_path, test_path = corpus_dir / "valid.txt", corpus_dir / "test.txt"
    if not valid_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Expected prepared valid.txt and test.txt under {corpus_dir}")

    run_base = project_path(config.get("output_dir", "results")) / config.get("name", corpus)
    percent_folders = discover_percent_folders(run_base, config)
    report_dir = run_base / "metrics"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "metrics": report_dir / "metrics.csv",
        "summary": report_dir / "perplexity_by_percentage.csv",
        "predictions": report_dir / "predictions.csv",
        "correct": report_dir / "predictions_correct.csv",
        "incorrect": report_dir / "predictions_incorrect.csv",
        "chart": report_dir / "charts.png",
        "events": report_dir / "scoring_events.jsonl.gz",
    }
    journal_files = {
        "metrics": report_dir / ".metrics.resume.csv",
        "predictions": report_dir / ".predictions.resume.csv",
    }
    print(f"Corpus: {corpus} | config: {display_path(config_path)}")
    print(f"Result runs: {display_path(run_base)}")
    print(f"Writing comparison files to {display_path(report_dir)}")

    rare_paths = {
        order: corpus_dir / f"rare-seq-{name}.csv"
        for order, name in ((2, "bigram"), (3, "trigram"))
    }
    rare_paths = {order: path for order, path in rare_paths.items() if path.is_file()}
    if not rare_paths:
        print("No rare-seq CSVs found; rare evaluation will be omitted.")

    cli = LanguageModelCLI(device_override or config.get("device", "auto"))

    # Build the expected unit/count plan once. This is the same accounting used
    # by --check, but it follows the actual saved checkpoints and their frozen
    # preprocessing metadata rather than assuming the source YAML is unchanged.
    percent_runs = []
    expected_totals: dict[tuple[float, str, str, str, str], dict[str, int]] = {}
    expected_signatures: dict[tuple[float, str, str, str, str], str] = {}
    expected_order: list[tuple[float, str, str, str, str]] = []
    split_count_cache = {}
    rare_count_cache = {}
    file_hash_cache: dict[str, str] = {}

    def hashed(path: Path) -> str:
        resolved = str(path.resolve())
        if resolved not in file_hash_cache:
            file_hash_cache[resolved] = file_sha256(path)
        return file_hash_cache[resolved]

    scoring_code_signature = stable_signature({
        str(path.relative_to(PROJECT_ROOT)): hashed(path)
        for path in (
            Path(__file__).resolve(),
            PROJECT_ROOT / "src/lm/evaluation.py",
            PROJECT_ROOT / "src/lm/cli.py",
            PROJECT_ROOT / "src/lm/statistical/ngram.py",
            PROJECT_ROOT / "src/lm/tokenization.py",
        )
    })

    def add_expected(key, totals, signature_payload: dict) -> None:
        signature = stable_signature(signature_payload)
        if key in expected_totals:
            if expected_totals[key] != totals or expected_signatures[key] != signature:
                raise ValueError(f"Conflicting expected plan for evaluation unit: {key}")
            return
        expected_totals[key] = totals
        expected_signatures[key] = signature
        expected_order.append(key)

    for percent, percent_dir in percent_folders:
        checkpoints = checkpoint_paths(percent_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No saved neural checkpoints found under {percent_dir}")
        first_key = cli.load_neural(checkpoints[0])
        first_loaded = cli.neural[first_key]
        first = {key: value for key, value in first_loaded.items() if key != "model"}
        del cli.neural[first_key], first_loaded
        if cli.device.type == "cuda":
            torch.cuda.empty_cache()
        elif cli.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

        train_path = first["train_path"]
        if not train_path.is_file():
            raise FileNotFoundError(f"Exact training snapshot for this run is missing: {train_path}")
        lowercase = first["lowercase"]
        tokenizer = first["tokenizer"]
        train_percent = first.get("train_file_percent", 100)

        def split_totals(path: Path, used_percent: float = 100) -> dict[str, int]:
            cache_key = (str(path.resolve()), lowercase, tokenizer, float(used_percent))
            if cache_key not in split_count_cache:
                split_count_cache[cache_key] = count_split_targets(
                    path, lowercase, tokenizer, used_percent
                )
            totals = split_count_cache[cache_key]
            if totals is None:
                raise FileNotFoundError(f"Evaluation split is missing: {path}")
            return totals

        split_totals_by_source = {
            "train": split_totals(train_path, train_percent),
            "valid": split_totals(valid_path),
            "test": split_totals(test_path),
        }
        rare_totals_by_order = {}
        for order, path in rare_paths.items():
            cache_key = (str(path.resolve()), order, tokenizer)
            if cache_key not in rare_count_cache:
                rare_count_cache[cache_key] = count_rare_targets(path, order, tokenizer)
            rare_totals_by_order[order] = rare_count_cache[cache_key]

        print(f"Fingerprinting comparison inputs for {percent:g}%...", flush=True)
        source_hashes = {
            "train": hashed(train_path),
            "valid": hashed(valid_path),
            "test": hashed(test_path),
        }
        rare_hashes = {order: hashed(path) for order, path in rare_paths.items()}
        common_identity = {
            "schema": 1,
            "corpus": corpus,
            "percent": percent,
            "lowercase": lowercase,
            "tokenizer": tokenizer,
            "vocab": stable_signature(first["vocab"]),
            "allowed_target_ids": stable_signature(
                first.get("allowed_target_ids", "vocab")
            ),
            "scoring_code": scoring_code_signature,
        }

        for spec in NGRAM_SPECS:
            label, order = spec["name"], spec["order"]
            for source, totals in split_totals_by_source.items():
                add_expected(
                    (round(percent, 8), label, "", source, str(order)),
                    totals,
                    {
                        **common_identity,
                        "model": {"family": "ngram", **spec},
                        "training_input": source_hashes["train"],
                        "evaluation_source": source,
                        "evaluation_input": source_hashes[source],
                        "order": order,
                    },
                )
            if order in rare_totals_by_order:
                add_expected(
                    (round(percent, 8), label, "", "rare", str(order)),
                    rare_totals_by_order[order],
                    {
                        **common_identity,
                        "model": {"family": "ngram", **spec},
                        "training_input": source_hashes["train"],
                        "evaluation_source": "rare",
                        "evaluation_input": rare_hashes[order],
                        "order": order,
                    },
                )

        checkpoint_identities = {}
        for checkpoint in checkpoints:
            label, seed, order = checkpoint_identity(checkpoint)
            checkpoint_identities[checkpoint] = (label, seed, order)
            print(f"  fingerprint checkpoint: {checkpoint.name}", flush=True)
            checkpoint_identity_hash = hashed(checkpoint)
            for source, totals in split_totals_by_source.items():
                add_expected(
                    (round(percent, 8), label, seed, source, order),
                    totals,
                    {
                        **common_identity,
                        "model": {
                            "family": "neural",
                            "label": label,
                            "seed": seed,
                            "checkpoint": checkpoint_identity_hash,
                        },
                        "training_input": source_hashes["train"],
                        "evaluation_source": source,
                        "evaluation_input": source_hashes[source],
                        "order": order,
                    },
                )
            if order == "2":
                rare_orders = (2,)
            elif order == "3":
                rare_orders = (3,)
            else:
                rare_orders = tuple(sorted(rare_totals_by_order))
            for rare_order in rare_orders:
                if rare_order in rare_totals_by_order:
                    add_expected(
                        (round(percent, 8), label, seed, "rare", str(rare_order)),
                        rare_totals_by_order[rare_order],
                        {
                            **common_identity,
                            "model": {
                                "family": "neural",
                                "label": label,
                                "seed": seed,
                                "checkpoint": checkpoint_identity_hash,
                            },
                            "training_input": source_hashes["train"],
                            "evaluation_source": "rare",
                            "evaluation_input": rare_hashes[rare_order],
                            "order": rare_order,
                        },
                    )

        percent_runs.append({
            "percent": percent,
            "percent_dir": percent_dir,
            "checkpoints": checkpoints,
            "checkpoint_identities": checkpoint_identities,
            "first": first,
            "train_path": train_path,
        })

    base_prediction_source = None
    base_stats = {}
    journal_stats = {}
    completed_rows: dict[tuple[float, str, str, str, str], dict] = {}
    metric_origins: dict[tuple[float, str, str, str, str], str] = {}
    prediction_origins: dict[tuple[float, str, str, str, str], str] = {}
    issues: list[str] = []
    if resume:
        base_prediction_source = select_prediction_source(report_dir, prediction_source)
        completed_rows, metric_origins, issues = load_resumable_metrics(
            [
                ("canonical", output_files["metrics"]),
                ("journal", journal_files["metrics"]),
            ],
            expected_totals,
            expected_signatures,
        )
        if base_prediction_source:
            shown = (
                base_prediction_source.relative_to(PROJECT_ROOT)
                if base_prediction_source.is_relative_to(PROJECT_ROOT)
                else base_prediction_source
            )
            print(f"Resume prediction source: {shown}", flush=True)
        else:
            print("Resume prediction source: none; completed held-out rows will be rescored.")

        base_stats = scan_prediction_stats(base_prediction_source, "preserved export")
        journal_stats = scan_prediction_stats(journal_files["predictions"], "resume journal")
        for key in list(completed_rows):
            if key[3] == "train":
                prediction_origins[key] = "none"
            elif prediction_group_valid(completed_rows[key], base_stats.get(key)):
                prediction_origins[key] = "base"
            elif prediction_group_valid(completed_rows[key], journal_stats.get(key)):
                prediction_origins[key] = "journal"
            else:
                issues.append(
                    f"will rescore legacy or incomplete metric/prediction group: "
                    f"{key[0]:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}"
                )
                completed_rows.pop(key)
                metric_origins.pop(key)
    elif prediction_source is not None:
        raise ValueError("--prediction-source requires --resume")

    event_journal = report_dir / ".scoring_events.resume.jsonl.gz"
    regular_units = {key for key in expected_order if key[3] != "rare"}
    reuse_event_archive = False
    if resume and output_files["events"].is_file() and output_files["metrics"].is_file():
        try:
            verification = verify_scoring_events(
                output_files["metrics"], output_files["events"]
            )
            reuse_event_archive = (
                verification.get("passed") is True
                and int(verification.get("conditions", -1)) == len(regular_units)
                and regular_units.issubset(completed_rows)
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            issues.append(f"will rebuild scoring-event archive: {exc}")
    if not reuse_event_archive:
        # A complete event archive cannot be assembled from skipped metric units.
        # Rescore regular splits while retaining any validated rare-only units.
        for key in regular_units:
            completed_rows.pop(key, None)
            metric_origins.pop(key, None)
            prediction_origins.pop(key, None)
        event_journal.unlink(missing_ok=True)
    else:
        print("Resume scoring-event archive: verified and reusable.", flush=True)

    for issue in issues:
        print(f"Resume note: {issue}", flush=True)

    journal_keys = {key for key, origin in prediction_origins.items() if origin == "journal"}
    if resume:
        write_metric_csv(
            journal_files["metrics"],
            [
                completed_rows[key]
                for key in expected_order
                if key in completed_rows and metric_origins.get(key) == "journal"
            ],
        )
        rewrite_prediction_journal(journal_files["predictions"], journal_keys)
    else:
        write_metric_csv(journal_files["metrics"], [])
        rewrite_prediction_journal(journal_files["predictions"], set())

    print(
        f"Comparison resume: {len(completed_rows):,}/{len(expected_order):,} "
        f"metric units already complete; {len(expected_order) - len(completed_rows):,} pending.",
        flush=True,
    )

    event_context = (
        nullcontext(None)
        if reuse_event_archive
        else gzip.open(event_journal, "wt", encoding="utf-8")
    )
    with journal_files["metrics"].open("a", encoding="utf-8", newline="") as metric_handle, \
         journal_files["predictions"].open("a", encoding="utf-8", newline="") as prediction_handle, \
         event_context as event_handle:
        metric_writer = csv.DictWriter(metric_handle, fieldnames=METRIC_FIELDS)
        writers = (
            csv.DictWriter(prediction_handle, fieldnames=PREDICTION_FIELDS),
            NullCSVWriter(),
            NullCSVWriter(),
        )
        discard_writers = (NullCSVWriter(), NullCSVWriter(), NullCSVWriter())

        def record_metric(row: dict, prediction_origin: str) -> None:
            key = metric_key(row)
            prediction_handle.flush()
            os.fsync(prediction_handle.fileno())
            metric_writer.writerow(row)
            metric_handle.flush()
            os.fsync(metric_handle.fileno())
            completed_rows[key] = row
            metric_origins[key] = "journal"
            prediction_origins[key] = prediction_origin

        def evaluate_pending(unit, evaluator) -> dict:
            """Score one unit and reuse an exactly matching preserved group."""
            can_try_base = (
                resume
                and unit[3] != "train"
                and bool(base_stats.get(unit, {}).get("rows"))
            )
            row = evaluator(
                discard_writers if can_try_base else writers,
                event_handle if unit[3] != "rare" else None,
            )
            if can_try_base and prediction_group_valid(row, base_stats.get(unit)):
                print(
                    f"  resume: reused preserved predictions for {unit[1]} / {unit[3]}"
                    + (f" seed={unit[2]}" if unit[2] else "")
                    + (f" order={unit[4]}" if unit[3] == "rare" else ""),
                    flush=True,
                )
                record_metric(row, "base")
                return row
            if can_try_base:
                print(
                    f"  resume: preserved predictions did not exactly match {unit[1]} / "
                    f"{unit[3]}; regenerating that unit",
                    flush=True,
                )
                # The first pass already emitted the scoring events; this pass
                # exists only to regenerate an incompatible prediction group.
                row = evaluator(writers, None)
            record_metric(row, "none" if unit[3] == "train" else "journal")
            return row

        def skip_message(key) -> None:
            print(
                f"  resume: skipping completed {key[1]} / {key[3]}"
                + (f" seed={key[2]}" if key[2] else "")
                + (f" order={key[4]}" if key[3] == "rare" else ""),
                flush=True,
            )

        for run_info in percent_runs:
            percent = run_info["percent"]
            percent_dir = run_info["percent_dir"]
            print(f"\n== {percent:g}% training data: {percent_dir.relative_to(PROJECT_ROOT)} ==", flush=True)
            checkpoints = run_info["checkpoints"]
            first = run_info["first"]
            vocab = Vocab.from_list(first["vocab"])
            train_path = run_info["train_path"]
            print(f"Train input: {train_path.relative_to(PROJECT_ROOT) if train_path.is_relative_to(PROJECT_ROOT) else train_path}")
            print(f"Vocabulary: {len(vocab):,}; tokenizer={first['tokenizer']}; checkpoints={len(checkpoints)}")

            ngram_pending = any(
                key[0] == round(percent, 8)
                and key[1] in {spec["name"] for spec in NGRAM_SPECS}
                and key not in completed_rows
                for key in expected_order
            )
            ngram_models = {}
            ngram_entries = []
            if ngram_pending:
                ngram_models, _ = make_ngram_variants(
                    train_path, vocab, first["lowercase"], first["tokenizer"],
                    first.get("train_file_percent", 100),
                )
                with gzip.open(percent_dir / "count_models.pkl.gz", "wb") as count_handle:
                    pickle.dump(
                        {
                            "schema_version": 1,
                            "variants": ngram_models,
                            "vocab": vocab.itos,
                            "train_sha256": file_sha256(train_path),
                            "tokenizer": first["tokenizer"],
                            "lowercase": first["lowercase"],
                        },
                        count_handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                ngram_entries = [
                    (label, {
                        "family": item["family"],
                        "order": item["order"],
                        "seed": "",
                        "lowercase": first["lowercase"],
                        "tokenizer": first["tokenizer"],
                        "allowed_target_ids": "vocab",
                        "train_seconds": item["train_seconds"],
                        "train_file_percent": first.get("train_file_percent", 100),
                    }, item["model"])
                    for label, item in ngram_models.items()
                ]
            else:
                print("  resume: all statistical metric units already complete; counts not rebuilt.")

            for label, info, model in ngram_entries:
                # Shared count tables, with smoothing selected per model before scoring.
                spec = ngram_models[label]["spec"]
                model.smoothing = spec["smoothing"]
                model.k = float(spec.get("k", 1.0))
                model.discount = float(spec.get("discount", 0.75))
                model.model.smoothing = model.smoothing
                model.model.k = model.k
                model.model.d = model.discount
                model.model._alpha_cache.clear()
                model._backoff_prediction_cache.clear()
                for source, path in (("train", train_path), ("valid", valid_path), ("test", test_path)):
                    unit = (round(percent, 8), label, "", source, str(info["order"]))
                    if unit in completed_rows:
                        skip_message(unit)
                        continue
                    info["evaluation_signature"] = expected_signatures[unit]
                    evaluate_pending(
                        unit,
                        lambda selected_writers, selected_event_handle,
                               path=path, source=source: evaluate_split(
                            corpus, percent, label, info, model, vocab,
                            path, source, cli, selected_writers,
                            selected_event_handle,
                        ),
                    )
                for order, rare_path in rare_paths.items():
                    if info["order"] != order:
                        continue
                    unit = (round(percent, 8), label, "", "rare", str(order))
                    if unit in completed_rows:
                        skip_message(unit)
                        continue
                    info["evaluation_signature"] = expected_signatures[unit]
                    evaluate_pending(
                        unit,
                        lambda selected_writers, _selected_event_handle,
                               rare_path=rare_path, order=order: evaluate_rare_file(
                            corpus, percent, label, info, model, vocab, rare_path,
                            order, cli, selected_writers,
                        ),
                    )

            # Neural model memory can be substantial for large vocabularies;
            # load, evaluate, and release one checkpoint before the next.
            for checkpoint in checkpoints:
                label, seed, matched_order = run_info["checkpoint_identities"][checkpoint]
                neural_units = [
                    (round(percent, 8), label, seed, source, matched_order)
                    for source in ("train", "valid", "test")
                ]
                if matched_order == "2":
                    neural_rare_orders = (2,)
                elif matched_order == "3":
                    neural_rare_orders = (3,)
                else:
                    neural_rare_orders = tuple(sorted(rare_paths))
                neural_units.extend(
                    (round(percent, 8), label, seed, "rare", str(order))
                    for order in neural_rare_orders if order in rare_paths
                )
                if all(unit in completed_rows for unit in neural_units):
                    for unit in neural_units:
                        skip_message(unit)
                    continue

                key = cli.load_neural(checkpoint)
                loaded = cli.neural[key]
                label = model_label(checkpoint, loaded["name"])
                info = dict(loaded)
                result_path = checkpoint.with_name(checkpoint.stem + "_result.json")
                fit_result = {}
                if result_path.is_file():
                    fit_result = json.loads(result_path.read_text(encoding="utf-8"))["result"]
                info.update({
                    "family": "neural",
                    "order": int(matched_order) if matched_order else "",
                    "seed": seed,
                    "train_seconds": fit_result.get("train_seconds", ""),
                })
                if tuple(info["vocab"]) != tuple(vocab.itos):
                    raise ValueError(f"{checkpoint} uses a different vocabulary from its percentage run")
                if (info["tokenizer"] != first["tokenizer"]
                        or info["lowercase"] != first["lowercase"]
                        or info["train_path"].resolve() != train_path.resolve()):
                    raise ValueError(f"{checkpoint} uses different preprocessing or training data")
                model = info["model"]
                for source, path in (("train", train_path), ("valid", valid_path), ("test", test_path)):
                    unit = (round(percent, 8), label, seed, source, matched_order)
                    if unit in completed_rows:
                        skip_message(unit)
                        continue
                    info["evaluation_signature"] = expected_signatures[unit]
                    evaluate_pending(
                        unit,
                        lambda selected_writers, selected_event_handle,
                               path=path, source=source: evaluate_split(
                            corpus, percent, label, info, model, vocab,
                            path, source, cli, selected_writers,
                            selected_event_handle,
                        ),
                    )
                for order, rare_path in rare_paths.items():
                    if matched_order == "2" and order != 2:
                        continue
                    if matched_order == "3" and order != 3:
                        continue
                    unit = (round(percent, 8), label, seed, "rare", str(order))
                    if unit in completed_rows:
                        skip_message(unit)
                        continue
                    info["evaluation_signature"] = expected_signatures[unit]
                    evaluate_pending(
                        unit,
                        lambda selected_writers, _selected_event_handle,
                               rare_path=rare_path, order=order: evaluate_rare_file(
                            corpus, percent, label, info, model, vocab, rare_path,
                            order, cli, selected_writers,
                        ),
                    )
                del model, loaded, info, cli.neural[key]
                if cli.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif cli.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

    missing_units = [key for key in expected_order if key not in completed_rows]
    if missing_units:
        sample = ", ".join(str(key) for key in missing_units[:5])
        raise RuntimeError(
            f"Comparison ended with {len(missing_units)} metric unit(s) incomplete: {sample}"
        )

    base_keys = {key for key, origin in prediction_origins.items() if origin == "base"}
    journal_keys = {key for key, origin in prediction_origins.items() if origin == "journal"}
    merge_prediction_exports(
        base_prediction_source,
        base_keys,
        journal_files["predictions"],
        journal_keys,
        output_files,
        completed_rows,
    )
    metrics = [completed_rows[key] for key in expected_order]
    # Replace metrics only after prediction exports have been verified and
    # committed. A metric row is the authority that a prediction group is safe.
    write_metric_csv(output_files["metrics"], metrics)
    if not reuse_event_archive:
        os.replace(event_journal, output_files["events"])

    from lm.preparation import atomic_json
    verification = verify_scoring_events(output_files["metrics"], output_files["events"])
    atomic_json(report_dir / "event_verification.json", verification)

    summary_rows = make_perplexity_summary(metrics)
    if summary_rows:
        summary_tmp = output_files["summary"].with_name(output_files["summary"].name + ".tmp")
        with summary_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(summary_tmp, output_files["summary"])
    plot_metrics(metrics, output_files["chart"], corpus)
    percentage_plots = plot_percentage_metrics(metrics, report_dir, corpus)
    for path in journal_files.values():
        path.unlink(missing_ok=True)
    artifact_hashes = {
        path.relative_to(run_base).as_posix(): file_sha256(path)
        for path in sorted(run_base.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    atomic_json(
        report_dir / "artifact_manifest.json",
        {
            "schema_version": 1,
            "config_sha256": file_sha256(config_path),
            "files": artifact_hashes,
        },
    )
    for name, path in output_files.items():
        if path.exists():
            print(f"Wrote {name}: {path.relative_to(PROJECT_ROOT)}")
    for path in percentage_plots:
        print(f"Wrote percentage plot: {path.relative_to(PROJECT_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", help="corpus key from src/configs and data.yaml")
    parser.add_argument("--device", help="override configured device (auto, cpu, cuda, mps)")
    parser.add_argument(
        "--config", type=Path,
        help="explicit corpus configuration path (useful when several configs share a corpus key)",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="read existing metrics.csv and report complete versus pending evaluations without inference or writes",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=(
            "reuse validated metric rows and prediction exports, discard incomplete "
            "groups, and evaluate only missing units"
        ),
    )
    parser.add_argument(
        "--prediction-source",
        type=Path,
        help=(
            "existing full predictions CSV to preserve during --resume; by default "
            "the largest compatible predictions*.csv is selected"
        ),
    )
    args = parser.parse_args(argv)
    if args.check:
        if args.device or args.resume or args.prediction_source or args.config:
            parser.error(
                "--check cannot be combined with --device, --resume, "
                "--prediction-source, or --config"
            )
        check_comparison(args.corpus)
        return 0
    if args.prediction_source and not args.resume:
        parser.error("--prediction-source requires --resume")
    return run(
        args.corpus,
        device_override=args.device,
        resume=args.resume,
        prediction_source=args.prediction_source,
        config_override=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
