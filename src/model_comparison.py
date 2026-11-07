#!/usr/bin/env python3
"""Evaluate saved neural and count-based LMs across available train percentages.

Usage: python src/model_comparison.py <corpus> [--check]

The script discovers ``results/<run-name>/train_*pct`` folders (or a legacy
single results folder), evaluates each saved neural checkpoint and six n-gram
baselines on the exact saved training subset plus valid/test, then writes
metrics, perplexity summaries, prediction details, and plots under
``results/<run-name>/metrics/``. Evaluation events are streamed to CSV to avoid
keeping corpus-sized prediction tables in memory.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from collections import defaultdict
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
METRIC_FIELDS = [
    "corpus", "percent", "model", "family", "order", "seed", "source",
    "sentences", "word_tokens", "prediction_events", "correct_predictions",
    "accuracy", "perplexity", "status", "zero_probability_events", "oov_tokens",
    "oov_rate", "evaluation_seconds", "train_seconds",
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


def project_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_corpus_config(corpus: str) -> tuple[dict, Path]:
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
    if re.fullmatch(r"ffnn_ctx\d+(?:_\d+)?", folder) or folder == "lstm":
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
               source: str, result: dict, elapsed: float) -> dict:
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
    }


def report_progress(label: str, source: str, started: float, state: dict) -> None:
    now = time.monotonic()
    if now - state["last"] >= 20:
        print(f"  {label} / {source}: {state['events']:,} events ({now - started:.0f}s)", flush=True)
        state["last"] = now


def evaluate_split(corpus: str, percent: float, label: str, info: dict, model,
                   vocab: Vocab, path: Path, source: str, cli: LanguageModelCLI,
                   writers: tuple[csv.DictWriter, csv.DictWriter, csv.DictWriter]) -> dict:
    all_writer, correct_writer, incorrect_writer = writers
    started = time.perf_counter()
    progress_started = time.monotonic()
    state = {"events": 0, "last": progress_started}

    def sink(event: dict) -> None:
        state["events"] += 1
        report_progress(label, source, progress_started, state)
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
    row = metric_row(corpus, percent, label, info, source, result, elapsed)
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
                           expected_id: int, weight: int, context_words: list[str]) -> None:
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
                                       target_id, weight, mapped_context_words)
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
                    target_id, weights[index], contexts[index],
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
    row = metric_row(corpus, percent, label, info, "rare", result, elapsed)
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


def check_comparison(corpus: str, expected_percentages: list[float] | None = None) -> dict:
    """Read existing comparison CSVs and report completed versus pending evaluations."""
    config, config_path = load_corpus_config(corpus)
    data = config.get("data", {})
    corpus_root = project_path(data.get("output", "data/processed")) / corpus
    run_base = project_path(config.get("output_dir", "results")) / config.get("name", corpus)
    metrics_path = run_base / "metrics" / "metrics.csv"
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
    if metrics_path.is_file() and metrics_path.stat().st_size:
        try:
            with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    csv_row_count += 1
                    try:
                        saved_rows[metric_key(row)].append(row)
                    except (TypeError, ValueError):
                        continue
        except (OSError, csv.Error) as exc:
            print(f"Could not read {metrics_path}: {exc}")
            return 2

    print("Read-only model-comparison check (no model inference or report writes).")
    print(f"Corpus: {corpus} | metrics rows read: {csv_row_count:,}")
    print(f"Metrics CSV: {metrics_path}")
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
            if len(rows) > 1:
                mismatched += 1
                issues.append(f"{percent:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}: duplicate metric rows")
                continue
            if not rows:
                issues.append(f"{percent:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}: metric row missing")
                continue
            if expected_totals is None:
                mismatched += 1
                issues.append(f"{percent:g}% {key[1]} seed={key[2]} {key[3]} order={key[4]}: source totals unavailable")
                continue
            row = rows[0]
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
    complete = overall_complete == overall_expected and not missing_reports
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


def plot_percentage_metrics(metrics: list[dict], output_dir: Path, corpus: str) -> list[Path]:
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
                labels=[f"{value:.1f}%" if not math.isnan(value) else "" for value in accuracies],
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
        output_path = output_dir / f"plot_{percent_label}p.png"
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


def run(corpus: str, device_override: str | None = None) -> int:
    config, config_path = load_corpus_config(corpus)
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
    }
    print(f"Corpus: {corpus} | config: {config_path.relative_to(PROJECT_ROOT)}")
    print(f"Result runs: {run_base.relative_to(PROJECT_ROOT)}")
    print(f"Writing comparison files to {report_dir.relative_to(PROJECT_ROOT)}")

    metrics: list[dict] = []
    rare_paths = {
        order: corpus_dir / f"rare-seq-{name}.csv"
        for order, name in ((2, "bigram"), (3, "trigram"))
    }
    rare_paths = {order: path for order, path in rare_paths.items() if path.is_file()}
    if not rare_paths:
        print("No rare-seq CSVs found; rare evaluation will be omitted.")

    cli = LanguageModelCLI(device_override or config.get("device", "auto"))
    with output_files["metrics"].open("w", encoding="utf-8", newline="") as metric_handle, \
         output_files["predictions"].open("w", encoding="utf-8", newline="") as prediction_handle, \
         output_files["correct"].open("w", encoding="utf-8", newline="") as correct_handle, \
         output_files["incorrect"].open("w", encoding="utf-8", newline="") as incorrect_handle:
        metric_writer = csv.DictWriter(metric_handle, fieldnames=METRIC_FIELDS)
        metric_writer.writeheader()
        writers = (
            csv.DictWriter(prediction_handle, fieldnames=PREDICTION_FIELDS),
            csv.DictWriter(correct_handle, fieldnames=PREDICTION_FIELDS),
            csv.DictWriter(incorrect_handle, fieldnames=PREDICTION_FIELDS),
        )
        for writer in writers:
            writer.writeheader()

        for percent, percent_dir in percent_folders:
            print(f"\n== {percent:g}% training data: {percent_dir.relative_to(PROJECT_ROOT)} ==", flush=True)
            checkpoints = checkpoint_paths(percent_dir)
            if not checkpoints:
                raise FileNotFoundError(f"No saved neural checkpoints found under {percent_dir}")

            # Load one checkpoint to obtain the run vocabulary and preprocessing;
            # later neural models are loaded and released one at a time.
            first_key = cli.load_neural(checkpoints[0])
            first_loaded = cli.neural[first_key]
            first = {key: value for key, value in first_loaded.items() if key != "model"}
            vocab = Vocab.from_list(first["vocab"])
            del cli.neural[first_key], first_loaded
            if cli.device.type == "cuda":
                torch.cuda.empty_cache()
            train_path = first["train_path"]
            if not train_path.is_file():
                raise FileNotFoundError(f"Exact training snapshot for this run is missing: {train_path}")
            print(f"Train input: {train_path.relative_to(PROJECT_ROOT) if train_path.is_relative_to(PROJECT_ROOT) else train_path}")
            print(f"Vocabulary: {len(vocab):,}; tokenizer={first['tokenizer']}; checkpoints={len(checkpoints)}")

            ngram_models, _ = make_ngram_variants(
                train_path, vocab, first["lowercase"], first["tokenizer"],
                first.get("train_file_percent", 100),
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
                    row = evaluate_split(corpus, percent, label, info, model, vocab,
                                         path, source, cli, writers)
                    metrics.append(row)
                    metric_writer.writerow(row)
                    metric_handle.flush()
                rare_metric_rows = []
                for order, rare_path in rare_paths.items():
                    if info["order"] != order:
                        continue
                    rare_metric_rows.append(evaluate_rare_file(
                        corpus, percent, label, info, model, vocab, rare_path,
                        order, cli, writers,
                    ))
                for rare_row in rare_metric_rows:
                    metrics.append(rare_row)
                    metric_writer.writerow(rare_row)
                    metric_handle.flush()

            # Neural model memory can be substantial for large vocabularies;
            # load, evaluate, and release one checkpoint before the next.
            for checkpoint in checkpoints:
                key = cli.load_neural(checkpoint)
                loaded = cli.neural[key]
                label = model_label(checkpoint, loaded["name"])
                seed_match = re.search(r"_seed(-?\d+)$", checkpoint.stem)
                info = dict(loaded)
                info.update({
                    "family": "neural",
                    "order": (
                        2 if label.startswith("ffnn_ctx1") else
                        3 if label.startswith("ffnn_ctx2") else ""
                    ),
                    "seed": seed_match.group(1) if seed_match else "",
                    "train_seconds": "",
                })
                if tuple(info["vocab"]) != tuple(vocab.itos):
                    raise ValueError(f"{checkpoint} uses a different vocabulary from its percentage run")
                if (info["tokenizer"] != first["tokenizer"]
                        or info["lowercase"] != first["lowercase"]
                        or info["train_path"].resolve() != train_path.resolve()):
                    raise ValueError(f"{checkpoint} uses different preprocessing or training data")
                model = info["model"]
                for source, path in (("train", train_path), ("valid", valid_path), ("test", test_path)):
                    row = evaluate_split(corpus, percent, label, info, model, vocab,
                                         path, source, cli, writers)
                    metrics.append(row)
                    metric_writer.writerow(row)
                    metric_handle.flush()
                rare_metric_rows = []
                for order, rare_path in rare_paths.items():
                    if label.startswith("ffnn_ctx1") and order != 2:
                        continue
                    if label.startswith("ffnn_ctx2") and order != 3:
                        continue
                    rare_metric_rows.append(evaluate_rare_file(
                        corpus, percent, label, info, model, vocab, rare_path,
                        order, cli, writers,
                    ))
                for rare_row in rare_metric_rows:
                    metrics.append(rare_row)
                    metric_writer.writerow(rare_row)
                    metric_handle.flush()
                del model, loaded, info, cli.neural[key]
                if cli.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif cli.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

    summary_rows = make_perplexity_summary(metrics)
    if summary_rows:
        with output_files["summary"].open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    plot_metrics(metrics, output_files["chart"], corpus)
    percentage_plots = plot_percentage_metrics(metrics, report_dir, corpus)
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
        "--check", action="store_true",
        help="read existing metrics.csv and report complete versus pending evaluations without inference or writes",
    )
    args = parser.parse_args(argv)
    if args.check:
        if args.device:
            parser.error("--device does not apply with --check")
        check_comparison(args.corpus)
        return 0
    return run(args.corpus, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
