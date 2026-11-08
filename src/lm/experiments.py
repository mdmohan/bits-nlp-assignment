"""Prepared-document sweeps using the existing count models and neural trainer."""

import contextlib
import importlib.metadata
import json
import math
import pickle
import platform
import re
import resource
import shutil
import sqlite3
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from .cli import CountNGram
from .data import Vocab, make_loader
from .evaluation import evaluate_model, original_counts
from .neural import MODELS, build_model, count_parameters
from .neural.trainer import Trainer, TrainConfig, get_device, set_seed
from .preparation import PreparationError, atomic_json, file_sha256, fingerprint, prepare_corpus, repair_variant_overlap, sentences_for, source_hashes, verify_prepared_manifest


def _now():
    return datetime.now(timezone.utc).isoformat()


def _finite_metrics(values):
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in values.items()}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _save_weights(model, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class ExperimentSuite:
    def __init__(self, config_path, corpora=None):
        self.root = Path(__file__).resolve().parents[2]
        self.config_path = Path(config_path).resolve()
        self.config = yaml.safe_load(self.config_path.read_text())
        self._validate()
        self.corpora = list(self.config["corpora"])
        if corpora:
            unknown = set(corpora) - set(self.corpora)
            if unknown:
                raise ValueError(f"Unknown corpora: {sorted(unknown)}")
            self.corpora = list(corpora)
        self.output = self._resolve(self.config["output_root"])
        self.device = get_device(self.config.get("device", "auto"))
        self.limits = {"max_training_events": 2_000_000, "max_vocab_size": 100_000,
                       "max_parameters": 50_000_000, "max_logits_elements": 8_000_000,
                       "max_padded_tokens": 1024, "max_sentence_tokens": 2048}
        self.limits.update(self.config.get("resources", {}))
        source_files = sorted((self.root / "src" / "lm").rglob("*.py"))
        source_files.extend([self.root / "src" / "run.py", self.root / "src" / "requirements.txt"])
        self.code_hashes = {str(path.relative_to(self.root)): file_sha256(path) for path in source_files}
        self.identity = fingerprint({"config": self.config, "code": self.code_hashes, "protocol": 1})
        self.environment = {"python": sys.version, "platform": platform.platform(), "device": str(self.device),
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "packages": dict(sorted((distribution.metadata["Name"], distribution.version)
                                                     for distribution in importlib.metadata.distributions()
                                                     if distribution.metadata["Name"]))}
        if self.device.type == "cuda":
            self.environment["gpu"] = torch.cuda.get_device_name(self.device)
        self.environment_id = fingerprint(self.environment)

    def _resolve(self, value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _validate(self):
        config = self.config
        if not isinstance(config, dict) or config.get("schema_version") != 1:
            raise ValueError("Expected suite schema_version: 1")
        if config.get("prepared_data", {}).get("rebuild") is not False:
            raise ValueError("prepared_data.rebuild must explicitly be false")
        if config["prepared_data"].get("sentence_quarantine") not in (None, "test_then_validation_v1"):
            raise ValueError("Unknown opt-in sentence quarantine policy")
        if config.get("vocab", {}).get("mode") != "per_budget" or config["vocab"].get("max_size") is not None:
            raise ValueError("This protocol requires uncapped per-budget training-only vocabularies")
        if not isinstance(config["vocab"].get("min_freq", 2), int) or config["vocab"].get("min_freq", 2) < 1:
            raise ValueError("vocab.min_freq must be a positive integer")
        budgets = config.get("budgets_percent", [])
        if not budgets or sorted(set(budgets)) != budgets or any(value <= 0 or value > 100 for value in budgets):
            raise ValueError("Provide increasing unique budget percentages in (0,100]")
        if not config.get("corpora") or not config.get("models"):
            raise ValueError("Configure at least one corpus and one model")
        names = list(config["corpora"]) + [spec["id"] for spec in config["models"]]
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
            raise ValueError("Corpus and model names must be safe path components")
        if len({spec["id"] for spec in config["models"]}) != len(config["models"]):
            raise ValueError("Model IDs must be unique")
        preprocessing = config.get("preprocessing", {})
        if preprocessing.get("tokenizer", "whitespace_v1") != "whitespace_v1":
            raise ValueError("Only prepared-line whitespace_v1 tokenization is supported")
        training = dict(config.get("training", {}))
        batch_size = training.pop("batch_size", 32)
        settings = TrainConfig(**training)
        if settings.epochs < 1 or batch_size < 1 or settings.patience < 1:
            raise ValueError("Epochs, batch size and patience must be positive")
        for spec in config["models"]:
            if spec["family"] == "neural":
                if spec.get("model", spec["id"]) not in MODELS:
                    raise ValueError(f"Unknown neural architecture: {spec['id']}")
                seeds = spec.get("final_seeds", [13, 42, 2026])
                if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(seed, int) for seed in seeds):
                    raise ValueError("Final seeds must be unique integers")
            elif spec["family"] == "ngram":
                if spec.get("order") not in (2, 3) or spec.get("smoothing") not in {"mle", "laplace"}:
                    raise ValueError("Suite count baselines support bigram/trigram MLE and add-k")
            else:
                raise ValueError(f"Unknown model family: {spec['family']}")
            key = "k"
            if spec["family"] == "neural":
                key = "lr"
            grid = spec.get("tuning", {}).get(key, [spec.get(key, 0.001)])
            if not grid or any(not math.isfinite(value) or value <= 0 for value in grid):
                raise ValueError("Tuning values must be finite and positive")

    def _entry(self, corpus):
        entry = dict(self.config["corpora"][corpus])
        for key in ("database", "train", "test"):
            if entry.get(key):
                entry[key] = str(self._resolve(entry[key]))
            import_manifest = self.config.get("prepared_data", {}).get("import_manifest")
            if import_manifest:
                manifest = json.loads(self._resolve(import_manifest).read_text())
                expected = {item["destination"]: item["sha256"] for item in manifest["files"]}
                expected.update({item["path"]: item["sha256"] for item in manifest.get("derived_databases", [])})
                resolved = {str(self._resolve(path)): value for path, value in expected.items()}
                entry["expected_hashes"] = {key: resolved[value] for key, value in entry.items()
                            if key in {"database", "train", "test"} and value in resolved}
        if entry.get("repair_policy") == "preserve_test_drop_train_overlap":
            return repair_variant_overlap(entry, self.output / "curated" / corpus)
        return entry

    def _initialize(self):
        path = self.output / "protocol.json"
        if path.exists() and json.loads(path.read_text())["fingerprint"] != self.identity:
            raise ValueError("Configuration or implementation changed; use a new output root")
        atomic_json(path, {"fingerprint": self.identity, "config": self.config, "source_hashes": self.code_hashes})
        environment_path = self.output / "environments" / (self.environment_id + ".json")
        atomic_json(environment_path, self.environment)

    def _event(self, **values):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": _now(), **values}, allow_nan=False) + "\n")

    def _prepared(self, corpus):
        destination = self.output / "prepared" / corpus
        if self.config.get("prepared_data", {}).get("packaged_only", False):
            path = destination / "manifest.json"
            index_directory = destination
            if self.config["prepared_data"].get("index_root"):
                index_directory = self._resolve(self.config["prepared_data"]["index_root"]) / corpus
                source_manifest = index_directory / "manifest.json"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_manifest, path)
                if file_sha256(path) != file_sha256(source_manifest):
                    raise PreparationError("Prepared partition manifest differs from its verified input bundle")
            manifest = json.loads(path.read_text())
            manifest["index"] = str(index_directory / "sentences.sqlite")
            if manifest.get("schema_version") == 2:
                verify_prepared_manifest(manifest)
            elif file_sha256(manifest["index"]) != manifest["index_sha256"]:
                raise PreparationError("Packaged sentence index checksum mismatch")
            return manifest
        return prepare_corpus(
            self._entry(corpus), destination, budgets=self.config["budgets_percent"],
            data_seed=self.config.get("data_seed", 42), budget_seed=self.config.get("budget_seed", 55),
            validation_fraction=self.config.get("validation_fraction_of_existing_train", 0.1),
            lowercase=self.config.get("preprocessing", {}).get("lowercase", True),
            max_sentence_tokens=self.limits["max_sentence_tokens"],
            near_duplicate_threshold=self.config.get("near_duplicate_threshold", 0.9),
            sentence_quarantine=self.config.get("prepared_data", {}).get("sentence_quarantine"),
        )

    def preflight(self, prepare=False):
        report = {}
        for corpus in self.corpora:
            try:
                if prepare or self.config.get("prepared_data", {}).get("packaged_only"):
                    manifest = self._prepared(corpus)
                    report[corpus] = {"status": "prepared", "fingerprint": manifest["fingerprint"]}
                    sentence_audit = manifest["duplicate_audit"].get("sentences")
                    if sentence_audit:
                        report[corpus].update(sentence_near_duplicate_status=sentence_audit["status"],
                                              sentence_near_duplicate_pairs=sentence_audit["pair_count"],
                                              sentence_near_duplicate_review_required=sentence_audit["review_required"])
                        print(f"{corpus}: sentence near-duplicate audit: {sentence_audit['status']} "
                              f"({sentence_audit['pair_count']} pairs; audit only, no removal)", flush=True)
                else:
                    hashes = source_hashes(self._entry(corpus))
                    report[corpus] = {"status": "ready_for_preparation", "sources": hashes}
            except (ValueError, OSError, sqlite3.Error, pickle.UnpicklingError) as error:
                report[corpus] = {"status": "blocked", "reason": str(error)}
            print(f"{corpus}: {report[corpus]['status']} {report[corpus].get('reason', '')}", flush=True)
        atomic_json(self.output / "corpus_status.json", report)
        return report

    def _data(self, corpus, manifest, percent):
        train = sentences_for(manifest, "train", percent)
        valid = sentences_for(manifest, "valid")
        test = sentences_for(manifest, "test")
        events = sum(length - 1 for length in train.lengths)
        if events > self.limits["max_training_events"]:
            raise PreparationError(f"Training events {events:,} exceed explicit resource limit; increase resources, do not truncate")
        counts = Counter(word for sentence in train for word in sentence)
        vocab = Vocab.from_counts(counts, self.config["vocab"].get("min_freq", 2))
        if len(vocab) > self.limits["max_vocab_size"]:
            raise PreparationError("Vocabulary exceeds resource limit; no vocabulary cap was applied")
        if any(spec["family"] == "neural" for spec in self.config["models"]):
            padded_limit = self._padded_limit(vocab)
            if max(max(train.lengths), max(valid.lengths), max(test.lengths)) - 1 > padded_limit:
                raise PreparationError("A complete train/held-out sentence exceeds the configured logits memory limit; no truncation was applied")
        directory = self.output / corpus / f"budget_{percent:03g}"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "vocab.json", vocab.itos)
        pd.DataFrame(counts.most_common(), columns=["word", "training_count"]).to_csv(directory / "vocab_counts.csv", index=False)
        allocation = next(budget for budget in manifest["budgets"] if budget["percent"] == percent)
        train_tokens = sum(counts.values())
        train_unk_tokens = sum(count for word, count in counts.items() if word not in vocab)
        valid_word_tokens = sum(length - 2 for length in valid.lengths)
        test_word_tokens = sum(length - 2 for length in test.lengths)
        valid_oov_tokens = sum(word not in vocab for sentence in valid for word in sentence)
        test_oov_tokens = sum(word not in vocab for sentence in test for word in sentence)
        stats = {"corpus": corpus, "budget_percent": percent, "documents": allocation["documents"],
                 "alias_of_percent": allocation.get("alias_of_percent"),
                 "document_count": len(allocation["documents"]), "train_sentences": len(train),
                 "train_tokens": train_tokens, "train_prediction_events": events,
                 "train_unk_tokens": train_unk_tokens, "train_unk_rate": train_unk_tokens / train_tokens,
                 "train_unk_target_events": train_unk_tokens, "train_unk_input_tokens": train_unk_tokens,
                 "train_unk_target_rate": train_unk_tokens / events,
                 "unk_exposure_policy": "record_only_training_min_freq_mapping",
                 "valid_sentences": len(valid), "test_sentences": len(test), "vocab_size": len(vocab),
                 "vocab_hash": vocab.fingerprint, "allowed_target_ids": vocab.allowed_target_ids,
                 "valid_word_tokens": valid_word_tokens, "test_word_tokens": test_word_tokens,
                 "valid_oov_tokens": valid_oov_tokens, "test_oov_tokens": test_oov_tokens,
                 "valid_oov_rate": valid_oov_tokens / valid_word_tokens,
                 "test_oov_rate": test_oov_tokens / test_word_tokens,
                 "prepared_fingerprint": manifest["fingerprint"], "subset_hash": fingerprint(allocation["documents"])}
        atomic_json(directory / "data_stats.json", stats)
        return directory, train, valid, test, vocab, stats

    def _padded_limit(self, vocab):
        return min(self.limits["max_padded_tokens"], self.limits["max_logits_elements"] // len(vocab))

    @staticmethod
    def verify_run(directory, expected=None):
        path = directory / "status.json"
        status = json.loads(path.read_text())
        if status["status"] != "complete":
            raise ValueError(f"Run is not complete: {directory}")
        if expected is not None and status["fingerprint"] != expected:
            raise ValueError(f"Run fingerprint differs: {directory}")
        for name, expected_hash in status["artifacts"].items():
            if file_sha256(directory / name) != expected_hash:
                raise ValueError(f"Run artifact checksum mismatch: {directory / name}")
        return status

    def _fit(self, directory, spec, seed, value, train, valid, vocab, stats, resume):
        job = {"suite": self.identity, "environment": self.environment_id, "model": spec,
               "seed": seed, "value": value, "subset": stats["subset_hash"], "vocab": vocab.fingerprint,
               "prepared": stats["prepared_fingerprint"], "phase": directory.parent.parent.name}
        job_id = fingerprint(job)
        status_path = directory / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("fingerprint") != job_id:
                raise ValueError(f"Refusing to mix different run fingerprints: {directory}")
            if status["status"] == "complete":
                if not resume:
                    raise ValueError(f"Completed run exists; pass --resume: {directory}")
                self.verify_run(directory, job_id)
                self._event(action="resume_verified", run=str(directory.relative_to(self.output)))
                return json.loads((directory / "result.json").read_text())
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(status_path, {"status": "running", "fingerprint": job_id, "started_at": _now()})
        self._event(action="fit_started", run=str(directory.relative_to(self.output)), fingerprint=job_id)
        attempt = directory / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        attempt.mkdir(parents=True)
        try:
            with (attempt / "run.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                    print(f"Run: {directory.relative_to(self.output)} | seed {seed} | value {value} | device {self.device}", flush=True)
                    set_seed(seed)
                    history = []
                    fit_summary = {}
                    checkpoint = None
                    if spec["family"] == "neural":
                        model_name = spec.get("model", spec["id"])
                        with torch.device("meta"):
                            parameters = count_parameters(build_model(model_name, len(vocab), vocab.pad_id, **spec.get("architecture", {})))
                        if parameters > self.limits["max_parameters"]:
                            raise PreparationError(f"Model has {parameters:,} parameters, above configured resource limit")
                        model = build_model(model_name, len(vocab), vocab.pad_id, **spec.get("architecture", {}))
                        training = dict(self.config.get("training", {}))
                        batch_size = training.pop("batch_size", 32)
                        training["lr"] = value
                        padded_limit = self._padded_limit(vocab)
                        if max(max(train.lengths), max(valid.lengths)) - 1 > padded_limit:
                            raise PreparationError("A complete sentence exceeds the configured logits memory limit; no sentence was truncated")
                        train_loader = make_loader(train, vocab, batch_size, True, padded_limit)
                        valid_loader = make_loader(valid, vocab, batch_size, False, padded_limit)
                        trainer = Trainer(model, TrainConfig(**training), vocab.pad_id, self.device, vocab.allowed_target_ids)
                        trainer.print_model_summary()
                        checkpoint = f"{model_name}_seed{seed}.pt"
                        recorded = []

                        def record_epoch(row, fitted_model):
                            row = _finite_metrics(row)
                            recorded.append(row)
                            pd.DataFrame(recorded).to_csv(attempt / "history.csv", index=False)
                            _save_weights(fitted_model, attempt / "latest.pt")
                            self._event(action="epoch_complete", run=str(directory.relative_to(self.output)), **row)

                        if self.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(self.device)
                        _sync(self.device)
                        started = time.perf_counter()
                        history = trainer.fit(train_loader, valid_loader, epoch_callback=record_epoch)
                        history = [_finite_metrics(row) for row in history]
                        fit_summary = _finite_metrics(trainer.fit_summary)
                        _sync(self.device)
                        train_seconds = time.perf_counter() - started
                        _save_weights(model, directory / checkpoint)
                        config = {"models": [{"name": model_name, "params": spec.get("architecture", {})}],
                                  "data": {"corpus": stats["corpus"], "lowercase": self.config.get("preprocessing", {}).get("lowercase", True)},
                                  "allowed_target_ids": vocab.allowed_target_ids, "suite_run": job,
                                  "budget_percent": stats["budget_percent"], "training": training}
                    else:
                        started = time.perf_counter()
                        model = CountNGram(spec["order"], vocab.itos, train, smoothing=spec["smoothing"],
                                           k=value, allowed_target_ids=vocab.allowed_target_ids)
                        train_seconds = time.perf_counter() - started
                        parameters = sum(len(model.model.ngram_counts[order]) for order in range(1, spec["order"] + 1))
                        checkpoint = "counts.pkl"
                        with (directory / checkpoint).open("wb") as stream:
                            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
                        config = {"model": spec, "k": value, "suite_run": job, "allowed_target_ids": vocab.allowed_target_ids}
                    _sync(self.device)
                    started = time.perf_counter()
                    validation = evaluate_model(model, vocab, valid, spec["family"], self.device,
                                                max_padded_tokens=self._padded_limit(vocab))
                    _sync(self.device)
                    validation_seconds = time.perf_counter() - started
                    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if sys.platform == "darwin":
                        rss /= 1024
                    result = {"fingerprint": job_id, "model": spec["id"], "family": spec["family"], "seed": seed,
                              **stats, "parameters": parameters, "selected_value": value, "epochs_trained": len(history),
                              "train_seconds": train_seconds, "valid_seconds": validation_seconds,
                              "valid_ppl": validation["perplexity"], "valid_nll": validation["nll"],
                              "valid_status": validation["status"], "valid_tokens": validation["tokens"],
                              "checkpoint": checkpoint, "environment": self.environment_id,
                              "process_peak_rss_mb": rss / 1024, "attempt": str(attempt.relative_to(directory))}
                    result.update(fit_summary)
                    if self.device.type == "cuda" and spec["family"] == "neural":
                        result["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
                    atomic_json(directory / "validation_metrics.json", validation)
                    atomic_json(directory / "result.json", result)
                    atomic_json(directory / "vocab.json", vocab.itos)
                    atomic_json(directory / "data_stats.json", stats)
                    (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
                    history_frame = pd.DataFrame(history) if history else pd.DataFrame(
                        columns=["epoch", "train_ppl", "valid_ppl", "seconds"])
                    history_frame.to_csv(directory / "history.csv", index=False)
            artifacts = {name: file_sha256(directory / name) for name in
                         (checkpoint, "result.json", "validation_metrics.json", "vocab.json", "data_stats.json", "config.yaml", "history.csv")}
            for name in ("run.log", "history.csv", "latest.pt"):
                path = attempt / name
                if path.exists():
                    artifacts[str(path.relative_to(directory))] = file_sha256(path)
            atomic_json(status_path, {"status": "complete", "fingerprint": job_id, "artifacts": artifacts, "completed_at": _now()})
            self._event(action="fit_complete", run=str(directory.relative_to(self.output)), fingerprint=job_id)
            return result
        except Exception as error:
            (attempt / "failure.txt").write_text(traceback.format_exc())
            atomic_json(status_path, {"status": "failed", "fingerprint": job_id, "reason": str(error), "failed_at": _now()})
            self._event(action="fit_failed", run=str(directory.relative_to(self.output)), reason=str(error))
            raise

    def _final_paths(self, budget_directory):
        for spec in self.config["models"]:
            seeds = [0]
            if spec["family"] == "neural":
                seeds = spec.get("final_seeds", [13, 42, 2026])
            for seed in seeds:
                yield spec, seed, budget_directory / "final" / spec["id"] / f"seed{seed}"

    def _train(self, resume):
        statuses = self.preflight(prepare=True)
        rows = []
        for corpus, status in statuses.items():
            if status["status"] == "blocked":
                continue
            manifest = self._prepared(corpus)
            for percent in self.config["budgets_percent"]:
                try:
                    directory, train, valid, _, vocab, stats = self._data(corpus, manifest, percent)
                    for spec in self.config["models"]:
                        key = "k"
                        default = spec.get("k", 1.0)
                        if spec["family"] == "neural":
                            key = "lr"
                            default = self.config.get("training", {}).get("lr", 0.001)
                        grid = spec.get("tuning", {}).get(key, [default])
                        candidates = []
                        if len(grid) > 1:
                            for candidate, value in enumerate(grid):
                                seed = spec.get("tuning", {}).get("seed", 7)
                                trial = directory / "tuning" / spec["id"] / f"candidate{candidate}_seed{seed}"
                                record = self._fit(trial, spec, seed, value, train, valid, vocab, stats, resume)
                                candidates.append({"value": value, "valid_nll": record["valid_nll"], "fingerprint": record["fingerprint"]})
                            if all(candidate["valid_nll"] is None for candidate in candidates):
                                raise PreparationError("No finite validation score for hyperparameter selection")
                            def objective(candidate):
                                if candidate["valid_nll"] is None:
                                    return math.inf
                                return candidate["valid_nll"]
                            value = min(candidates, key=objective)["value"]
                        else:
                            value = grid[0]
                        selection = {"parameter": key, "selected": value, "candidates": candidates, "test_used": False}
                        atomic_json(directory / (spec["id"] + "_selection.json"), selection)
                        for final_spec, seed, run_directory in self._final_paths(directory):
                            if final_spec["id"] == spec["id"]:
                                rows.append(self._fit(run_directory, spec, seed, value, train, valid, vocab, stats, resume))
                    atomic_json(directory / "budget_status.json", {"status": "complete"})
                    from .reporting import build_report
                    build_report(self.output)
                except PreparationError as error:
                    directory = self.output / corpus / f"budget_{percent:03g}"
                    atomic_json(directory / "budget_status.json", {"status": "blocked", "reason": str(error)})
                    self._event(action="budget_blocked", corpus=corpus, budget_percent=percent, reason=str(error))
        pd.DataFrame(rows).drop(columns=["documents", "allowed_target_ids"], errors="ignore").to_csv(self.output / "runs.csv", index=False)
        return rows

    def _seal_content(self):
        statuses = json.loads((self.output / "corpus_status.json").read_text())
        runs = {}
        prepared = {}
        excluded = {corpus: value for corpus, value in statuses.items() if value["status"] == "blocked"}
        for corpus in self.corpora:
            if corpus in excluded:
                continue
            prepared[corpus] = file_sha256(self.output / "prepared" / corpus / "manifest.json")
            for percent in self.config["budgets_percent"]:
                directory = self.output / corpus / f"budget_{percent:03g}"
                budget_status = json.loads((directory / "budget_status.json").read_text())
                if budget_status["status"] != "complete":
                    raise PreparationError(f"Cannot seal an incomplete budget: {corpus}/{percent}")
                for _, _, run_directory in self._final_paths(directory):
                    self.verify_run(run_directory)
                for status_path in sorted(directory.glob("**/status.json")):
                    status = self.verify_run(status_path.parent)
                    runs[str(status_path.parent.relative_to(self.output))] = status["artifacts"]
                for selection in sorted(directory.glob("*_selection.json")):
                    runs[str(selection.relative_to(self.output))] = file_sha256(selection)
                for name in ("vocab.json", "vocab_counts.csv", "data_stats.json"):
                    path = directory / name
                    runs[str(path.relative_to(self.output))] = file_sha256(path)
        if not runs:
            raise PreparationError("No eligible completed runs to seal")
        return {"suite": self.identity, "corpora": self.corpora, "runs": runs, "prepared_manifests": prepared, "excluded_corpora": excluded}

    def _seal(self):
        content = self._seal_content()
        payload = {"fingerprint": fingerprint(content), "content": content}
        atomic_json(self.output / "seal.json", payload)
        self._event(action="protocol_sealed", fingerprint=payload["fingerprint"])
        return payload

    def _evaluate(self, resume):
        seal = json.loads((self.output / "seal.json").read_text())
        if seal["fingerprint"] != fingerprint(self._seal_content()):
            raise ValueError("Sealed artifacts changed; refusing test evaluation")
        for corpus in self.corpora:
            if corpus in seal["content"]["excluded_corpora"]:
                continue
            manifest = self._prepared(corpus)
            smallest = sentences_for(manifest, "train", self.config["budgets_percent"][0])
            fixed = original_counts(smallest)
            for percent in self.config["budgets_percent"]:
                directory, train, _, test, vocab, _ = self._data(corpus, manifest, percent)
                reference = CountNGram(3, vocab.itos, train, allowed_target_ids=vocab.allowed_target_ids).model
                current = original_counts(train)
                for spec, seed, run_directory in self._final_paths(directory):
                    result = json.loads((run_directory / "result.json").read_text())
                    evaluation_id = fingerprint([result["fingerprint"], seal["fingerprint"]])
                    status_path = run_directory / "evaluation_status.json"
                    if status_path.exists() and resume:
                        status = json.loads(status_path.read_text())
                        if status["fingerprint"] != evaluation_id or file_sha256(run_directory / "test_metrics.json") != status["sha256"]:
                            raise ValueError("Test result checksum or sealed fingerprint mismatch")
                        continue
                    if spec["family"] == "neural":
                        model = build_model(spec.get("model", spec["id"]), len(vocab), vocab.pad_id, **spec.get("architecture", {}))
                        model.load_state_dict(torch.load(run_directory / result["checkpoint"], map_location="cpu", weights_only=True))
                        model.to(self.device)
                    else:
                        with (run_directory / result["checkpoint"]).open("rb") as stream:
                            model = pickle.load(stream)
                    _sync(self.device)
                    started = time.perf_counter()
                    metrics = evaluate_model(model, vocab, test, spec["family"], self.device,
                                             max_padded_tokens=self._padded_limit(vocab),
                                             mapped_reference=reference, current_original=current, fixed_original=fixed)
                    _sync(self.device)
                    metrics.update(run_fingerprint=result["fingerprint"], seal_fingerprint=seal["fingerprint"],
                                   test_seconds=time.perf_counter() - started)
                    atomic_json(run_directory / "test_metrics.json", metrics)
                    atomic_json(status_path, {"fingerprint": evaluation_id, "sha256": file_sha256(run_directory / "test_metrics.json")})
                    self._event(action="test_evaluated", corpus=corpus, budget_percent=percent, model=spec["id"], seed=seed,
                                status=metrics["status"], perplexity=metrics["perplexity"])

    def run(self, stage, resume=False, dry_run=False):
        if dry_run:
            stages = {"stage": stage, "corpora": self.corpora, "budgets": self.config["budgets_percent"],
                      "models": [spec["id"] for spec in self.config["models"]], "epochs": self.config.get("training", {}).get("epochs", 20),
                      "output": str(self.output), "training_started": False}
            print(json.dumps(stages, indent=2))
            return stages
        self._initialize()
        self._event(action="stage_started", stage=stage)
        if stage == "preflight":
            return self.preflight()
        if stage == "prepare":
            return self.preflight(prepare=True)
        if stage == "train":
            return self._train(resume)
        if stage == "seal":
            return self._seal()
        if stage == "evaluate":
            return self._evaluate(resume)
        if stage == "report":
            from .reporting import build_report
            return build_report(self.output)
        if stage == "all":
            self._train(resume)
            self._seal()
            self._evaluate(resume)
            from .reporting import build_report
            return build_report(self.output)
        raise ValueError(f"Unknown suite stage: {stage}")
