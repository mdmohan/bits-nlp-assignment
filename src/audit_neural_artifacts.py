"""Read-only audit of frozen Gutenberg/Inaugural neural run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml


CORPORA = ("gutenberg", "inaugural")
PERCENTAGES = (20, 40, 60, 80, 100)
HISTORY_FIELDS = (
    "epoch", "train_ppl", "valid_ppl", "seconds", "train_nll", "valid_nll",
    "train_tokens", "valid_tokens", "train_word_tokens", "train_seconds", "valid_seconds",
)
PAYLOAD_FIELDS = (
    "model", "model_params", "variant", "seed", "train_config", "batch_size",
    "train_percent", "tokenizer", "lowercase", "corpus", "split_fingerprints",
    "vocab_fingerprint", "allowed_target_ids", "device", "torch_version", "code_fingerprints",
)
RESULT_FIELDS = (
    "model", "seed", "variant", "train_percent", "parameters", "vocab_size",
    "valid_ppl", "test_ppl", "epochs_trained", "train_seconds", "test_seconds",
    "test_oov_rate", "test_tokens", "best_epoch", "best_valid_nll", "stop_reason",
    "last_epoch", "best_valid_ppl", "last_valid_nll", "last_valid_ppl",
    "plateau_reached", "budget_limited", "convergence_status", "checkpoint_selection",
    "training_target_events_processed", "training_word_tokens_processed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expanded_model_specs(config):
    models = []
    for spec in config["models"]:
        contexts = config.get("context_sizes") if spec["name"] == "ffnn" else None
        if contexts:
            for context in contexts:
                models.append({**spec, "params": {**spec.get("params", {}), "context_size": context}})
        else:
            models.append({**spec, "params": dict(spec.get("params", {}))})
    return models


class ArtifactAudit:
    def __init__(self, workspace: Path, drive_root: Path, output: Path):
        self.workspace = workspace
        self.drive_root = drive_root
        self.output = output
        self.records = []
        self.conditions = []
        self.current = {}
        self.expected_stats = {}
        self.expected_vocab = None
        self.expected_config = None
        self.expected_payload = {}
        self.expected_rows = set()

    @staticmethod
    def require_fields(value, expected, record):
        for name, required in expected.items():
            if name not in value:
                record["issues"].append(f"Missing field: {name}")
            elif isinstance(required, dict) and isinstance(value[name], dict):
                ArtifactAudit.require_fields(value[name], required, record)
            elif value[name] != required:
                record["issues"].append(f"Field differs from frozen experiment: {name}")

    def config(self, path, record):
        value = self.mapping(path, record)
        for field in ("name", "data", "models", "seeds", "training", "vocab", "target_support"):
            if field not in value:
                record["issues"].append(f"Missing config field: {field}")
        if self.expected_config:
            expected = self.expected_config
            self.require_fields(value, {name: expected[name] for name in ("name", "models", "seeds", "training", "vocab", "target_support")}, record)
            from lm.tokenization import canonical_tokenizer_name
            expected_data = {name: expected["data"][name] for name in ("corpus", "split_seed", "split_ratios")}
            expected_data["tokenizer"] = canonical_tokenizer_name(expected["data"].get("tokenizer", "space"))
            self.require_fields(value.get("data", {}), expected_data, record)
            if value.get("data", {}).get("lowercase", True) != expected["data"].get("lowercase", True):
                record["issues"].append("Field differs from frozen experiment: data.lowercase")
            self.require_fields(value, {"train_percent": self.current["percent"], "model_variant": self.current["variant"]}
                                if self.current["variant"] else {"train_percent": self.current["percent"]}, record)
            if expected.get("target_support") == "vocab":
                from lm.data import Vocab
                self.require_fields(value, {"allowed_target_ids": Vocab.from_list(self.expected_vocab).allowed_target_ids}, record)
        return value

    def stats(self, path, record):
        value = self.mapping(path, record)
        self.require_fields(value, self.expected_stats, record)
        return value

    def exact_vocabulary(self, path, record):
        value = self.vocabulary(path, record)
        if self.expected_vocab is not None and value != self.expected_vocab:
            record["issues"].append("Vocabulary entries/order differ from the exact frozen training prefix")
        return value

    def snapshot(self, expected_sha, count):
        def check(path, record):
            if record["sha256"] != expected_sha:
                record["issues"].append(f"Training snapshot is not the exact frozen prefix ({count} sentences)")
        return check

    def inspect(self, path: Path, stage: str, kind: str, checker=None, required=True):
        record = {**self.current, "path": str(path), "stage": stage,
                  "kind": kind, "required": required, "status": "ok", "issues": []}
        self.records.append(record)
        if not path.is_file():
            record["status"] = "missing" if required else "not_retained"
            if required:
                record["issues"].append("Required file absent")
            return None
        value = None
        try:
            record["bytes"] = path.stat().st_size
            if not record["bytes"]:
                raise ValueError("File is empty")
            record["sha256"] = sha256(path)
            if checker:
                value = checker(path, record)
            if record["issues"]:
                record["status"] = "invalid"
        except Exception as error:
            record["status"] = "invalid"
            record["issues"].append(f"{type(error).__name__}: {error}")
        return value

    @staticmethod
    def mapping(path, record):
        value = yaml.safe_load(path.read_text()) if path.suffix == ".yaml" else json.loads(path.read_text())
        if not isinstance(value, dict) or not value:
            raise ValueError("Expected a nonempty mapping")
        return value

    @staticmethod
    def vocabulary(path, record):
        value = json.loads(path.read_text())
        if not isinstance(value, list) or not value or any(not isinstance(word, str) for word in value):
            raise ValueError("Expected a nonempty vocabulary of strings")
        if len(value) != len(set(value)):
            record["issues"].append("Vocabulary contains duplicate entries")
        from lm.data import PAD, UNK, BOS, EOS
        for token in (PAD, UNK, BOS, EOS):
            if token not in value:
                record["issues"].append(f"Missing special token: {token}")
        return value

    @staticmethod
    def table(path, record):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if not reader.fieldnames or not rows:
                raise ValueError("CSV has no header or data rows")
            for index, row in enumerate(rows, 1):
                if None in row or any(value is None for value in row.values()):
                    record["issues"].append(f"Malformed/truncated CSV row {index}")
        return rows

    @classmethod
    def history(cls, path, record):
        rows = cls.table(path, record)
        missing = set(HISTORY_FIELDS) - set(rows[0])
        if missing:
            record["issues"].append(f"Missing history columns: {sorted(missing)}")
        for index, row in enumerate(rows, 1):
            for field in HISTORY_FIELDS:
                if field not in row:
                    continue
                try:
                    value = float(row[field])
                    if not math.isfinite(value):
                        raise ValueError("nonfinite value")
                    if field == "epoch" and value != index:
                        raise ValueError("epoch sequence is incomplete")
                except (TypeError, ValueError) as error:
                    record["issues"].append(f"History row {index}, {field}: {error}")
        return rows

    @classmethod
    def signed_result(cls, path, record):
        value = cls.mapping(path, record)
        for field in ("signature", "signature_payload", "result"):
            if field not in value:
                record["issues"].append(f"Missing result field: {field}")
        payload = value.get("signature_payload", {})
        result = value.get("result", {})
        if not isinstance(payload, dict) or not isinstance(result, dict):
            raise ValueError("Result/payload must be mappings")
        for field in PAYLOAD_FIELDS:
            if field not in payload:
                record["issues"].append(f"Missing signature payload field: {field}")
        for field in RESULT_FIELDS:
            if field not in result:
                record["issues"].append(f"Missing fitted-result field: {field}")
        observed = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if value.get("signature") != observed:
            record["issues"].append("Signature does not match the complete payload")
        return value

    def result(self, path, record):
        value = self.signed_result(path, record)
        self.require_fields(value.get("signature_payload", {}), self.expected_payload, record)
        fitted = value.get("result", {})
        self.require_fields(fitted, {"variant": self.current["variant"], "seed": self.current["seed"],
                            "train_percent": self.current["percent"], "vocab_size": len(self.expected_vocab)}, record)
        for name in ("valid_ppl", "test_ppl", "best_valid_nll", "train_seconds", "test_seconds"):
            if name in fitted and (not isinstance(fitted[name], (float, int)) or not math.isfinite(fitted[name])):
                record["issues"].append(f"Invalid fitted metric: {name}")
        return value

    def runs_table(self, path, record):
        rows = self.table(path, record)
        observed = [(row.get("variant"), row.get("seed")) for row in rows]
        for variant, seed in sorted(self.expected_rows - set(observed)):
            record["issues"].append(f"Missing runs.csv row: {variant}, seed {seed}")
        if len(observed) != len(set(observed)):
            record["issues"].append("Duplicate model/seed rows in runs.csv")
        if set(observed) - self.expected_rows:
            record["issues"].append("Unexpected model/seed rows in runs.csv")
        for index, row in enumerate(rows, 1):
            for name in RESULT_FIELDS:
                if name not in row or row[name] is None or row[name] == "":
                    record["issues"].append(f"Missing runs.csv value: row {index}, {name}")
        return rows

    def summary_table(self, path, record):
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        if len(rows) < 4:
            raise ValueError("Summary CSV lacks its multi-level header or model rows")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            record["issues"].append("Truncated summary CSV row")
            return rows
        expected = {(metric, statistic) for metric in ("valid_ppl", "test_ppl", "train_seconds", "test_seconds")
                    for statistic in ("mean", "std")}
        observed = set(zip(rows[0], rows[1]))
        if expected - observed:
            record["issues"].append(f"Missing summary fields: {sorted(expected - observed)}")
        variants = {row[0] for row in rows[3:]}
        missing = {variant for variant, _ in self.expected_rows} - variants
        if missing:
            record["issues"].append(f"Missing summary model rows: {sorted(missing)}")
        return rows

    def resume(self, path, record):
        import numpy as np
        import torch

        allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype("uint32"))]
        with torch.serialization.safe_globals(allowed):
            state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Resume checkpoint is not a mapping")
        for name in ("format_version", "metadata", "model_state", "optimizer_state", "history", "best_nll",
                     "best_epoch", "epochs_without_improvement", "completed_epoch", "stopped_early", "best_checkpoint_file", "rng_state"):
            if name not in state:
                record["issues"].append(f"Missing resume field: {name}")
        if state.get("format_version") != 2:
            record["issues"].append("Unsupported resume format")
        for name in ("python", "numpy", "torch_cpu", "torch_cuda"):
            if name not in state.get("rng_state", {}):
                record["issues"].append(f"Missing RNG state for GPU resume: {name}")
        reference = state.get("best_checkpoint_file")
        if reference:
            self.inspect(Path(reference), "resume", "referenced_best_weights", self.weights(self.variant_config, self.variant_vocab))
            if not Path(reference).is_file():
                record["issues"].append(f"Referenced best checkpoint absent: {reference}")
        elif state.get("best_epoch") is not None:
            record["issues"].append("Best epoch exists but best checkpoint reference is absent")
        record["completed_epoch"] = state.get("completed_epoch")
        record["best_checkpoint_file"] = reference
        return state

    def weights(self, config, vocab):
        def check(path, record):
            import torch
            from lm.neural import build_model

            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not state:
                raise ValueError("Expected a nonempty weights-only state dictionary")
            for name, tensor in state.items():
                if not isinstance(tensor, torch.Tensor):
                    record["issues"].append(f"Non-tensor state entry: {name}")
                elif not torch.isfinite(tensor).all().item():
                    record["issues"].append(f"Nonfinite tensor: {name}")
            if config is None or vocab is None:
                record["issues"].append("Cannot validate tensor keys/shapes: config or vocabulary absent/invalid")
                return
            spec = config["models"][0]
            model = build_model(spec["name"], len(vocab), vocab.index("<pad>"), **spec.get("params", {}))
            model.load_state_dict(state, strict=True)
            record["tensor_shapes"] = {name: list(tensor.shape) for name, tensor in state.items()}
        return check

    def run(self):
        sys.path.insert(0, str(self.workspace / "src"))
        from run import expected_variants, verify_prepared_inputs
        from lm.data import Vocab, load_corpus_files
        from lm.neural.trainer import TrainConfig
        from lm.tokenization import canonical_tokenizer_name, get_tokenizer

        self.output.mkdir(parents=True, exist_ok=False)
        for corpus in CORPORA:
            self.current = {"corpus": corpus, "percent": "", "variant": "", "seed": ""}
            config_path = self.workspace / "snapshot/configs" / f"{corpus}.yaml"
            config = self.inspect(config_path, "prepared_input", "frozen_config", self.mapping)
            if config is None:
                raise ValueError(f"Cannot enumerate expected artifacts without {config_path}")
            corpus_root = self.workspace / "snapshot/processed" / corpus
            for name in ("train.txt", "valid.txt", "test.txt"):
                self.inspect(corpus_root / name, "prepared_input", "frozen_split")
            input_config = {**config, "data": {**config["data"], "output": str(self.workspace / "snapshot/processed"),
                            "prepared_manifest": str(corpus_root / "prepared_manifest.json")}}

            def prepared_manifest(path, record):
                value = self.mapping(path, record)
                verify_prepared_inputs(input_config)
                return value

            self.inspect(corpus_root / "prepared_manifest.json", "prepared_input", "prepared_manifest", prepared_manifest)
            run_root = self.drive_root / "full/runs" / config["name"]
            tokenizer_name = canonical_tokenizer_name(config["data"].get("tokenizer", "space"))
            tokenizer = get_tokenizer(tokenizer_name)
            split_hashes = {name: sha256(corpus_root / f"{name}.txt") for name in ("train", "valid", "test")}
            train_options = {name: value for name, value in config["training"].items() if name != "batch_size"}
            training_config = TrainConfig(**train_options).__dict__
            self.expected_rows = {(variant, str(seed)) for variant, _, _ in expected_variants(config) for seed in config["seeds"]}
            for percent in PERCENTAGES:
                self.current = {"corpus": corpus, "percent": percent, "variant": "", "seed": ""}
                print(f"AUDIT {corpus} {percent}%: files, metadata and CPU weight validation", flush=True)
                data = load_corpus_files(corpus_root / "train.txt", corpus_root / "valid.txt", corpus_root / "test.txt",
                                         train_percent=percent, tokenizer_name=tokenizer_name)
                if config["data"].get("lowercase", True):
                    data = data.lowercase()
                vocab_options = config.get("vocab", {})
                expected_vocab = Vocab(data.train, min_freq=vocab_options.get("min_freq", 2), max_size=vocab_options.get("max_size"))
                self.expected_vocab = expected_vocab.itos
                self.expected_stats = {**data.describe(), "tokenizer": tokenizer_name, "vocab_size": len(expected_vocab),
                                       "valid_oov_rate": expected_vocab.oov_rate(data.valid), "test_oov_rate": expected_vocab.oov_rate(data.test),
                                       "train_percent": percent}
                model_specs = expanded_model_specs(config)
                self.expected_config = {**config, "models": model_specs}
                digest = hashlib.sha256()
                copied = 0
                with (corpus_root / "train.txt").open("rb") as handle:
                    for raw_line in handle:
                        if not raw_line.strip() or not tokenizer.tokenize(raw_line.decode("utf-8")):
                            continue
                        digest.update(raw_line)
                        copied += 1
                        if copied == len(data.train):
                            break
                budget = run_root / f"train_{percent}pct"
                for name in ("train.txt", "config.yaml", "vocab.json", "data_stats.json", "runs.csv", "summary.csv"):
                    checker = self.snapshot(digest.hexdigest(), copied)
                    if name.endswith(".yaml"):
                        checker = self.config
                    elif name == "data_stats.json":
                        checker = self.stats
                    elif name == "vocab.json":
                        checker = self.exact_vocabulary
                    elif name == "runs.csv":
                        checker = self.runs_table
                    elif name == "summary.csv":
                        checker = self.summary_table
                    self.inspect(budget / name, "neural_training", "budget_artifact", checker)
                for variant, model_name, _ in expected_variants(config):
                    self.current["variant"] = variant
                    directory = budget / "models" / variant
                    spec = next(item for item in model_specs if item["name"] == model_name and
                                (model_name != "ffnn" or f"ffnn_ctx{item.get('params', {}).get('context_size', 3)}" == variant))
                    self.expected_config = {**config, "models": [spec]}
                    variant_config = self.inspect(directory / "config.yaml", "neural_training", "variant_config", self.config)
                    vocab = self.inspect(directory / "vocab.json", "neural_training", "variant_vocabulary", self.exact_vocabulary)
                    self.inspect(directory / "data_stats.json", "neural_training", "variant_data_stats", self.stats)
                    self.variant_config, self.variant_vocab = variant_config, vocab
                    for seed in config["seeds"]:
                        self.current["seed"] = seed
                        start = len(self.records)
                        stem = f"{model_name}_seed{seed}"
                        self.expected_payload = {"model": model_name, "model_params": spec.get("params", {}), "variant": variant,
                            "seed": seed, "train_config": training_config, "batch_size": config["training"]["batch_size"],
                            "train_percent": percent, "tokenizer": tokenizer_name, "lowercase": config["data"].get("lowercase", True),
                            "corpus": corpus, "split_fingerprints": split_hashes, "vocab_fingerprint": expected_vocab.fingerprint,
                            "allowed_target_ids": expected_vocab.allowed_target_ids if config.get("target_support") == "vocab" else None,
                            "code_fingerprints": {name: sha256(path) for name, path in {
                                "trainer": self.workspace / "src/lm/neural/trainer.py", "data": self.workspace / "src/lm/data.py",
                                "tokenization": self.workspace / "src/lm/tokenization.py",
                                "model": self.workspace / "src/lm/neural" / model_name / "model.py"}.items()}}
                        print(f"  AUDIT {variant} seed {seed}", flush=True)
                        self.inspect(directory / f"{stem}.pt", "neural_training", "final_weights", self.weights(variant_config, vocab))
                        history = self.inspect(directory / f"{stem}_history.csv", "neural_training", "history", self.history)
                        result = self.inspect(directory / f"{stem}_result.json", "neural_training", "signed_result", self.result)
                        if history is not None and result is not None and result.get("result", {}).get("epochs_trained") != len(history):
                            self.records[-1]["status"] = "invalid"
                            self.records[-1]["issues"].append("History length differs from epochs_trained")
                        own_records = self.records[start:]
                        valid = all(record["status"] == "ok" for record in own_records)
                        shared_valid = all(record["status"] == "ok" for record in self.records
                                           if record["corpus"] == corpus and record["percent"] == percent
                                           and record["stage"] == "neural_training"
                                           and record["kind"] in ("budget_artifact", "variant_config", "variant_vocabulary", "variant_data_stats")
                                           and record["variant"] in ("", variant))
                        self.conditions.append({**self.current, "complete": valid and shared_valid,
                                                "weights_usable": own_records[0]["status"] == "ok",
                                                "required_file_issues": [record["path"] for record in own_records if record["status"] != "ok"]})
                        resume_path = budget / "checkpoints" / f"{variant}_{stem}.resume.pt"
                        self.inspect(resume_path, "resume", "resume_checkpoint", self.resume, required=False)
                    self.current["seed"] = ""
                self.current["variant"] = ""
                self.inspect(budget / "count_models.pkl.gz", "comparison", "count_models")
                (self.output / "audit.json").write_text(json.dumps({"records": self.records, "conditions": self.conditions}, indent=2))
            self.current = {"corpus": corpus, "percent": "", "variant": "", "seed": ""}
            for name in ("metrics.csv", "perplexity_by_percentage.csv", "predictions.csv", "predictions_correct.csv",
                         "predictions_incorrect.csv", "charts.png", "scoring_events.jsonl.gz", "event_verification.json", "artifact_manifest.json"):
                self.inspect(run_root / "metrics" / name, "comparison", "comparison_output")
        report = {"schema_version": 1, "read_only": True, "neural_training": False,
                  "weight_validation": "CPU deserialization, finite tensors and strict model state keys/shapes; no trusted baseline weight hashes available",
                  "records": self.records, "conditions": self.conditions}
        (self.output / "audit.json").write_text(json.dumps(report, indent=2))
        columns = ("corpus", "percent", "variant", "seed", "stage", "kind", "required", "status", "path", "issues")
        with (self.output / "artifact_status.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                writer.writerow({**record, "issues": "; ".join(record["issues"])})
        summary = []
        for corpus in CORPORA:
            for percent in PERCENTAGES:
                records = [record for record in self.records if record["corpus"] == corpus and record["percent"] == percent]
                counts = Counter(record["status"] for record in records if record["stage"] == "neural_training")
                conditions = [condition for condition in self.conditions if condition["corpus"] == corpus and condition["percent"] == percent]
                summary.append({"corpus": corpus, "percent": percent, "complete_conditions": sum(item["complete"] for item in conditions),
                                "usable_weights": sum(item["weights_usable"] for item in conditions),
                                "expected_conditions": len(conditions), "missing_training_files": counts["missing"], "invalid_training_files": counts["invalid"]})
        report["summary"] = summary
        (self.output / "audit.json").write_text(json.dumps(report, indent=2))
        notes = ["# Neural Artifact Audit", "", "Read-only audit; no fitting or scoring. Full training runs only, not smoke.", "",
                 "| Corpus | Budget | Usable Weights | Complete Conditions | Missing Training Files | Invalid Training Files |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in summary:
            notes.append(f"| {row['corpus']} | {row['percent']}% | {row['usable_weights']}/9 | {row['complete_conditions']}/9 | {row['missing_training_files']} | {row['invalid_training_files']} |")
        for corpus in CORPORA:
            for percent in PERCENTAGES:
                notes.extend(["", f"## {corpus.title()} {percent}%", ""])
                for record in self.records:
                    if record["corpus"] == corpus and record["percent"] == percent and record["status"] != "ok":
                        label = "Optional checkpoint not retained" if record["status"] == "not_retained" else record["status"].upper()
                        relative = str(Path(record["path"]).relative_to(self.drive_root))
                        notes.append(f"- [{record['stage']}; {label}] `{relative}`: {'; '.join(record['issues'])}")
        notes.extend(["", "## Interpretation", "", "Missing comparison outputs mean scoring is not yet archived; they are not evidence of failed neural fitting.",
                      "Completed runs delete optional resume checkpoints. A retained resume file must include its referenced best weights.",
                      "Do not use signed result JSON alone to skip training when final weights are missing.",
                      "Rerun the A100 experiments into a new output root, using the unchanged frozen splits, budgets, settings and seeds.",
                      "Recorded weight hashes have no trusted pre-loss baseline; CPU loading/keys/shapes/finite checks cannot prove original numerical identity."])
        (self.output / "MISSING_ARTIFACTS.md").write_text("\n".join(notes) + "\n")
        print(json.dumps({"output": str(self.output), "summary": summary}, indent=2), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ArtifactAudit(args.workspace, args.drive_root, args.output).run()


if __name__ == "__main__":
    main()