"""Experiment controller: plug in any Corpus and any registered model, get comparable results.

    controller = LanguageModelController(corpus, min_freq=2, batch_size=32)
    result = controller.run("lstm", {"hidden_dim": 256}, TrainConfig(epochs=10), seed=42)

All models run by one controller share the same vocabulary and the same train/valid/test
batches, so their perplexities are directly comparable.
"""

import json
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from .data import Corpus, Vocab, load_corpus_files, make_loader
from .neural import build_model, count_parameters
from .neural.trainer import Trainer, TrainConfig, get_device, set_seed
from .tokenization import canonical_tokenizer_name, get_tokenizer


class LanguageModelController:
    def __init__(self, corpus: Corpus, min_freq: int = 2, max_vocab: int | None = None,
                 batch_size: int = 32, device: str = "auto", output_dir: str | None = None,
                 tokenizer_name: str = "space", target_support: str | None = None):
        self.corpus = corpus
        self.tokenizer_name = canonical_tokenizer_name(tokenizer_name)
        self.vocab = Vocab(corpus.train, min_freq=min_freq, max_size=max_vocab)
        if target_support not in {None, "vocab"}:
            raise ValueError("target_support must be 'vocab' or omitted")
        self.allowed_target_ids = None
        if target_support == "vocab":
            self.allowed_target_ids = self.vocab.allowed_target_ids
        self.device = get_device(device)
        self.output_dir = None
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader = make_loader(corpus.train, self.vocab, batch_size, shuffle=True)
        self.valid_loader = make_loader(corpus.valid, self.vocab, batch_size, shuffle=False)
        self.test_loader = make_loader(corpus.test, self.vocab, batch_size, shuffle=False)

        self.data_stats = {
            **corpus.describe(),
            "tokenizer": self.tokenizer_name,
            "vocab_size": len(self.vocab),
            "valid_oov_rate": self.vocab.oov_rate(corpus.valid),
            "test_oov_rate": self.vocab.oov_rate(corpus.test),
        }
        print("data:", json.dumps(self.data_stats, indent=2))

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write_training_subset(
        source_path: Path,
        destination: Path,
        sentence_count: int,
        tokenizer_name: str,
    ) -> None:
        """Save the exact nonempty source lines consumed by the neural training run."""
        tokenizer = get_tokenizer(tokenizer_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        copied = 0
        try:
            with source_path.open("rb") as source, temporary.open("wb") as target:
                for raw_line in source:
                    line = raw_line.decode("utf-8")
                    if not line.strip() or not tokenizer.tokenize(line):
                        continue
                    target.write(raw_line)
                    copied += 1
                    if copied == sentence_count:
                        break
                target.flush()
                os.fsync(target.fileno())
            if copied != sentence_count:
                raise ValueError(
                    f"Training subset mismatch: expected {sentence_count:,} sentences, "
                    f"copied {copied:,} from {source_path}"
                )
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_json_atomic(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
            with temporary.open("r+b") as stream:
                import os
                os.fsync(stream.fileno())
            temporary.replace(path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def _recover_completed_result(
        self, model_name: str, model_params: dict, seed: int, final_model: Path,
        history_path: Path,
    ) -> dict:
        """Rebuild metrics from a final weights-only checkpoint without retraining."""
        model = build_model(model_name, len(self.vocab), self.vocab.pad_id, **model_params)
        model.load_state_dict(
            torch.load(final_model, map_location=self.device, weights_only=True)
        )
        trainer = Trainer(model, TrainConfig(), self.vocab.pad_id, self.device,
                  allowed_target_ids=self.allowed_target_ids)
        test_started = time.perf_counter()
        test = trainer.evaluate(self.test_loader, "resume test evaluation")
        test_seconds = time.perf_counter() - test_started
        valid = trainer.evaluate(self.valid_loader, "resume validation evaluation")
        history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
        train_seconds = float(history["seconds"].sum()) if "seconds" in history else 0.0
        epochs_trained = int(len(history))
        result = {
            "model": model_name,
            "seed": seed,
            "parameters": count_parameters(model),
            "vocab_size": len(self.vocab),
            "test_oov_rate": self.data_stats["test_oov_rate"],
            "valid_ppl": valid["perplexity"],
            "test_ppl": test["perplexity"],
            "test_tokens": test["tokens"],
            "epochs_trained": epochs_trained,
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
        }
        print(f"  Reused completed weights: {final_model.name}", flush=True)
        return result

    def run(
        self,
        model_name: str,
        model_params: dict,
        train_config: TrainConfig,
        seed: int,
        artifact_dir: Path | None = None,
        checkpoint_path: Path | None = None,
        best_checkpoint_path: Path | None = None,
        checkpoint_metadata: dict | None = None,
        resume: bool = False,
    ) -> dict:
        """Train one model with one seed and evaluate it on the test set."""
        print(f"\n=== {model_name} | seed {seed} | device {self.device} ===")
        set_seed(seed)
        model = build_model(model_name, len(self.vocab), self.vocab.pad_id, **model_params)
        trainer = Trainer(model, train_config, self.vocab.pad_id, self.device,
                  allowed_target_ids=self.allowed_target_ids)

        start = time.perf_counter()
        history = trainer.fit(
            self.train_loader,
            self.valid_loader,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            checkpoint_metadata=checkpoint_metadata,
            resume=resume,
        )
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        test = trainer.evaluate(self.test_loader, "test")
        test_seconds = time.perf_counter() - start
        valid = trainer.evaluate(self.valid_loader, "valid")

        result = {
            **trainer.fit_summary,
            "model": model_name,
            "seed": seed,
            "parameters": count_parameters(model),
            "vocab_size": len(self.vocab),
            "test_oov_rate": self.data_stats["test_oov_rate"],
            "valid_ppl": valid["perplexity"],
            "test_ppl": test["perplexity"],
            "test_tokens": test["tokens"],
            "epochs_trained": len(history),
            "train_seconds": train_seconds,
            "test_seconds": test_seconds,
        }
        print(f"  test ppl {test['perplexity']:.2f}")

        save_dir = Path(artifact_dir) if artifact_dir is not None else self.output_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            run_name = f"{model_name}_seed{seed}"
            pd.DataFrame(history).to_csv(save_dir / f"{run_name}_history.csv", index=False)
            torch.save(model.state_dict(), save_dir / f"{run_name}.pt")
        return result


def run_from_config(
    config_path: str,
    train_percent: float = 100,
    algorithm: str | None = None,
    seed_override: int | None = None,
    context_size_override: int | None = None,
    tokenizer_override: str | None = None,
    resume: bool = False,
    prepare_data: bool = True,
) -> pd.DataFrame:
    """Run configured models/seeds, with optional algorithm, seed and FFNN context overrides."""
    if not 0 < train_percent <= 100:
        raise ValueError("train_percent must be greater than 0 and at most 100")
    if context_size_override is not None and context_size_override < 1:
        raise ValueError("FFNN context size must be at least 1")

    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    configured_models = config.get("models", [])
    if not configured_models:
        raise ValueError("The config must define at least one model under 'models'.")
    if algorithm is None:
        selected_models = configured_models
    else:
        matches = [
            model for model in configured_models
            if str(model.get("name", "")).casefold() == algorithm.casefold()
        ]
        if not matches:
            available = ", ".join(str(model.get("name", "<unnamed>")) for model in configured_models)
            raise ValueError(f"Unknown algorithm '{algorithm}'. Configured algorithms: {available}")
        selected_models = matches
    configured_context_sizes = config.get("context_sizes")
    if configured_context_sizes is not None:
        if (
            not isinstance(configured_context_sizes, list)
            or not configured_context_sizes
            or any(type(value) is not int or value < 1 for value in configured_context_sizes)
        ):
            raise ValueError("context_sizes must be a non-empty list of positive integers.")
        if len(set(configured_context_sizes)) != len(configured_context_sizes):
            raise ValueError("context_sizes must not contain duplicates.")

    # Copy model specs before expanding context variants so the loaded config
    # remains an independent source object.
    configured_models = [
        {**model, "params": dict(model.get("params", {}))}
        for model in selected_models
    ]
    selected_models = []
    has_ffnn = any(str(model.get("name", "")).casefold() == "ffnn" for model in configured_models)
    if context_size_override is not None and not has_ffnn:
        raise ValueError("--context-size applies to FFNN; select/configure an FFNN model.")
    context_sizes = (
        [context_size_override]
        if context_size_override is not None
        else configured_context_sizes
    )
    for model in configured_models:
        if str(model.get("name", "")).casefold() == "ffnn" and context_sizes:
            for context_size in context_sizes:
                variant = {**model, "params": dict(model["params"])}
                variant["params"]["context_size"] = context_size
                selected_models.append(variant)
        else:
            selected_models.append(model)
    configured_seeds = config.get("seeds", [42])
    selected_seeds = [seed_override] if seed_override is not None else configured_seeds
    if not selected_seeds:
        raise ValueError("The config must define at least one seed, or provide --seed.")
    # Save the effective selection alongside results so the output config records
    # precisely which models and seeds produced this run.
    config["models"] = selected_models
    config["seeds"] = selected_seeds
    output_name = config["name"]
    output_dir = (
        Path(config.get("output_dir", "results"))
        / output_name
        / f"train_{train_percent:g}pct"
    )
    if context_size_override is not None:
        output_name = f"{output_name}_ffnn_ctx{context_size_override}"
        config["context_size_override"] = context_size_override
        config["context_sizes"] = [context_size_override]
        output_dir = output_dir / f"ffnn_ctx{context_size_override}"

    data_config = config["data"]
    if not isinstance(data_config, dict):
        raise ValueError("The config must define a 'data' mapping.")
    tokenizer_name = canonical_tokenizer_name(
        tokenizer_override if tokenizer_override is not None
        else data_config.get("tokenizer", "space")
    )
    data_config["tokenizer"] = tokenizer_name
    vocab_config = config.get("vocab", {})
    training = dict(config.get("training", {}))
    batch_size = training.pop("batch_size", 32)

    if "corpus" not in data_config:
        raise ValueError(
            "data.corpus is required; prepare data through data_orchestrator.py "
            "and load the generated train/valid/test files."
        )
    ratios = data_config.get("split_ratio", data_config.get("split_ratios"))
    if isinstance(ratios, dict):
        ratios = ratios.get("ratios")
    if ratios is None:
        ratios = data_config.get("params", {}).get("ratios", [0.8, 0.2])
    if len(ratios) != 3:
        raise ValueError("Neural model training requires train/valid/test; configure three split ratios.")

    project_root = Path(__file__).resolve().parents[2]
    pipeline = project_root / "src" / "data-pipeline" / "data_orchestrator.py"
    if prepare_data:
        try:
            subprocess.run(
                [sys.executable, str(pipeline), "--config", str(config_path), "--token", tokenizer_name],
                cwd=project_root,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Data preparation failed (exit code {exc.returncode}).") from exc
    else:
        print("Skipping data preparation; using the already prepared corpus splits.", flush=True)

    processed_root = Path(data_config.get("output", "data/processed")).expanduser()
    if not processed_root.is_absolute():
        processed_root = project_root / processed_root
    corpus_dir = processed_root / data_config["corpus"]
    if config.get("execution", {}).get("colab_only"):
        if not os.environ.get("COLAB_RELEASE_TAG") or not torch.cuda.is_available():
            raise RuntimeError("This experiment may only fit models in a Colab GPU runtime")
    corpus = load_corpus_files(
        corpus_dir / "train.txt",
        corpus_dir / "valid.txt",
        corpus_dir / "test.txt",
        train_percent=train_percent,
        tokenizer_name=tokenizer_name,
    )
    if data_config.get("lowercase", True):
        corpus = corpus.lowercase()
    controller = LanguageModelController(
        corpus,
        min_freq=vocab_config.get("min_freq", 2),
        max_vocab=vocab_config.get("max_size"),
        batch_size=batch_size,
        device=config.get("device", "auto"),
        output_dir=output_dir,
        tokenizer_name=tokenizer_name,
        target_support=config.get("target_support"),
    )
    if controller.allowed_target_ids is not None:
        config["allowed_target_ids"] = list(controller.allowed_target_ids)
    controller.data_stats["train_percent"] = train_percent
    config["train_percent"] = train_percent
    training_snapshot = output_dir / "train.txt"
    controller._write_training_subset(
        corpus_dir / "train.txt",
        training_snapshot,
        len(corpus.train),
        tokenizer_name,
    )
    print(
        f"Saved exact training subset: {training_snapshot} "
        f"({len(corpus.train):,} sentences; {train_percent:g}%)",
        flush=True,
    )
    train_config = TrainConfig(**training)
    split_paths = {
        "train": corpus_dir / "train.txt",
        "valid": corpus_dir / "valid.txt",
        "test": corpus_dir / "test.txt",
    }
    log_start = time.perf_counter()
    split_fingerprints = {
        name: controller._file_sha256(path) for name, path in split_paths.items()
    }
    print(
        "Checkpoint input fingerprints ready "
        f"({time.perf_counter() - log_start:.1f}s)", flush=True,
    )

    model_labels = []
    for model in selected_models:
        label = str(model.get("name"))
        if label.casefold() == "ffnn":
            context_size = model.get("params", {}).get("context_size", 3)
            label += f" (context_size={context_size})"
        model_labels.append(label)

    # Each configured model entry gets its own artifact directory. Duplicate
    # algorithm names (for example, FFNNs with contexts 1 and 2) therefore
    # cannot overwrite one another's checkpoint or training history.
    variant_counts: dict[str, int] = {}
    run_specs = []
    for model in selected_models:
        model_name = str(model.get("name", "model"))
        if model_name.casefold() == "ffnn":
            context_size = model.get("params", {}).get("context_size", 3)
            base_label = f"ffnn_ctx{context_size}"
        else:
            base_label = model_name
        variant_counts[base_label] = variant_counts.get(base_label, 0) + 1
        occurrence = variant_counts[base_label]
        variant_label = base_label if occurrence == 1 else f"{base_label}_{occurrence}"
        run_specs.append((model, variant_label))

    print(
        "Run selection: "
        + ", ".join(model_labels)
        + f" | seeds {', '.join(str(seed) for seed in selected_seeds)}",
        flush=True,
    )
    results = []
    for model, variant_label in run_specs:
        is_ffnn = str(model.get("name", "")).casefold() == "ffnn"
        context_size = model.get("params", {}).get("context_size", 3) if is_ffnn else None
        variant_dir = output_dir / "models" / variant_label
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_config = dict(config)
        variant_config["models"] = [model]
        variant_config["model_variant"] = variant_label
        if is_ffnn:
            variant_config["context_sizes"] = [context_size]
        (variant_dir / "config.yaml").write_text(yaml.safe_dump(variant_config, sort_keys=False))
        (variant_dir / "vocab.json").write_text(json.dumps(controller.vocab.itos))
        (variant_dir / "data_stats.json").write_text(json.dumps(controller.data_stats, indent=2))
        for seed in selected_seeds:
            model_name = str(model["name"])
            run_name = f"{model_name}_seed{seed}"
            final_model = variant_dir / f"{run_name}.pt"
            history_path = variant_dir / f"{run_name}_history.csv"
            result_path = variant_dir / f"{run_name}_result.json"
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_path = checkpoint_dir / f"{variant_label}_{run_name}.resume.pt"
            best_checkpoint_path = checkpoint_dir / f"{variant_label}_{run_name}.best.pt"

            selected_model_file = (
                Path(__file__).parent / "neural" / model_name.lower() / "model.py"
            )
            signature_payload = {
                "model": model_name,
                "model_params": model.get("params", {}),
                "variant": variant_label,
                "seed": seed,
                "train_config": train_config.__dict__,
                "batch_size": batch_size,
                "train_percent": train_percent,
                "tokenizer": tokenizer_name,
                "lowercase": bool(data_config.get("lowercase", True)),
                "corpus": data_config["corpus"],
                "split_fingerprints": split_fingerprints,
                "vocab_fingerprint": controller.vocab.fingerprint,
                "allowed_target_ids": controller.allowed_target_ids,
                "device": str(controller.device),
                "torch_version": torch.__version__,
                "code_fingerprints": {
                    "trainer": controller._file_sha256(Path(__file__).parent / "neural" / "trainer.py"),
                    "data": controller._file_sha256(Path(__file__).parent / "data.py"),
                    "tokenization": controller._file_sha256(Path(__file__).parent / "tokenization.py"),
                    "model": controller._file_sha256(selected_model_file),
                },
            }
            signature = hashlib.sha256(
                json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            checkpoint_metadata = {"signature": signature}

            reuse_saved_result = False
            if resume and result_path.is_file():
                saved_result = json.loads(result_path.read_text(encoding="utf-8"))
                if saved_result.get("signature") != signature:
                    previous_payload = saved_result.get("signature_payload")
                    if isinstance(previous_payload, dict):
                        differing_fields = [
                            key
                            for key in sorted(set(previous_payload) | set(signature_payload))
                            if previous_payload.get(key) != signature_payload.get(key)
                        ]
                    else:
                        differing_fields = ["signature_payload (missing from saved result)"]
                    runtime_only_differences = {"device", "torch_version"}
                    if differing_fields and set(differing_fields) <= runtime_only_differences:
                        previous_device = previous_payload.get("device", "unknown")
                        previous_torch = previous_payload.get("torch_version", "unknown")
                        print(
                            f"Reusing completed fit {variant_label}/{run_name}; "
                            "only runtime metadata differs "
                            f"(device {previous_device} -> {controller.device}, "
                            f"torch {previous_torch} -> {torch.__version__}).",
                            flush=True,
                        )
                    else:
                        difference_summary = ", ".join(differing_fields) or "unknown signature difference"
                        raise ValueError(
                            f"Completed result settings do not match this run: {result_path}. "
                            f"Differing signature fields: {difference_summary}. "
                            "Use matching model, data, and code settings, or start a fresh run without --resume."
                        )
                else:
                    print(f"Skipping completed fit {variant_label}/{run_name}", flush=True)
                result = saved_result["result"]
                reuse_saved_result = True
                checkpoint_path.unlink(missing_ok=True)
                best_checkpoint_path.unlink(missing_ok=True)
                for stale_best in best_checkpoint_path.parent.glob(
                    f"{best_checkpoint_path.stem}.epoch-*.pt"
                ):
                    stale_best.unlink(missing_ok=True)
            elif resume and final_model.is_file() and not checkpoint_path.is_file():
                if data_config.get("prepared_manifest"):
                    raise ValueError(f"Cannot verify completed weights without signed result: {result_path}")
                result = controller._recover_completed_result(
                    model_name, model.get("params", {}), seed, final_model, history_path,
                )
            else:
                if not resume:
                    checkpoint_path.unlink(missing_ok=True)
                    best_checkpoint_path.unlink(missing_ok=True)
                result = controller.run(
                    model_name, model.get("params", {}), train_config, seed,
                    artifact_dir=variant_dir,
                    checkpoint_path=checkpoint_path,
                    best_checkpoint_path=best_checkpoint_path,
                    checkpoint_metadata=checkpoint_metadata,
                    resume=resume,
                )
            if is_ffnn:
                result["context_size"] = context_size
            result["variant"] = variant_label
            result["train_percent"] = train_percent
            if not reuse_saved_result:
                controller._write_json_atomic(
                    result_path,
                    {"signature": signature, "signature_payload": signature_payload, "result": result},
                )
            checkpoint_path.unlink(missing_ok=True)
            best_checkpoint_path.unlink(missing_ok=True)
            for stale_best in best_checkpoint_path.parent.glob(
                f"{best_checkpoint_path.stem}.epoch-*.pt"
            ):
                stale_best.unlink(missing_ok=True)
            results.append(result)
    for result in results:
        result["train_percent"] = train_percent

    runs = pd.DataFrame(results)
    group_columns = ["variant", "model"]
    if "context_size" in runs.columns:
        group_columns.append("context_size")
    summary = runs.groupby(group_columns, dropna=False)[
        ["valid_ppl", "test_ppl", "train_seconds", "test_seconds"]
    ].agg(["mean", "std"])

    runs.to_csv(output_dir / "runs.csv", index=False)
    summary.to_csv(output_dir / "summary.csv")
    (output_dir / "data_stats.json").write_text(json.dumps(controller.data_stats, indent=2))
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir / "vocab.json").write_text(json.dumps(controller.vocab.itos))

    print("\n", summary.round(2).to_string())
    print(f"\nresults saved to {output_dir}")
    return runs
