"""Experiment controller: plug in any Corpus and any registered model, get comparable results.

    controller = LanguageModelController(corpus, min_freq=2, batch_size=32)
    result = controller.run("lstm", {"hidden_dim": 256}, TrainConfig(epochs=10), seed=42)

All models run by one controller share the same vocabulary and the same train/valid/test
batches, so their perplexities are directly comparable.
"""

import json
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


class LanguageModelController:
    def __init__(self, corpus: Corpus, min_freq: int = 2, max_vocab: int | None = None,
                 batch_size: int = 32, device: str = "auto", output_dir: str | None = None):
        self.corpus = corpus
        self.vocab = Vocab(corpus.train, min_freq=min_freq, max_size=max_vocab)
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
            "vocab_size": len(self.vocab),
            "valid_oov_rate": self.vocab.oov_rate(corpus.valid),
            "test_oov_rate": self.vocab.oov_rate(corpus.test),
        }
        print("data:", json.dumps(self.data_stats, indent=2))

    def run(self, model_name: str, model_params: dict, train_config: TrainConfig, seed: int) -> dict:
        """Train one model with one seed and evaluate it on the test set."""
        print(f"\n=== {model_name} | seed {seed} | device {self.device} ===")
        set_seed(seed)
        model = build_model(model_name, len(self.vocab), self.vocab.pad_id, **model_params)
        trainer = Trainer(model, train_config, self.vocab.pad_id, self.device)

        start = time.perf_counter()
        history = trainer.fit(self.train_loader, self.valid_loader)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        test = trainer.evaluate(self.test_loader)
        test_seconds = time.perf_counter() - start
        valid = trainer.evaluate(self.valid_loader)

        result = {
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

        if self.output_dir:
            run_name = f"{model_name}_seed{seed}"
            pd.DataFrame(history).to_csv(self.output_dir / f"{run_name}_history.csv", index=False)
            torch.save(model.state_dict(), self.output_dir / f"{run_name}.pt")
        return result


def run_from_config(config_path: str) -> pd.DataFrame:
    """Run every model in a YAML config for every seed; save per-run and summary tables."""
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    output_dir = Path(config.get("output_dir", "results")) / config["name"]

    data_config = config["data"]
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
    try:
        subprocess.run(
            [sys.executable, str(pipeline), "--config", str(config_path)],
            cwd=project_root,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Data preparation failed (exit code {exc.returncode}).") from exc

    processed_root = Path(data_config.get("output", "data/processed")).expanduser()
    if not processed_root.is_absolute():
        processed_root = project_root / processed_root
    corpus_dir = processed_root / data_config["corpus"]
    corpus = load_corpus_files(
        corpus_dir / "train.txt",
        corpus_dir / "valid.txt",
        corpus_dir / "test.txt",
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
    )
    train_config = TrainConfig(**training)

    results = [
        controller.run(model["name"], model.get("params", {}), train_config, seed)
        for model in config["models"]
        for seed in config.get("seeds", [42])
    ]

    runs = pd.DataFrame(results)
    summary = runs.groupby("model")[["valid_ppl", "test_ppl", "train_seconds", "test_seconds"]].agg(["mean", "std"])

    runs.to_csv(output_dir / "runs.csv", index=False)
    summary.to_csv(output_dir / "summary.csv")
    (output_dir / "data_stats.json").write_text(json.dumps(controller.data_stats, indent=2))
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir / "vocab.json").write_text(json.dumps(controller.vocab.itos))

    print("\n", summary.round(2).to_string())
    print(f"\nresults saved to {output_dir}")
    return runs
