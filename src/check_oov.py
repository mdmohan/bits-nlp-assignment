"""Find a split seed with low validation OOV for the configured corpus."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from lm.data import Vocab, load_corpus_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "src/configs/webtext.yaml"

# Change this list to try a different set of reproducible split seeds.
#SEEDS = [13, 42, 55, 123, 2000]

# SEEDS = [10,11,12,13,14]
SEEDS=[]

def read_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError(f"Expected a top-level data section in {path}")
    return config


def run_orchestrator(config_path: Path, seed: int) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "src/data-pipeline/data_orchestrator.py"),
        "--config",
        str(config_path),
        "--seed",
        str(seed),
        "--no-rare-check",
    ]
    print(f"\n=== Preparing split with seed {seed} ===", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def validation_oov(config: dict) -> tuple[float, int, int, int]:
    data = config["data"]
    corpus_name = data["corpus"]
    output_root = Path(data.get("output", "data/processed")).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    split_dir = output_root / corpus_name

    corpus = load_corpus_files(
        split_dir / "train.txt",
        split_dir / "valid.txt",
        split_dir / "test.txt",
        tokenizer_name=data.get("tokenizer", "space"),
    ).lowercase()
    vocab_config = config.get("vocab", {})
    vocab = Vocab(
        corpus.train,
        min_freq=int(vocab_config.get("min_freq", 2)),
        max_size=vocab_config.get("max_size"),
    )

    total_tokens = sum(map(len, corpus.valid))
    oov_tokens = sum(
        word not in vocab.stoi
        for sentence in corpus.valid
        for word in sentence
    )
    rate = oov_tokens / total_tokens if total_tokens else 0.0
    return rate, oov_tokens, total_tokens, len(vocab)


def main() -> int:

    config_path = CONFIG_PATH.resolve()
    config = read_config(config_path)
    results: list[tuple[int, float, int, int, int]] = []

    if len(SEEDS) > 0:
        for seed in SEEDS:
            run_orchestrator(config_path, seed)
            rate, oov_tokens, total_tokens, vocab_size = validation_oov(config)
            results.append((seed, rate, oov_tokens, total_tokens, vocab_size))
            print(
                f"seed={seed}: validation OOV {oov_tokens:,}/{total_tokens:,} "
                f"({rate:.2%}); vocabulary size={vocab_size:,}",
                flush=True,
            )

        print("\nSeed comparison (lowest validation OOV first)")
        print(f"{'seed':>8}  {'validation OOV':>18}  {'vocabulary':>12}")
        for seed, rate, oov_tokens, total_tokens, vocab_size in sorted(results, key=lambda row: row[1]):
            print(
                f"{seed:>8}  {oov_tokens:>9,}/{total_tokens:<8,} "
                f"({rate:>6.2%})  {vocab_size:>12,}"
            )

        best_seed, best_rate, best_oov, best_total, _ = min(results, key=lambda row: row[1])
        print(
            f"\nSelected seed {best_seed}: validation OOV "
            f"{best_oov:,}/{best_total:,} ({best_rate:.2%})"
        )

        # Leave the processed train/valid/test files on the best split.
        if SEEDS[-1] != best_seed:
            run_orchestrator(config_path, best_seed)
            print(f"Restored processed split for selected seed {best_seed}.")

    else:
        rate, oov_tokens, total_tokens, vocab_size = validation_oov(config)
        print(
                        f"validation OOV {oov_tokens:,}/{total_tokens:,} "
                        f"({rate:.2%}); vocabulary size={vocab_size:,}",
                        flush=True,
                    )
        

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
